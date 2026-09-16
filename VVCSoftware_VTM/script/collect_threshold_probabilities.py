"""Infer node probabilities directly from createDataset NPY/PKL and an ONNX bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import onnxruntime as ort

CLASS_NAMES = ['NO_SPLIT', 'QT', 'BTH', 'BTV', 'TTH', 'TTV']


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


class ThresholdOnnx:
    def __init__(self, bundle):
        self.bundle = Path(bundle).resolve()
        names = ['swin.onnx'] + [f'classifier_{h}x{w}.onnx'
            for h in (1, 2, 4, 8) for w in (1, 2, 4, 8) if (h, w) != (1, 1)]
        self.model_hashes = {name: digest(self.bundle / name) for name in names}
        self.options = ort.SessionOptions()
        self.options.intra_op_num_threads = 1
        self.options.inter_op_num_threads = 1
        self.options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.swin = self._session('swin.onnx')
        self.classifiers = {}

    def _session(self, name):
        return ort.InferenceSession(str(self.bundle / name), self.options, providers=['CPUExecutionProvider'])

    @staticmethod
    def _tensor(value, shape, name):
        if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != shape:
            raise ValueError(f'{name} requires float32 shape {shape}')
        if not np.all(np.isfinite(value)):
            raise ValueError(name + ' contains nonfinite values')
        return np.ascontiguousarray(value)

    def infer_batch(self, pixels, qp_normalized):
        """Infer the supplied independent 48x48 samples in their stored order."""
        batch = len(pixels)
        if batch < 1:
            raise ValueError('Empty inference batch')
        pixels = self._tensor(pixels, (batch, 1, 48, 48), 'pixels')
        qp = self._tensor(qp_normalized, (batch, 1), 'qp_normalized')
        grid = self.swin.run(['gridmap'], {'input': pixels, 'qp': qp})[0]
        return self._tensor(grid, (batch, 2, 8, 8), 'gridmap')

    def infer_roi(self, roi):
        """Raw six-class probabilities, batch=1 as in VTM; no legal renormalization."""
        if not isinstance(roi, np.ndarray) or roi.ndim != 4:
            raise ValueError('ROI must have NCHW layout')
        h, w = roi.shape[-2:]
        roi = self._tensor(roi, (1, 2, h, w), 'roi')
        if (h, w) == (1, 1):
            return np.array([[1, 0, 0, 0, 0, 0]], dtype=np.float32)
        name = f'classifier_{h}x{w}.onnx'
        if name not in self.model_hashes:
            raise ValueError(f'Unsupported pixel shape W×H={w*4}×{h*4}')
        if (h, w) not in self.classifiers:
            self.classifiers[h, w] = self._session(name)
        probabilities = self.classifiers[h, w].run(['probabilities'], {'roi': roi})[0]
        probabilities = self._tensor(probabilities, (1, 6), 'probabilities')
        if np.any(probabilities < 0) or np.any(probabilities > 1.0001) or abs(float(probabilities.sum()) - 1) > .001:
            raise ValueError('Invalid classifier probabilities')
        return probabilities

def collect_probabilities(dataset_dir, bundle, require_rd=False, batch_size=16):
    """Infer supplied samples only; no CTU recipe or missing-label synthesis."""
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    dataset = Path(dataset_dir).resolve()
    names = ['Luma_Input', 'Luma_CU_Tree']
    if require_rd or all((dataset / ('Luma_CU_RDCost' + ext)).is_file() for ext in ('.npy', '.pkl')):
        names.append('Luma_CU_RDCost')
    hashes = {name + ext: digest(dataset / (name + ext)) for name in names for ext in ('.npy', '.pkl')}
    input_meta = pd.read_pickle(dataset / 'Luma_Input.pkl')
    tree_meta = pd.read_pickle(dataset / 'Luma_CU_Tree.pkl')
    inputs = np.load(dataset / 'Luma_Input.npy', mmap_mode='r')
    nodes = np.load(dataset / 'Luma_CU_Tree.npy', mmap_mode='r')
    ids, samples = input_meta['ids'], tree_meta['samples']
    offsets = np.asarray(tree_meta['offsets'])
    if not ids.index.equals(samples.index) or not ids.index.is_unique:
        raise ValueError('Input and tree sample identities must match uniquely')
    if inputs.shape != (len(ids), 1, 48, 48) or not len(ids):
        raise ValueError('Expected nonempty Luma32 inputs [N,1,48,48]')
    if nodes.ndim != 2 or nodes.shape[1] != 5 or not np.issubdtype(nodes.dtype, np.integer):
        raise ValueError('Expected integer CU nodes [N,5]')
    if (offsets.shape != (len(ids)+1,) or not np.issubdtype(offsets.dtype, np.integer)
            or offsets[0] != 0 or offsets[-1] != len(nodes) or np.any(np.diff(offsets) <= 0)):
        raise ValueError('Invalid CU node offsets')
    for table in (ids, samples):
        if not np.array_equal(table['sample_index'].to_numpy(), np.arange(len(ids))):
            raise ValueError('Sample rows must follow array order')
    if tree_meta.get('class_order') != CLASS_NAMES:
        raise ValueError('Unexpected class order')
    if require_rd:
        rd_meta = pd.read_pickle(dataset / 'Luma_CU_RDCost.pkl')
        if rd_meta.get('class_order') != CLASS_NAMES or not np.array_equal(offsets, rd_meta['offsets']):
            raise ValueError('RD and CU tree offsets differ')
    qp_values = ids['qp'].to_numpy(dtype=np.float32)
    if not np.isfinite(qp_values).all() or np.any((qp_values < 0) | (qp_values > 63)):
        raise ValueError('Invalid sample QP')
    backend = ThresholdOnnx(bundle)
    probabilities = np.empty((len(nodes), 6), dtype=np.float32)
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        pixels = np.asarray(inputs[start:end], dtype=np.float32)
        qp = np.ascontiguousarray(qp_values[start:end, None] / np.float32(51))
        grids = backend.infer_batch(pixels, qp)
        for local, sample in enumerate(range(start, end)):
            for index in range(int(offsets[sample]), int(offsets[sample+1])):
                y, x, h, w, label = map(int, nodes[index])
                if min(y,x) < 0 or min(h,w) <= 0 or y+h > 8 or x+w > 8 or not 0 <= label < 6:
                    raise ValueError('Invalid node ROI/label at index ' + str(index))
                probabilities[index] = backend.infer_roi(grids[local:local+1, :, y:y+h, x:x+w])[0]
        if start // 1000 != end // 1000 or end == len(ids):
            print(f'ONNX samples {end}/{len(ids)}', flush=True)
    manifest = dict(status='complete', scope='dataset_samples', inference_batch=batch_size,
                    batching='stored sample order; final batch uses remaining samples',
                    deployment_ctu_batch_matched=False,
                    dataset_dir=str(dataset), bundle=str(Path(bundle).resolve()),
                    model_hashes=backend.model_hashes, dataset_hashes=hashes,
                    total_nodes=len(nodes), collected_nodes=len(nodes), samples=len(ids),
                    probabilities_sha256=hashlib.sha256(probabilities.tobytes()).hexdigest(),
                    class_order=CLASS_NAMES)
    return manifest, nodes, probabilities, np.ones(len(nodes), dtype=bool)



def load_probability_cache(directory, require_rd=False):
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest.get('status') != 'complete' or manifest.get('scope') != 'dataset_samples':
        raise ValueError('Expected a complete dataset-sample cache; regenerate legacy CTU caches')
    if manifest.get('class_order') != CLASS_NAMES:
        raise ValueError('Unexpected cache class order')
    dataset = Path(manifest['dataset_dir'])
    required = {n + ext for n in ('Luma_Input', 'Luma_CU_Tree') for ext in ('.npy', '.pkl')}
    if require_rd:
        required.update({'Luma_CU_RDCost.npy', 'Luma_CU_RDCost.pkl'})
    if not required.issubset(manifest['dataset_hashes']):
        raise ValueError('Cache does not bind all required dataset files; regenerate it with RD files present')
    for name, expected in manifest['dataset_hashes'].items():
        if digest(dataset/name) != expected:
            raise ValueError('Dataset changed: ' + name)
    for name, expected in manifest['model_hashes'].items():
        if digest(Path(manifest['bundle'])/name) != expected:
            raise ValueError('Model changed: ' + name)
    nodes = np.load(dataset/'Luma_CU_Tree.npy', mmap_mode='r')
    probability = np.load(directory/'probabilities.npy', mmap_mode='r')
    if probability.shape != (len(nodes), 6) or probability.dtype != np.float32:
        raise ValueError('Invalid probability dimensions or dtype')
    if hashlib.sha256(probability.tobytes()).hexdigest() != manifest['probabilities_sha256']:
        raise ValueError('Probability cache changed')
    if (manifest['total_nodes'] != len(nodes) or manifest['collected_nodes'] != len(nodes)
            or not np.isfinite(probability).all() or np.any(probability < 0)
            or np.any(probability > 1.0001) or not np.allclose(probability.sum(axis=1), 1, atol=.001, rtol=0)):
        raise ValueError('Incomplete or invalid probabilities')
    if require_rd:
        tree = pd.read_pickle(dataset/'Luma_CU_Tree.pkl')
        rd = pd.read_pickle(dataset/'Luma_CU_RDCost.pkl')
        if rd.get('class_order') != CLASS_NAMES or not np.array_equal(tree['offsets'], rd['offsets']):
            raise ValueError('RD and tree metadata disagree')
    return manifest, nodes, probability, np.ones(len(nodes), dtype=bool)


def search_probabilities(args, require_rd=False):
    if args.probability_cache:
        if args.dataset_dir or args.bundle:
            raise ValueError('Use --probability-cache OR --dataset-dir with --bundle')
        return load_probability_cache(args.probability_cache, require_rd=require_rd)
    if not args.dataset_dir or not args.bundle:
        raise ValueError('Provide --probability-cache, or both --dataset-dir and --bundle')
    return collect_probabilities(args.dataset_dir, args.bundle, require_rd=require_rd, batch_size=args.batch_size)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', required=True, help='Directory containing createDataset NPY/PKL files')
    parser.add_argument('--bundle', required=True, help='ONNX bundle directory')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError(out)
    manifest, _, probability, _ = collect_probabilities(args.dataset_dir, args.bundle, batch_size=args.batch_size)
    out.mkdir(parents=True)
    np.save(out/'probabilities.npy', probability)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
