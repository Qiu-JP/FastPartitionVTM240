#!/usr/bin/env python3
"""Serial intra encode/decode evaluation using the ref_model anchor."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
VTM = ROOT / 'VVCSoftware_VTM'
REF = ROOT / 'ref_model'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_log(path):
    text = path.read_text(errors='replace')
    values = re.findall(r'^\s*(\d+)\s+a\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)', text, re.M)
    elapsed = re.findall(r'Total Time:\s*[\d.]+\s*sec\.\s*\[user\]\s*([\d.]+)\s*sec\.\s*\[elapsed\]', text)
    if not values or not elapsed:
        raise ValueError(f'Incomplete encoder log: {path}')
    counts = re.findall(r'^\[RdoModeStats\]\s+channel=(\S+)\s+testedModes=\d+\s+tested4x4=(\d+)', text, re.M)
    result = dict(frames=int(values[-1][0]), rd=list(map(float, values[-1][1:])), elapsed=float(elapsed[-1]),
                  compute_luma=sum(int(n) for ch,n in counts if ch=='luma') if counts else None,
                  compute_total=sum(int(n) for ch,n in counts) if counts else None)
    if not all(math.isfinite(v) for v in result['rd']+[result['elapsed']]) or result['rd'][0] <= 0:
        raise ValueError(f'Invalid RD/time metrics: {path}')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seq-list', type=Path, default=REF/'script/Testing_Sequences_VVC.txt')
    p.add_argument('--yuv-root', type=Path, default=ROOT/'data/video/VVC_CTC')
    p.add_argument('--sequence-cfg-root', type=Path, default=VTM/'cfg/per-sequence')
    p.add_argument('--only-seq')
    p.add_argument('--qps', type=int, nargs='+', default=[22,27,32,37])
    p.add_argument('--max-frames', type=int, default=1)
    p.add_argument('--main-cfg', type=Path, default=REF/'cfg/encoder_intra_vtm.cfg')
    p.add_argument('--bundle', type=Path, default=ROOT/'network/checkpoints/onnx/final', help='ONNX bundle directory; defaults to the bundled final model')
    p.add_argument('--thresholds', type=Path, required=True, help='cfg containing only FastPartitionThBySize')
    p.add_argument('--test-bin', type=Path, default=VTM/'script/bin/EncoderApp')
    p.add_argument('--test-decoder', type=Path, default=VTM/'script/bin/DecoderApp')
    p.add_argument('--anchor-bin', type=Path, default=REF/'bin/vtm240/EncoderAppStatic')
    p.add_argument('--anchor-decoder', type=Path, default=REF/'bin/vtm240/DecoderAppStatic')
    p.add_argument('--workdir', type=Path, default=VTM/'script/output'/datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--keep-yuv', action='store_true')
    p.add_argument('--keep-logs', action='store_true', help='Keep encoder/decoder logs after successful evaluation')
    args = p.parse_args()
    if args.max_frames < 1 or len(set(args.qps)) != len(args.qps) or any(q < 0 or q > 63 for q in args.qps):
        p.error('max-frames must be positive and QPs distinct integers in 0..63')
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.expanduser().resolve())
    for name in ['seq_list','main_cfg','thresholds','test_bin','test_decoder','anchor_bin','anchor_decoder']:
        if not getattr(args,name).is_file():p.error(f'Missing {name}: {getattr(args,name)}')
    for line in args.thresholds.read_text().splitlines():
        line=line.split('#',1)[0].strip()
        if line and line.split(':',1)[0].strip() != 'FastPartitionThBySize':
            p.error('threshold cfg may contain only FastPartitionThBySize')
    # Validate calibration/deployment identity before launching costly encodes.
    from collect_threshold_probabilities import ThresholdOnnx
    backend = ThresholdOnnx(args.bundle)
    sequences=[]
    for row in csv.reader(args.seq_list.read_text().splitlines()):
        if not row or not row[0].strip() or row[0].lstrip().startswith('#'):continue
        if 'end!!!!' in row[0]:break
        if len(row)!=6:raise ValueError(f'Expected six sequence columns: {row}')
        name,file,w,h,frames,fps=[v.strip() for v in row]
        if args.only_seq and name!=args.only_seq:continue
        if Path(name).name!=name or name in {'.','..'}:raise ValueError('Invalid sequence name')
        seq_cfg=args.sequence_cfg_root/f'{name}.cfg';yuv=args.yuv_root/file
        if not seq_cfg.is_file() or not yuv.is_file():raise FileNotFoundError(f'{seq_cfg} / {yuv}')
        sequences.append((name,yuv,seq_cfg,int(w),int(h),min(args.max_frames,int(frames)),int(fps)))
    if not sequences or len({s[0] for s in sequences})!=len(sequences):p.error('Empty or duplicate sequence selection')
    args.workdir.mkdir(parents=True,exist_ok=False)
    manifest={'status':'running','arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'hashes':{k:digest(getattr(args,k)) for k in ['seq_list','main_cfg','thresholds','test_bin','test_decoder','anchor_bin','anchor_decoder']},
              'model_hashes':backend.model_hashes,'runs':[]}
    def save(): (args.workdir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    save()
    # Ignore inherited experiment/dump switches. Measurements are serial, CPU ORT uses one thread.
    env={k:v for k,v in os.environ.items() if not k.startswith(('FASTPARTITION_', 'VTM_RDO_'))}
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    curves={}
    try:
        for name,yuv,seq_cfg,w,h,frames,fps in sequences:
            manifest.setdefault('inputs', {})[name] = {'path': str(yuv), 'sha256': digest(yuv)}
            save()
            curves[name]=[]
            for qp in args.qps:
                record={'qp':qp}
                for mode,encoder,decoder in [('anchor',args.anchor_bin,args.anchor_decoder),('test',args.test_bin,args.test_decoder)]:
                    d=args.workdir/name/f'QP{qp}'/mode;d.mkdir(parents=True)
                    bit=d/'stream.vvc';recon=d/'recon.yuv'
                    cmd=[str(encoder),'-c',str(args.main_cfg),'-c',str(seq_cfg),'-i',str(yuv),'-b',str(bit),'-o',str(recon),
                         f'--SourceWidth={w}',f'--SourceHeight={h}',f'--FramesToBeEncoded={frames}',f'--FrameRate={fps}',f'--QP={qp}','--Verbosity=6']
                    if mode=='test':cmd+=['-c',str(args.thresholds),f'--FastPartitionSwinModel={args.bundle}/swin.onnx',f'--FastPartitionClassifierModel={args.bundle}','--FastPartitionLumaModelScale=32']
                    run={'sequence':name,'qp':qp,'mode':mode,'command':cmd,'sequence_cfg_sha256':digest(seq_cfg)}
                    manifest['runs'].append(run);save();print(f'{name} QP{qp} {mode}',flush=True)
                    with (d/'encode.log').open('w') as log:subprocess.run(cmd,cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                    metrics=read_log(d/'encode.log')
                    if metrics['frames']!=frames:raise ValueError('Unexpected encoded frame count')
                    metrics.update(bitstream_sha256=digest(bit),recon_sha256=digest(recon))
                    record[mode]=metrics
                    if decoder:
                        dec=d/'decoded.yuv';dcmd=[str(decoder),'-b',str(bit),'-o',str(dec)]
                        denv=dict(env,FASTPARTITION_DEPTH_DIR=str(d/'partition'),FASTPARTITION_DEPTH_PREFIX=name)
                        run['decode_command']=dcmd
                        with (d/'decode.log').open('w') as log:subprocess.run(dcmd,cwd=d,env=denv,stdout=log,stderr=subprocess.STDOUT,check=True)
                        if digest(dec)!=metrics['recon_sha256']:raise ValueError('Encode/decode reconstruction mismatch')
                        if not args.keep_yuv:dec.unlink()
                    if not args.keep_yuv:recon.unlink()
                    run['metrics']=metrics;run['status']='complete';save()
                curves[name].append(record)
                (args.workdir/'curves.json').write_text(json.dumps(curves,indent=2))
        subprocess.run([sys.executable,str(VTM/'script/getBdRate.py'),str(args.workdir/'curves.json'),'--output',str(args.workdir/'summary.csv')],check=True,env=env)
        if not args.keep_logs:
            for run in manifest['runs']:
                directory=args.workdir/run['sequence']/f"QP{run['qp']}"/run['mode']
                for name in ('encode.log', 'decode.log'):
                    (directory/name).unlink(missing_ok=True)
        manifest['status']='complete';save()
    except BaseException:
        manifest['status']='failed';save();raise


if __name__=='__main__':main()
