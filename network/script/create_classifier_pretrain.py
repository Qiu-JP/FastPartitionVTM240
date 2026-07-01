import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paths

CLASSIFIER_I_CLASS_NAMES = {
   0: "NO_SPLIT",
   1: "QT",
   2: "BTH",
   3: "BTV",
   4: "TTH",
   5: "TTV",
}

CLASSIFIER_I_GRID_SPECS = {
   # grid_h, grid_w: allowed class ids in Classifier_I order.
   (16, 16): [0, 1],
   (8, 8): [0, 1, 2, 3, 4, 5],
   (8, 4): [0, 2, 3, 4, 5],
   (4, 8): [0, 2, 3, 4, 5],
   (8, 2): [0, 2, 3, 4],
   (2, 8): [0, 2, 3, 5],
   (8, 1): [0, 2, 4],
   (1, 8): [0, 3, 5],
   (4, 2): [0, 2, 3, 4],
   (2, 4): [0, 2, 3, 5],
   (4, 1): [0, 2, 4],
   (1, 4): [0, 3, 5],
   (4, 4): [0, 1, 2, 3, 4, 5],
   (2, 2): [0, 2, 3],
   (2, 1): [0, 2],
   (1, 2): [0, 3],
}


def Classifier_I_Logical_Gridmap(grid_h, grid_w, class_id):
   gridmap = np.zeros((2, grid_h, grid_w), dtype=np.float32)

   if class_id == 0:
      return gridmap
   if class_id == 1:
      gridmap[0, :, grid_w // 2 - 1] = 1
      gridmap[1, grid_h // 2 - 1, :] = 1
      return gridmap
   if class_id == 2:
      gridmap[1, grid_h // 2 - 1, :] = 1
      return gridmap
   if class_id == 3:
      gridmap[0, :, grid_w // 2 - 1] = 1
      return gridmap
   if class_id == 4:
      gridmap[1, grid_h // 4 - 1, :] = 1
      gridmap[1, (3 * grid_h) // 4 - 1, :] = 1
      return gridmap
   if class_id == 5:
      gridmap[0, :, grid_w // 4 - 1] = 1
      gridmap[0, :, (3 * grid_w) // 4 - 1] = 1
      return gridmap

   raise ValueError("Unsupported Classifier_I class id: {}".format(class_id))


def Create_Classifier_I_Pretrain_Logical_Dataset(samples_per_class=256):
   save_root = paths.dataset_root() / "Classifier_I" / "pretrain_logical"
   os.makedirs(save_root, exist_ok=True)

   for (grid_h, grid_w), allowed_classes in CLASSIFIER_I_GRID_SPECS.items():
      gridmaps = []
      labels = []

      for class_id in allowed_classes:
         minimal = Classifier_I_Logical_Gridmap(grid_h, grid_w, class_id)
         for _ in range(samples_per_class):
            gridmaps.append(minimal.copy())
            labels.append(class_id)

      gridmaps = np.stack(gridmaps, axis=0).astype(np.float32)
      labels = np.asarray(labels, dtype=np.int64)

      dirname = "{}x{}".format(grid_h, grid_w)
      save_dir = save_root / dirname
      os.makedirs(save_dir, exist_ok=True)

      np.save(save_dir / "gridmap.npy", gridmaps)
      np.save(save_dir / "label.npy", labels)

if __name__ == "__main__":  
    
    Create_Classifier_I_Pretrain_Logical_Dataset()
