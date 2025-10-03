import time
from collections import deque
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from PIL import Image

from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs

# --------------------------------------------------------------------
# Data locations and camera intrinsics
# --------------------------------------------------------------------
BASE = Path(
    "/run/user/1000/gvfs/smb-share:server=unas-pro.local,share=dwe_nas/Software/"
    "scrippsDivesWithDepth/depth3"
)
RGB_DIR = BASE / "linearPNG"
DEPTH_DIR = BASE / "decoded_npy"

# K = np.array(
#     [
#         [987.46, 0.0, 830.36],
#         [0.0, 987.46, 644.75],
#         [0.0, 0.0, 1.0],
#     ],
#     dtype=np.float32,
# )

K = np.array(
    [
        [1523.27, 0.0, 798.97],
        [0.0, 1523.27, 646.51],
        [0.0,    0.0,   1.0 ],
    ],
    dtype=np.float32,
)

FRAME_IDS = sorted(RGB_DIR.glob("*.png"))
FRAME_SKIP = 60             # advance ~1 s @ 60 fps
WINDOW_SIZE = 3             # keep last 3 frames

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MapAnything.from_pretrained("facebook/map-anything").to(device).eval()

# --------------------------------------------------------------------
# Open3D viewer setup
# --------------------------------------------------------------------
pcd = o3d.geometry.PointCloud()
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Live RGB-D Map")
vis.add_geometry(pcd)

frame_window: deque[Path] = deque(maxlen=WINDOW_SIZE)

with torch.inference_mode():
    for idx in range(0, len(FRAME_IDS), FRAME_SKIP):
        rgb_path = FRAME_IDS[idx]
        depth_path = DEPTH_DIR / f"{rgb_path.stem}.npy"
        if not depth_path.exists():
            continue

        frame_window.append(rgb_path)
        # ----------------------------------------------------------------
        # Build multimodal views for the current window
        # ----------------------------------------------------------------
        views = []
        for path in frame_window:
            depth_file = DEPTH_DIR / f"{path.stem}.npy"
            if not depth_file.exists():
                continue
            rgb = np.array(Image.open(path).convert("RGB"))
            depth = np.load(depth_file).astype(np.float32)

            views.append(
                {
                    "img": rgb,
                    "intrinsics": K,
                    "depth_z": depth,
                    "is_metric_scale": torch.tensor([True], device=device),
                }
            )

        if len(views) < 1:
            continue

        processed = preprocess_inputs(views)
        # Move tensors to the same device as the model
        processed = [
            {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in view.items()}
            for view in processed
        ]

        # ----------------------------------------------------------------
        # MapAnything inference for the current window
        # ----------------------------------------------------------------
        predictions = model.infer(processed, use_amp=True)

        xyz_all = []
        rgb_all = []
        for pred in predictions:
            pts3d_world = pred["pts3d"].squeeze().cpu().numpy()
            img_rgb = pred["img_no_norm"].squeeze().cpu().numpy()
            mask = pred["mask"].squeeze().cpu().numpy().astype(bool)

            xyz_all.append(pts3d_world[mask])
            rgb_all.append(img_rgb[mask])

        if not xyz_all:
            continue

        xyz_live = np.concatenate(xyz_all, axis=0)
        rgb_live = np.concatenate(rgb_all, axis=0)

        # ----------------------------------------------------------------
        # Refresh Open3D with the latest window (overwrites previous data)
        # ----------------------------------------------------------------
        pcd.points = o3d.utility.Vector3dVector(xyz_live)
        pcd.colors = o3d.utility.Vector3dVector(rgb_live)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # Simple pacing so the visualization is readable; adjust as needed
        time.sleep(0.1)

vis.destroy_window()
