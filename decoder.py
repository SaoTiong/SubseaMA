from pathlib import Path
import gc


import cv2
import numpy as np

from tqdm import tqdm


BASE_DIR = Path(
    "/run/user/1000/gvfs/smb-share:server=unas-pro.local,share=dwe_nas/Software/"
    "scrippsDivesWithDepth/depth3"
)

img_dir = BASE_DIR / "linearPNG"
depth_dir = BASE_DIR / "encoded"

imgs = list(img_dir.glob("*.png"))
depths = list(depth_dir.glob("*.png"))

decoded_dir = BASE_DIR / "decoded_npy"
preview_dir = BASE_DIR / "decoded_preview_16bit"

decoded_dir.mkdir(exist_ok=True)
preview_dir.mkdir(exist_ok=True)

img_map = {}



for file in imgs:

    img_map[file.stem] = file

depth_map = {}

for file in depths:
    
    key = file.stem# [5:]
    
    depth_map[key] = file

inter = set(img_map.keys()).intersection(set(depth_map.keys()))

# takes in a depth image path and decodes as needed.
def load_and_decode_depth(depth_path: Path):

    depth_uint8: np.ndarray = cv2.cvtColor(cv2.imread(depth_path.as_posix()), cv2.COLOR_RGB2BGR)

    depth_uint8 = depth_uint8.astype(float)
    out = depth_uint8[...,0]*255*255 + depth_uint8[...,1]*255 + depth_uint8[...,2]
    

    return out/float(1000)

#   REMEMBER TO RUN DECODE DEPTH ON DEPTH IMAGE.

for depth_png in sorted(depth_dir.glob("*.png")):
    key = depth_png.stem
    if (img_dir / f"{key}.png").exists():  # keep only matching RGB frames
        decoded = load_and_decode_depth(depth_png)

        np.save(decoded_dir / f"{key}.npy", decoded.astype(np.float32))

        scaled = np.clip(decoded * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite((preview_dir / f"{key}.png").as_posix(), scaled)
