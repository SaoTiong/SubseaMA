"""Utility to playback RGB frames and decoded depth previews side-by-side."""

from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(
    "/run/user/1000/gvfs/smb-share:server=unas-pro.local,share=dwe_nas/Software/"
    "scrippsDivesWithDepth/depth3"
)

RGB_DIR = BASE_DIR / "linearPNG"
DEPTH_PREVIEW_DIR = BASE_DIR / "decoded_preview_16bit"


def load_pairs():
    """Yield matching rgb/depth preview frame pairs sorted by name."""
    rgb_paths = sorted(RGB_DIR.glob("*.png"))
    for rgb_path in rgb_paths:
        depth_path = DEPTH_PREVIEW_DIR / rgb_path.name
        if depth_path.exists():
            yield rgb_path, depth_path


def depth_to_colormap(depth_img: np.ndarray) -> np.ndarray:
    """Convert 16-bit depth preview image to a color-mapped BGR image for display."""
    if depth_img.ndim == 3:
        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
    depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)


def main():
    pairs = list(load_pairs())
    if not pairs:
        raise RuntimeError("No matching rgb/depth preview frames found.")

    window_name = "RGB (left) | Depth (right)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    paused = False
    idx = 0
    while idx < len(pairs):
        rgb_path, depth_path = pairs[idx]
        rgb = cv2.imread(rgb_path.as_posix())
        depth_raw = cv2.imread(depth_path.as_posix(), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth_raw is None:
            idx += 1
            continue

        depth_color = depth_to_colormap(depth_raw)
        if depth_color.shape[:2] != rgb.shape[:2]:
            depth_color = cv2.resize(depth_color, (rgb.shape[1], rgb.shape[0]))

        combined = np.hstack([rgb, depth_color])

        label = f"Frame {idx + 1}/{len(pairs)}"
        cv2.putText(
            combined,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

        cv2.imshow(window_name, combined)
        key = cv2.waitKey(0 if paused else 40) & 0xFF

        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("d"):
            idx = min(idx + 1, len(pairs) - 1)
        elif key == ord("a"):
            idx = max(idx - 1, 0)
        else:
            idx += 0 if paused else 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

