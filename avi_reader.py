import cv2
import os
import sys

# --- Configuration ---
video_path = '/run/user/1000/gvfs/smb-share:server=unas-pro.local,share=dwe_nas/Recordings/2025/09-23-maritime-museum/AVI/stereo_2025-09-23_10-29-44_PDT_boat1_923000_part1_cam1.avi'
output_folder = '/home/tong/recordings/maritime1'
EVERY_N = 60  # save every 60th frame

# --- Setup ---
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # might be 0/unknown for some codecs
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Opened video: {video_path}")
print(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames if total_frames>0 else 'unknown'}")
print(f"Saving every {EVERY_N} frames to: {output_folder}")
print("Starting...")

def progress_bar(current, total):
    if total <= 0:  # unknown length
        sys.stdout.write(f"\rProcessed: {current:,} frames | Saved: {saved:,}")
        sys.stdout.flush()
        return
    bar_len = 30
    pct = min(current / total, 1.0)
    filled = int(bar_len * pct)
    bar = "█" * filled + "-" * (bar_len - filled)
    sys.stdout.write(f"\r[{bar}] {pct*100:6.2f}% | {current:,}/{total:,} frames | Saved: {saved:,}")
    sys.stdout.flush()

frame_idx = 0
saved = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Save every Nth frame
    if frame_idx % EVERY_N == 0:
        out_path = os.path.join(output_folder, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(out_path, frame)
        saved += 1

    frame_idx += 1

    # Update progress
    progress_bar(frame_idx, total_frames)

cap.release()
print("\nDone.")
print(f"Processed {frame_idx:,} frames, saved {saved:,} frames to '{output_folder}'.")
