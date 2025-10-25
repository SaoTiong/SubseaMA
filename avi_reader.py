import cv2
import tkinter as tk
from tkinter import messagebox, filedialog
import os

# --- 1. Configuration ---
# Your video file path
VIDEO_PATH = '/home/tong/recordings/Rocks_cam1.avi' 
WINDOW_NAME = 'Interactive Frame Saver'
SEEK_AMOUNT = 150 # Number of frames to jump forward or backward

# --- 2. Main Application Logic ---
def run_player():
    """
    Initializes and runs the main video player loop.
    """
    # --- Initialization ---
    root = tk.Tk()
    root.withdraw() 

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        messagebox.showerror("Error", f"Could not open video file:\n{VIDEO_PATH}")
        return
    
    # State variables
    is_paused = False
    is_saving_frames = False
    save_folder = ""
    saved_frame_count = 0

    print("--- Controls ---")
    print("  r : Start saving frames")
    print("  s : Stop saving frames")
    print("  d : Fast-forward")
    print("  a : Rewind")
    print("  q : Quit the player")
    print("----------------")

    # --- Main Loop ---
    while True:
        if not is_paused:
            success, frame = cap.read()
            if not success:
                print("End of video reached.")
                break
        
        display_frame = frame.copy()
        status_text = ""
        status_color = (0, 0, 0)
        if is_saving_frames:
            status_text = f"● SAVING FRAMES ({saved_frame_count})"
            status_color = (0, 0, 255) # Red
        elif is_paused:
            status_text = "|| PAUSED"
            status_color = (255, 165, 0) # Orange
        
        if status_text:
             cv2.putText(display_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
             cv2.putText(display_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, display_frame)

        # Handle frame saving if active
        if is_saving_frames and not is_paused:
            filename = os.path.join(save_folder, f"frame_{saved_frame_count:06d}.jpg")
            cv2.imwrite(filename, frame)
            saved_frame_count += 1

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break

        # Start Saving Frames
        elif key == ord('r'):
            if is_saving_frames:
                messagebox.showwarning("Frame Saver", "Already saving frames!")
                continue

            is_paused = True
            if messagebox.askyesno("Frame Saver", "Start saving frames?"):
                folder_path = filedialog.askdirectory(title="Select a folder to save frames")
                
                if folder_path:
                    save_folder = folder_path
                    is_saving_frames = True
                    saved_frame_count = 0
                    print(f"Saving frames to: {save_folder}")
                else:
                    print("Frame saving cancelled by user.")
            
            is_paused = False

        # Stop Saving Frames
        elif key == ord('s'):
            if not is_saving_frames:
                messagebox.showinfo("Frame Saver", "Not currently saving any frames.")
                continue

            is_paused = True
            if messagebox.askyesno("Frame Saver", f"Stop saving frames? ({saved_frame_count} frames saved)"):
                is_saving_frames = False
                print(f"Stopped saving. Total frames saved in this session: {saved_frame_count}")
            
            is_paused = False
        
        # Rewind
        elif key == ord('a'):
            if not is_saving_frames:
                current_frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                new_pos = max(0, current_frame_pos - SEEK_AMOUNT)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                print(f"Rewound to frame ~{int(new_pos)}")

        # Fast-forward
        elif key == ord('d'):
            if not is_saving_frames:
                current_frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos + SEEK_AMOUNT)
                print(f"Fast-forwarded to frame ~{int(current_frame_pos + SEEK_AMOUNT)}")

    # --- Cleanup ---
    print("Exiting player.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_player()
