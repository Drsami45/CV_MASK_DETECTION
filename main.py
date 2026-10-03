import sys
import cv2
from ultralytics import YOLO

def main():
    # 1. Safely load the YOLO model
    try:
        # Load small pre-trained nano model (downloads on first run if missing)
        model = YOLO("yolov8n.pt")
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model: {e}")
        sys.exit(1)

    # 2. Safely initialize webcam device
    camera_index = 0
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam at index {camera_index}. Check camera connections or permissions.")
        sys.exit(1)

    # Set camera resolution (fallback handled automatically by hardware driver)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Webcam stream active. Press 'q' or 'ESC' in the display window to exit.")

    # 3. Robust frame loop
    try:
        while True:
            ret, frame = cap.read()

            # Guard against dropped or empty frames
            if not ret or frame is None:
                print("[WARNING] Empty frame received from camera stream. Skipping...")
                continue

            # Run inference (verbose=False prevents log spamming in console)
            results = model(frame, stream=True, verbose=False)

            # Extract annotated frame safely
            annotated_frame = frame.copy()
            for r in results:
                annotated_frame = r.plot()

            # Display the result
            cv2.imshow("YOLO Real-Time Detection", annotated_frame)

            # Handle window close button (X) or keyboard quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27 or cv2.getWindowProperty("YOLO Real-Time Detection", cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print("\n[INFO] Stream stopped manually by user.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during execution: {e}")
    finally:
        # Guarantee hardware resources are released even if script crashes
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released and windows closed cleanly.")

if __name__ == "__main__":
    main()