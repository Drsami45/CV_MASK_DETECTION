##Real-Time Object Detection with YOLOv8 & OpenCV##

A fast, exception-safe computer vision application built with Python, OpenCV, and Ultralytics YOLOv8 for real-time webcam object detection.

##Features

Real-Time Inference: Uses the lightweight YOLOv8 Nano model (yolov8n.pt) for low-latency object detection on standard CPUs.

Robust Hardware Handling: Auto-detects platform video drivers (CAP_DSHOW on Windows) and handles camera errors smoothly.

Auto-Recovery: Guards against empty or dropped frames during live streaming.

Clean Exit: Allows quitting via keyboard shortcuts (q, ESC) or by closing the window directly without terminal crashes.

Project Structure

CV_MASK_DETECTION/
│
├── main.py          # Main application script
├── yolov8n.pt       # Pre-trained YOLOv8 weights (auto-downloaded on first run)
└── README.md        # Project documentation


Setup & Installation

1. Clone the Repository

git clone https://github.com/Drsami45/CV_MASK_DETECTION.git
cd CV_MASK_DETECTION


2. Set Up Virtual Environment (Optional)

# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate


3. Install Required Libraries

pip install ultralytics opencv-python


Usage

Run the main Python script:

python main.py


Press q or ESC in the video window to quit the application cleanly.
