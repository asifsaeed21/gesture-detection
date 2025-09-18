# Gesture Control for Windows (Webcam + MediaPipe + OpenCV)

Control Windows apps using hand gestures captured from your webcam.

- Index finger up → Open Google Chrome
- Two fingers up → Open VS Code
- Three fingers up → Open File Explorer
- Fist → Open Notepad

The webcam feed shows hand landmarks and the detected gesture name on-screen. A cooldown prevents repeated rapid triggers.

## Requirements
- Windows 10/11
- Python 3.12 (MediaPipe does not support Python 3.13 yet)
- Webcam

## Setup
```powershell
cd C:\gesturedetection
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.12` is not found, install Python 3.12:
```powershell
winget install -e --id Python.Python.3.12
# Restart PowerShell, then repeat setup
```

## Run
```powershell
# In the activated venv
python gesture_control.py
```

Press `Q` to quit the webcam window.

## What you should see
- A window with your webcam feed
- Green hand landmarks and connections drawn by MediaPipe
- The current gesture name (e.g., "IndexUp", "TwoFingers") rendered at the top-left
- When a gesture is recognized and stable, the corresponding app opens; a brief cooldown prevents multiple openings

## Extend the project
- Add more gestures (e.g., thumb up, open palm) by enhancing the `detect_gesture()` logic
- Map gestures to system actions (volume up/down, media control) via `subprocess` or `pycaw`
- Add on-screen UI feedback (cooldown timer, last triggered app)
- Support both hands and prefer the most confident hand
- Debounce logic using moving average of detections or majority vote over last N frames

## Troubleshooting
- No camera: Ensure no other app is using the webcam; try a different camera index in `cv2.VideoCapture(0)` (e.g., 1 or 2)
- MediaPipe install error: Use Python 3.12, not 3.13
- Apps not opening: Ensure the command paths in `perform_action()` suit your system; adjust VS Code path or use `code` if on PATH
