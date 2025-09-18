import time
import os
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# MediaPipe Hands
try:
	import mediapipe as mp
except ImportError as e:
	raise SystemExit("MediaPipe is required. Install with: pip install mediapipe") from e

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


@dataclass
class Cooldown:
	duration_seconds: float = 2.0
	_last_trigger_time: float = 0.0

	def ready(self) -> bool:
		return (time.time() - self._last_trigger_time) >= self.duration_seconds

	def stamp(self) -> None:
		self._last_trigger_time = time.time()


# ----------------------------
# Gesture Detection Utilities
# ----------------------------

def _count_extended_fingers(landmarks: List[Tuple[float, float]], handedness: str) -> Tuple[int, List[bool]]:
	"""
	Determine which fingers are extended using landmarks.
	Returns (count, [thumb, index, middle, ring, pinky]) booleans.

	Heuristics:
	- For index/middle/ring/pinky: fingertip y is above PIP y (in image coords, smaller y is 'up').
	- For thumb: use x comparison due to sideways orientation; depends on handedness.
	"""
	# MediaPipe landmark indices
	TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
	PIP = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}

	thumb_tip_x = landmarks[TIP["thumb"]][0]
	thumb_ip_x = landmarks[3][0]  # thumb IP joint index 3

	# Handedness label is 'Left' or 'Right'. Determine when thumb is extended.
	if handedness == "Right":
		thumb_extended = thumb_tip_x < thumb_ip_x  # right hand thumb to the left when extended
	else:
		thumb_extended = thumb_tip_x > thumb_ip_x  # left hand thumb to the right when extended

	index_extended = landmarks[TIP["index"]][1] < landmarks[PIP["index"]][1]
	middle_extended = landmarks[TIP["middle"]][1] < landmarks[PIP["middle"]][1]
	ring_extended = landmarks[TIP["ring"]][1] < landmarks[PIP["ring"]][1]
	pinky_extended = landmarks[TIP["pinky"]][1] < landmarks[PIP["pinky"]][1]

	fingers = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]
	return sum(fingers), fingers


def detect_gesture(landmarks: List[Tuple[float, float]], handedness: str) -> str:
	"""
	Detects the gesture name based on hand landmarks in image pixel coordinates.
	Returns one of: 'IndexUp', 'TwoFingers', 'ThreeFingers', 'FourFingers', or 'Unknown'.
	"""
	count, fingers = _count_extended_fingers(landmarks, handedness)

	thumb, index_f, middle_f, ring_f, pinky_f = fingers

	# Index up: only index extended
	if count == 1 and index_f:
		return "IndexUp"

	# Two fingers up: index + middle
	if count == 2 and index_f and middle_f and not ring_f and not pinky_f:
		return "TwoFingers"

	# Three fingers up: index + middle + ring
	if count == 3 and index_f and middle_f and ring_f and not pinky_f:
		return "ThreeFingers"

	# Four fingers up (no thumb): index + middle + ring + pinky
	if count == 4 and not thumb and index_f and middle_f and ring_f and pinky_f:
		return "FourFingers"

	return "Unknown"


# ----------------------------
# Actions for each gesture
# ----------------------------

def _open_with_start(command: str) -> None:
	"""Use 'start' via cmd to open Windows apps reliably."""
	subprocess.Popen(["cmd", "/c", "start", "", command], shell=False)


def perform_action(gesture: str) -> Optional[str]:
	"""
	Opens corresponding application based on gesture.
	- IndexUp → Google Chrome
	- TwoFingers → YouTube (in Chrome)
	- ThreeFingers → File Explorer
	- FourFingers → Notepad
	Returns a short status string if an action was triggered.
	"""
	gesture = gesture or ""
	if gesture == "IndexUp":
		# Try chrome; fall back to msedge if needed by changing command
		_open_with_start("chrome")
		return "Opening Chrome"
	elif gesture == "TwoFingers":
		# Open YouTube directly in Chrome
		_open_with_start("chrome https://www.youtube.com")
		return "Opening YouTube"
	elif gesture == "ThreeFingers":
		_open_with_start("explorer")
		return "Opening File Explorer"
	elif gesture == "FourFingers":
		_open_with_start("notepad")
		return "Opening Notepad"
	return None


# ----------------------------
# Main loop
# ----------------------------

def _landmarks_to_pixels(hand_landmarks, image_width: int, image_height: int) -> List[Tuple[float, float]]:
	points: List[Tuple[float, float]] = []
	for lm in hand_landmarks.landmark:
		x = lm.x * image_width
		y = lm.y * image_height
		points.append((x, y))
	return points


def main() -> None:
	"""Runs webcam loop, displays feed, detects gestures, and triggers actions with cooldown."""
	cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

	cooldown = Cooldown(duration_seconds=2.0)
	label_text = ""
	label_color = (50, 220, 50)

	with mp_hands.Hands(
		model_complexity=1,
		max_num_hands=1,
		min_detection_confidence=0.6,
		min_tracking_confidence=0.6,
	) as hands:
		while True:
			ret, frame = cap.read()
			if not ret:
				break

			# Convert BGR to RGB for MediaPipe
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			rgb.flags.writeable = False
			results = hands.process(rgb)
			rgb.flags.writeable = True

			image_h, image_w = frame.shape[:2]

			current_gesture = "Unknown"
			if results.multi_hand_landmarks and results.multi_handedness:
				# Use the first detected hand
				hand_landmarks = results.multi_hand_landmarks[0]
				hand_label = results.multi_handedness[0].classification[0].label  # 'Left' or 'Right'

				# Draw landmarks
				mp_drawing.draw_landmarks(
					frame,
					hand_landmarks,
					mp_hands.HAND_CONNECTIONS,
					mp_styles.get_default_hand_landmarks_style(),
					mp_styles.get_default_hand_connections_style(),
				)

				# Convert to pixels and detect gesture
				points = _landmarks_to_pixels(hand_landmarks, image_w, image_h)
				current_gesture = detect_gesture(points, hand_label)

			# Debounce / cooldown for actions (only 1,2,3,4 fingers)
			if current_gesture in {"IndexUp", "TwoFingers", "ThreeFingers", "FourFingers"}:
				if cooldown.ready():
					status = perform_action(current_gesture)
					if status:
						label_text = f"{current_gesture} - {status}"
						cooldown.stamp()
				else:
					label_text = f"{current_gesture} (cooldown)"
			else:
				label_text = current_gesture

			# Draw UI label
			cv2.rectangle(frame, (10, 10), (460, 60), (0, 0, 0), thickness=-1)
			cv2.putText(frame, label_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color, 2, cv2.LINE_AA)

			cv2.imshow("Gesture Control", frame)

			key = cv2.waitKey(1) & 0xFF
			if key == ord('q') or key == ord('Q'):
				break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
