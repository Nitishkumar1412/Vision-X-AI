import cv2
import mediapipe as mp
from ultralytics import YOLO
import time
import csv
import copy
import numpy as np
from pathlib import Path
from datetime import datetime


# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]

SCREENSHOTS_DIR = BASE_DIR / "screenshots"
LOGS_DIR = BASE_DIR / "logs"
RECORDINGS_DIR = BASE_DIR / "recordings"

SCREENSHOTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
RECORDINGS_DIR.mkdir(exist_ok=True)

EVENT_LOG_PATH = LOGS_DIR / "vision_events.csv"


# =========================
# Event Logger
# =========================
def initialize_log_file():
    if not EVENT_LOG_PATH.exists():
        with open(EVENT_LOG_PATH, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "event_type", "details"])


def log_event(event_type, details):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(EVENT_LOG_PATH, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, event_type, details])


# =========================
# MediaPipe Setup
# =========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)


# =========================
# Color Palette (cycled with gestures / keys)
# =========================
COLOR_PALETTE = [
    ("Magenta", (255, 0, 255)),
    ("Cyan", (255, 255, 0)),
    ("Yellow", (0, 255, 255)),
    ("Green", (0, 255, 0)),
    ("Red", (0, 0, 255)),
    ("White", (255, 255, 255)),
]


# =========================
# Gesture Recognition
# =========================
def count_fingers_up(hand_landmarks):
    landmarks = hand_landmarks.landmark

    fingers = {
        "thumb": landmarks[4].y < landmarks[3].y,
        "index": landmarks[8].y < landmarks[6].y,
        "middle": landmarks[12].y < landmarks[10].y,
        "ring": landmarks[16].y < landmarks[14].y,
        "pinky": landmarks[20].y < landmarks[18].y,
    }

    return fingers


def recognize_gesture(hand_landmarks):
    fingers = count_fingers_up(hand_landmarks)

    thumb, index, middle, ring, pinky = (
        fingers["thumb"], fingers["index"], fingers["middle"],
        fingers["ring"], fingers["pinky"],
    )

    fingers_up_count = sum([thumb, index, middle, ring, pinky])

    if fingers_up_count == 5:
        return "Open Palm"
    if fingers_up_count == 0:
        return "Fist"
    if index and middle and not ring and not pinky:
        return "Peace Sign"
    if thumb and not index and not middle and not ring and not pinky:
        return "Thumbs Up"
    if index and not middle and not ring and not pinky:
        return "Index Finger"
    if index and middle and ring and not pinky and not thumb:
        return "Three Fingers"
    if thumb and pinky and not index and not middle and not ring:
        return "Call Me"
    return "Unknown"


def get_landmark_point(hand_landmarks, idx, frame_width, frame_height):
    lm = hand_landmarks.landmark[idx]
    return int(lm.x * frame_width), int(lm.y * frame_height)


def get_pinch_distance(hand_landmarks, frame_width, frame_height):
    x1, y1 = get_landmark_point(hand_landmarks, 4, frame_width, frame_height)
    x2, y2 = get_landmark_point(hand_landmarks, 8, frame_width, frame_height)
    return int(np.hypot(x2 - x1, y2 - y1)), ((x1 + x2) // 2, (y1 + y2) // 2)


# =========================
# Screenshot / Recording Actions
# =========================
def save_screenshot(frame):
    filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    screenshot_path = SCREENSHOTS_DIR / filename
    cv2.imwrite(str(screenshot_path), frame)
    log_event("screenshot_saved", filename)
    return filename


def flash_effect(frame):
    """Briefly whiten the frame to simulate a camera flash."""
    flash = np.full_like(frame, 255)
    return cv2.addWeighted(frame, 0.4, flash, 0.6, 0)


# =========================
# YOLO Loader
# =========================
def load_yolo_model():
    try:
        print("Loading YOLO11n model...")
        return YOLO("yolo11n.pt")
    except Exception:
        print("Could not load yolo11n.pt. Trying yolov8n.pt...")
        return YOLO("yolov8n.pt")


# =========================
# Transparent Rounded HUD Panel
# =========================
def draw_rounded_transparent_panel(frame, x1, y1, x2, y2, radius=18, alpha=0.30, color=(15, 15, 15)):
    """
    Draws a rounded-corner semi-transparent panel directly onto `frame`
    without ever painting a hard opaque black rectangle. Only the panel
    region is blended; everything outside is untouched (fully transparent).
    """
    panel_w, panel_h = x2 - x1, y2 - y1
    if panel_w <= 0 or panel_h <= 0:
        return frame

    mask = np.zeros((panel_h, panel_w), dtype=np.uint8)
    cv2.rectangle(mask, (radius, 0), (panel_w - radius, panel_h), 255, -1)
    cv2.rectangle(mask, (0, radius), (panel_w, panel_h - radius), 255, -1)
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    cv2.circle(mask, (panel_w - radius, radius), radius, 255, -1)
    cv2.circle(mask, (radius, panel_h - radius), radius, 255, -1)
    cv2.circle(mask, (panel_w - radius, panel_h - radius), radius, 255, -1)

    roi = frame[y1:y2, x1:x2]
    color_layer = np.full_like(roi, color)
    blended = cv2.addWeighted(color_layer, alpha, roi, 1 - alpha, 0)

    mask_3c = cv2.merge([mask, mask, mask]).astype(float) / 255.0
    roi[:] = (blended * mask_3c + roi * (1 - mask_3c)).astype(np.uint8)

    # subtle border for definition without opacity
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 200), 1, lineType=cv2.LINE_AA)

    return frame


# =========================
# Main App
# =========================
def main():
    initialize_log_file()

    print("Starting AI Vision Control Center...")
    print("Loading YOLO model...")

    model = load_yolo_model()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        print("Try changing VideoCapture(0) to VideoCapture(1).")
        return

    print("Camera started.")
    print("Controls:")
    print("Open Palm     -> Active Mode / Pause Drawing")
    print("Index Finger  -> Draw Mode")
    print("Peace Sign    -> Take Screenshot")
    print("Thumbs Up     -> Next Brush Color")
    print("Call Me       -> Toggle Eraser Mode")
    print("Pinch (Thumb+Index, Three Fingers) -> Adjust Brush Size")
    print("Fist          -> Hold to Exit")
    print("Keys: C clear | Z undo | O toggle detection | H toggle landmarks")
    print("Keys: R toggle recording | [ / ] confidence | + / - thickness | Q quit")

    log_event("system_started", "AI Vision Control Center started")

    previous_time = 0
    fps_history = []

    last_screenshot_time = 0
    screenshot_cooldown = 3

    last_object_log_time = 0
    object_log_cooldown = 5

    fist_start_time = None
    fist_hold_seconds_to_exit = 1.5

    current_status = "Running"
    flash_frames_remaining = 0

    # Toggles
    detection_enabled = True
    landmarks_visible = True
    eraser_mode = False
    recording = False
    video_writer = None

    confidence_threshold = 0.45

    # Drawing state
    drawing_canvas = None
    canvas_history = []
    max_history = 15
    previous_draw_point = None
    color_index = 0
    drawing_color_name, drawing_color = COLOR_PALETTE[color_index]
    brush_thickness = 8
    last_color_switch_time = 0
    color_switch_cooldown = 0.8

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape

        if drawing_canvas is None:
            drawing_canvas = np.zeros_like(frame)

        current_time = time.time()

        # =========================
        # YOLO Object Detection
        # =========================
        person_count = 0
        detected_objects = []

        if detection_enabled:
            results = model(frame, verbose=False)

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]

                    if confidence < confidence_threshold:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if class_name == "person":
                        person_count += 1

                    detected_objects.append(class_name)
                    label = f"{class_name} {confidence:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )

            if detected_objects and current_time - last_object_log_time >= object_log_cooldown:
                unique_objects = sorted(set(detected_objects))
                log_event(
                    "objects_detected",
                    f"objects={unique_objects}, people_count={person_count}",
                )
                last_object_log_time = current_time

        # =========================
        # Hand Gesture Detection
        # =========================
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb_frame)

        detected_gesture = "No Hand"
        index_finger_point = None
        pinch_distance = None
        pinch_midpoint = None

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                if landmarks_visible:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                detected_gesture = recognize_gesture(hand_landmarks)
                index_finger_point = get_landmark_point(hand_landmarks, 8, frame_width, frame_height)
                pinch_distance, pinch_midpoint = get_pinch_distance(hand_landmarks, frame_width, frame_height)

        # =========================
        # Smart Actions
        # =========================
        action_message = "No Action"

        if detected_gesture == "Open Palm":
            current_status = "Active Mode"
            action_message = "Drawing Paused / System Active"
            previous_draw_point = None
            fist_start_time = None

        elif detected_gesture == "Index Finger":
            current_status = "Drawing Mode"
            action_message = f"Drawing in {drawing_color_name}" if not eraser_mode else "Erasing"
            fist_start_time = None

            if index_finger_point is not None:
                x, y = index_finger_point
                draw_color = (0, 0, 0) if eraser_mode else drawing_color
                draw_thickness = brush_thickness * 3 if eraser_mode else brush_thickness

                cv2.circle(frame, (x, y), 10, draw_color if not eraser_mode else (200, 200, 200), -1)

                if previous_draw_point is not None:
                    if eraser_mode:
                        cv2.line(drawing_canvas, previous_draw_point, (x, y), (0, 0, 0), draw_thickness)
                    else:
                        cv2.line(drawing_canvas, previous_draw_point, (x, y), draw_color, draw_thickness)

                previous_draw_point = (x, y)

        elif detected_gesture == "Three Fingers" and pinch_distance is not None:
            current_status = "Brush Size Mode"
            brush_thickness = int(np.clip(pinch_distance / 6, 2, 40))
            action_message = f"Brush Size: {brush_thickness}"
            previous_draw_point = None
            fist_start_time = None

        elif detected_gesture == "Thumbs Up":
            current_status = "Color Select"
            if current_time - last_color_switch_time >= color_switch_cooldown:
                color_index = (color_index + 1) % len(COLOR_PALETTE)
                drawing_color_name, drawing_color = COLOR_PALETTE[color_index]
                last_color_switch_time = current_time
            action_message = f"Color -> {drawing_color_name}"
            previous_draw_point = None
            fist_start_time = None

        elif detected_gesture == "Call Me":
            current_status = "Mode Toggle"
            if current_time - last_color_switch_time >= color_switch_cooldown:
                eraser_mode = not eraser_mode
                last_color_switch_time = current_time
            action_message = "Eraser ON" if eraser_mode else "Eraser OFF"
            previous_draw_point = None
            fist_start_time = None

        elif detected_gesture == "Peace Sign":
            previous_draw_point = None
            fist_start_time = None

            if current_time - last_screenshot_time >= screenshot_cooldown:
                screenshot_frame = cv2.addWeighted(frame, 1, drawing_canvas, 1, 0)
                filename = save_screenshot(screenshot_frame)
                action_message = f"Screenshot Saved: {filename}"
                last_screenshot_time = current_time
                flash_frames_remaining = 3
            else:
                action_message = "Screenshot Cooldown"

        elif detected_gesture == "Fist":
            previous_draw_point = None

            if fist_start_time is None:
                fist_start_time = current_time

            hold_duration = current_time - fist_start_time
            action_message = f"Hold Fist to Exit: {hold_duration:.1f}s"

            if hold_duration >= fist_hold_seconds_to_exit:
                log_event("system_exit", "Exited by Fist gesture")
                print("Fist held. Closing camera...")
                break

        else:
            previous_draw_point = None
            fist_start_time = None

        # =========================
        # Apply Drawing Canvas (black areas add nothing -> naturally transparent)
        # =========================
        frame = cv2.addWeighted(frame, 1, drawing_canvas, 1, 0)

        if flash_frames_remaining > 0:
            frame = flash_effect(frame)
            flash_frames_remaining -= 1

        # =========================
        # FPS (smoothed)
        # =========================
        instant_fps = 1 / (current_time - previous_time) if previous_time else 0
        previous_time = current_time
        fps_history.append(instant_fps)
        if len(fps_history) > 15:
            fps_history.pop(0)
        fps = sum(fps_history) / len(fps_history) if fps_history else 0

        # =========================
        # Transparent Rounded HUD Panel
        # =========================
        frame = draw_rounded_transparent_panel(
            frame, 12, 12, 480, 252, radius=16, alpha=0.32, color=(10, 10, 10)
        )

        text_color_primary = (0, 255, 220)
        text_color_secondary = (235, 235, 235)

        cv2.putText(frame, "AI VISION CONTROL CENTER", (28, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, text_color_primary, 2, cv2.LINE_AA)
        cv2.putText(frame, f"People: {person_count}", (28, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color_secondary, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Gesture: {detected_gesture}", (28, 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color_secondary, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Status: {current_status}", (28, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color_secondary, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Action: {action_message}", (28, 146),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color_secondary, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Brush: {drawing_color_name} | Size {brush_thickness} | Eraser {'ON' if eraser_mode else 'OFF'}",
                    (28, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color_secondary, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Detection: {'ON' if detection_enabled else 'OFF'} (conf {confidence_threshold:.2f}) | REC {'●' if recording else '○'}",
                    (28, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color_secondary, 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {int(fps)} | C clear | Z undo | Q quit",
                    (28, 224), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color_secondary, 1, cv2.LINE_AA)

        # Small color swatch indicator
        cv2.circle(frame, (450, 60), 10, drawing_color, -1)
        cv2.circle(frame, (450, 60), 10, (255, 255, 255), 1)

        # Recording indicator dot (top-right of frame)
        if recording:
            cv2.circle(frame, (frame_width - 30, 30), 8, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (frame_width - 70, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if recording and video_writer is not None:
            video_writer.write(frame)

        cv2.imshow("VisionX AI", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            log_event("system_exit", "Exited by keyboard Q")
            break

        elif key == ord("c"):
            canvas_history.append(drawing_canvas.copy())
            if len(canvas_history) > max_history:
                canvas_history.pop(0)
            drawing_canvas = np.zeros_like(frame)
            previous_draw_point = None
            log_event("drawing_cleared", "Drawing canvas cleared")
            print("Drawing cleared.")

        elif key == ord("z"):
            if canvas_history:
                drawing_canvas = canvas_history.pop()
                print("Undo: restored previous canvas.")
            else:
                print("Nothing to undo.")

        elif key == ord("o"):
            detection_enabled = not detection_enabled
            log_event("detection_toggled", f"enabled={detection_enabled}")
            print(f"Object detection {'enabled' if detection_enabled else 'disabled'}.")

        elif key == ord("h"):
            landmarks_visible = not landmarks_visible
            print(f"Hand landmarks {'shown' if landmarks_visible else 'hidden'}.")

        elif key == ord("r"):
            recording = not recording
            if recording:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                rec_path = RECORDINGS_DIR / f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                video_writer = cv2.VideoWriter(str(rec_path), fourcc, 20.0, (frame_width, frame_height))
                log_event("recording_started", rec_path.name)
                print(f"Recording started: {rec_path.name}")
            else:
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                log_event("recording_stopped", "Recording stopped")
                print("Recording stopped.")

        elif key == ord("["):
            confidence_threshold = round(max(0.10, confidence_threshold - 0.05), 2)
            print(f"Confidence threshold: {confidence_threshold}")

        elif key == ord("]"):
            confidence_threshold = round(min(0.95, confidence_threshold + 0.05), 2)
            print(f"Confidence threshold: {confidence_threshold}")

        elif key in (ord("+"), ord("=")):
            brush_thickness = min(40, brush_thickness + 1)

        elif key == ord("-"):
            brush_thickness = max(2, brush_thickness - 1)

    camera.release()
    cv2.destroyAllWindows()
    hands.close()

    if video_writer is not None:
        video_writer.release()

    log_event("system_closed", "Camera closed")
    print("VisionX AI closed.")


if __name__ == "__main__":
    main()