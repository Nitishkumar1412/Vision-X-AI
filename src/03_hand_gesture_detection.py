import cv2
import mediapipe as mp
import time


# =========================
# MediaPipe Setup
# =========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# =========================
# Gesture Logic
# =========================
def count_fingers_up(hand_landmarks):
    """
    Counts which fingers are up based on landmark positions.
    Returns a dictionary of finger states.
    """

    landmarks = hand_landmarks.landmark

    fingers = {
        "thumb": False,
        "index": False,
        "middle": False,
        "ring": False,
        "pinky": False,
    }

    # For index, middle, ring, pinky:
    # finger is up if tip is higher than PIP joint
    # y is smaller when point is higher in the image
    fingers["index"] = landmarks[8].y < landmarks[6].y
    fingers["middle"] = landmarks[12].y < landmarks[10].y
    fingers["ring"] = landmarks[16].y < landmarks[14].y
    fingers["pinky"] = landmarks[20].y < landmarks[18].y

    # Simple thumb-up detection:
    # thumb tip is higher than thumb IP joint
    fingers["thumb"] = landmarks[4].y < landmarks[3].y

    return fingers


def recognize_gesture(hand_landmarks):
    fingers = count_fingers_up(hand_landmarks)

    thumb = fingers["thumb"]
    index = fingers["index"]
    middle = fingers["middle"]
    ring = fingers["ring"]
    pinky = fingers["pinky"]

    fingers_up_count = sum([thumb, index, middle, ring, pinky])

    # Open Palm: all fingers up
    if fingers_up_count == 5:
        return "Open Palm"

    # Fist: no fingers up
    if fingers_up_count == 0:
        return "Fist"

    # Peace Sign: index and middle up only
    if index and middle and not ring and not pinky:
        return "Peace Sign"

    # Thumbs Up: thumb up and other fingers down
    if thumb and not index and not middle and not ring and not pinky:
        return "Thumbs Up"

    # Index Finger: only index up
    if index and not middle and not ring and not pinky:
        return "Index Finger"

    return "Unknown"


# =========================
# Main Camera Loop
# =========================
def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        print("Try changing VideoCapture(0) to VideoCapture(1).")
        return

    print("Hand Gesture Detection started.")
    print("Press Q to quit.")

    previous_time = 0

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        detected_gesture = "No Hand"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                detected_gesture = recognize_gesture(hand_landmarks)

        # FPS calculation
        current_time = time.time()
        fps = 1 / (current_time - previous_time) if previous_time else 0
        previous_time = current_time

        # Display gesture
        cv2.putText(
            frame,
            f"Gesture: {detected_gesture}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        # Display FPS
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("VisionX AI - Hand Gesture Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    hands.close()

    print("Hand Gesture Detection closed.")


if __name__ == "__main__":
    main()