import cv2
from ultralytics import YOLO
import time


def main():
    # Small YOLO model for fast real-time detection.
    # It will download automatically the first time.
    model = YOLO("yolo11n.pt")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        print("Try changing VideoCapture(0) to VideoCapture(1).")
        return

    print("YOLO Object Detection started.")
    print("Press Q to quit.")

    previous_time = 0

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)

        # Run YOLO detection
        results = model(frame, verbose=False)

        person_count = 0
        detected_objects = []

        for result in results:
            boxes = result.boxes

            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]

                if confidence < 0.45:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if class_name == "person":
                    person_count += 1

                detected_objects.append(class_name)

                label = f"{class_name} {confidence:.2f}"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        # FPS
        current_time = time.time()
        fps = 1 / (current_time - previous_time) if previous_time else 0
        previous_time = current_time

        cv2.putText(
            frame,
            f"People Count: {person_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("VisionX AI - YOLO Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    print("YOLO Object Detection closed.")


if __name__ == "__main__":
    main()