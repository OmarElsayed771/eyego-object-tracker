import cv2


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read the first frame.")
            return

        bbox = cv2.selectROI(
            "EyeGo Object Tracker",
            frame,
            fromCenter=False,
            showCrosshair=True
        )

        x, y, w, h = bbox

        if w == 0 or h == 0:
            print("Error: No valid object was selected.")
            return

        print(f"Selected bounding box: {bbox}")

        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)

        print("CSRT tracker initialized successfully.")

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            # Update tracker
            success, bbox = tracker.update(frame)

            if success:
                x, y, w, h = [int(v) for v in bbox]

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                # Tracking status
                cv2.putText(
                    frame,
                    "TRACKING",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            else:
                # Tracking failed
                cv2.putText(
                    frame,
                    "TRACKING LOST",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("EyeGo Object Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released. Application closed.")


if __name__ == "__main__":
    main()