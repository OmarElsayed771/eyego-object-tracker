import cv2
from ultralytics import YOLO

from association import calculate_iou


YOLO_INTERVAL = 10
YOLO_CONFIDENCE = 0.40
IOU_THRESHOLD = 0.30

MAX_VERIFICATION_FAILURES = 3
verification_failures = 0


def create_csrt_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()

    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()

    raise RuntimeError(
        "CSRT tracker is not available in this OpenCV installation."
    )



def recover_tracker(tracker, frame, target_bbox):
    """
    Reinitialize CSRT using a new target bounding box.
    """

    tracker = create_csrt_tracker()
    tracker.init(frame, target_bbox)

    return tracker



def main():
    # Load YOLO
    model = YOLO("yolo11n.pt")

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        # --------------------------------------------------
        # 1. Capture first frame
        # --------------------------------------------------
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read first frame.")
            return

        # --------------------------------------------------
        # 2. User selects target
        # --------------------------------------------------
        user_roi = cv2.selectROI(
            "Select Target",
            frame,
            fromCenter=False,
            showCrosshair=True,
        )

        cv2.destroyWindow("Select Target")

        x, y, w, h = user_roi

        if w == 0 or h == 0:
            print("Error: No valid object selected.")
            return

        print(f"User ROI: {user_roi}")

        # 3. Run YOLO on the first frame to identify the target
        results = model(
            frame,
            conf=YOLO_CONFIDENCE,
            verbose=False,
        )

        detections = results[0].boxes

        best_iou = 0.0
        target_class_id = None
        target_class_name = None

        for box in detections:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detection_bbox = (
                int(x1),
                int(y1),
                int(x2 - x1),
                int(y2 - y1),
            )

            iou = calculate_iou(user_roi, detection_bbox)

            if iou > best_iou:
                best_iou = iou
                target_class_id = int(box.cls[0].item())
                target_class_name = model.names[target_class_id]

        if target_class_id is None:
            print("Warning: YOLO could not identify the selected object.")
            print("Tracking will continue with CSRT only.")
        else:
            print(
                f"Target identified: {target_class_name} "
                f"(class_id={target_class_id}, IoU={best_iou:.2f})"
            )
        # --------------------------------------------------
        # 3. Initialize CSRT
        # --------------------------------------------------
        tracker = create_csrt_tracker()
        tracker.init(frame, user_roi)

        print("CSRT tracker initialized.")

        frame_count = 0
        verification_failures = 0
        tracking_status = "TRACKING"
        verification_status = "NOT VERIFIED"

        # --------------------------------------------------
        # 4. Main tracking loop
        # --------------------------------------------------
        while True:

            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break

            frame_count += 1

            # --------------------------------------------------
            # CSRT tracking
            # --------------------------------------------------
            success, tracker_bbox = tracker.update(frame)

            if success:
                tx, ty, tw, th = [
                    int(value) for value in tracker_bbox
                ]

                # Draw CSRT box
                cv2.rectangle(
                    frame,
                    (tx, ty),
                    (tx + tw, ty + th),
                    (0, 255, 0),
                    2,
                )

                tracking_status = "TRACKING"

            else:
                tracking_status = "TRACKING LOST"

            # --------------------------------------------------
            # YOLO verification every N frames
            # --------------------------------------------------
            if frame_count % YOLO_INTERVAL == 0:
                results = model(
                    frame,
                    conf=YOLO_CONFIDENCE,
                    verbose=False,
                )

                detections = results[0].boxes

                best_iou = 0.0
                best_detection = None

                if success and target_class_id is not None:

                    for box in detections:

                        # Ignore objects belonging to other classes
                        class_id = int(box.cls[0].item())

                        if class_id != target_class_id:
                            continue

                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        detection_bbox = (
                            int(x1),
                            int(y1),
                            int(x2 - x1),
                            int(y2 - y1),
                        )

                        iou = calculate_iou(
                            tracker_bbox,
                            detection_bbox,
                        )

                        if iou > best_iou:
                            best_iou = iou
                            best_detection = detection_bbox

                    if best_detection is not None:

                        if best_iou >= IOU_THRESHOLD:
                            verification_status = (
                                f"VERIFIED {target_class_name} "
                                f"IoU={best_iou:.2f}"
                            )
                            # Target is probably correct, reset verification failures
                            verification_failures = 0


                        else:

                            verification_failures += 1

                            verification_status = (
                                f"UNCERTAIN {target_class_name} "
                                f"IoU={best_iou:.2f}"
                            )

                    else:
                        verification_failures += 1
                        verification_status = (
                            f"NO TARGET MATCH "
                            f"FAILURES={verification_failures}"
                        )

                elif  not success:
                    verification_failures += 1
                    verification_status = (
                            f"TRACKER LOST "
                            f"FAILURES={verification_failures}"
                        )

                else:
                    verification_status = "TRACKER LOST"

            # --------------------------------
            # Recovery
            # --------------------------------

            if verification_failures >= MAX_VERIFICATION_FAILURES:
                print(
                    "Verification failures exceeded threshold. "
                    "Attempting to recover tracker."
                )

                if best_detection is not None:
                    tracker = recover_tracker(
                        tracker,
                        frame,
                        best_detection,
                    )
                    verification_failures = 0
                    verification_status = (
                        f"RECOVERED {target_class_name} "
                        f"IoU={best_iou:.2f}"
                    )
                    print("Tracker recovered using YOLO detection.")

                else:
                    print(
                        "Recovery failed: No valid YOLO detection available."
                    )
                    verification_status = (
                        "RECOVERY FAILED: NO VALID DETECTION"
                    )
                
            # --------------------------------------------------
            # Display status
            # --------------------------------------------------
            cv2.putText(
                frame,
                tracking_status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if success else (0, 0, 255),
                2,
            )

            cv2.putText(
                frame,
                verification_status,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            cv2.imshow(
                "EyeGo Robust Object Tracker",
                frame,
            )

            # --------------------------------------------------
            # Quit
            # --------------------------------------------------
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        print("Webcam released.")
        print("Application closed.")


if __name__ == "__main__":
    main()
