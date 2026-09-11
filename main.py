import time

import cv2
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================

YOLO_INTERVAL = 10
YOLO_CONFIDENCE = 0.40

# IoU between CSRT and YOLO detection
IOU_THRESHOLD = 0.30

# Appearance similarity threshold.
# Higher = stricter identity verification.
APPEARANCE_THRESHOLD = 0.45

# Number of consecutive verification failures before
# declaring the target lost.
MAX_VERIFICATION_FAILURES = 3


# ============================================================
# CSRT
# ============================================================

def create_csrt_tracker():
    """
    Create a CSRT tracker compatible with different OpenCV versions.
    """

    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()

    if (
        hasattr(cv2, "legacy")
        and hasattr(cv2.legacy, "TrackerCSRT_create")
    ):
        return cv2.legacy.TrackerCSRT_create()

    raise RuntimeError(
        "CSRT tracker is not available in this OpenCV installation."
    )


# ============================================================
# Bounding Box Utilities
# ============================================================

def calculate_iou(box_a, box_b):
    """
    Calculate Intersection over Union.

    Box format:
        (x, y, width, height)
    """

    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2 = ax + aw
    ay2 = ay + ah

    bx2 = bx + bw
    by2 = by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    area_a = aw * ah
    area_b = bw * bh

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def clamp_bbox(bbox, frame_shape):
    """
    Keep bounding box inside the frame.
    """

    frame_height, frame_width = frame_shape[:2]

    x, y, w, h = [int(v) for v in bbox]

    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))

    w = max(1, min(w, frame_width - x))
    h = max(1, min(h, frame_height - y))

    return x, y, w, h


# ============================================================
# Appearance
# ============================================================

def calculate_appearance_histogram(frame, bbox):
    """
    Create an HSV color histogram for the target.

    This gives the tracker a simple appearance signature
    of the object selected by the user.
    """

    x, y, w, h = clamp_bbox(bbox, frame.shape)

    crop = frame[y:y + h, x:x + w]

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [30, 32],
        [0, 180, 0, 256],
    )

    cv2.normalize(
        histogram,
        histogram,
        0,
        1,
        cv2.NORM_MINMAX,
    )

    return histogram


def calculate_appearance_similarity(
    frame,
    bbox,
    target_histogram,
):
    """
    Compare the current tracker crop against
    the original target appearance.

    Returns:
        similarity in range approximately [-1, 1]
    """

    current_histogram = calculate_appearance_histogram(
        frame,
        bbox,
    )

    if current_histogram is None or target_histogram is None:
        return 0.0

    similarity = cv2.compareHist(
        target_histogram,
        current_histogram,
        cv2.HISTCMP_CORREL,
    )

    return float(similarity)


# ============================================================
# YOLO
# ============================================================

def identify_initial_target(
    model,
    frame,
    user_bbox,
):
    """
    Try to associate the user's selected ROI with a YOLO detection.

    Important:
    YOLO is NOT allowed to redefine the user's target.

    It is only used if it can identify the selected object.
    """

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

        iou = calculate_iou(
            user_bbox,
            detection_bbox,
        )

        if iou > best_iou:

            best_iou = iou

            target_class_id = int(
                box.cls[0].item()
            )

            target_class_name = model.names[
                target_class_id
            ]

    return (
        target_class_id,
        target_class_name,
        best_iou,
    )


def verify_with_yolo(
    model,
    frame,
    tracker_bbox,
    target_class_id,
):
    """
    Verify the tracker against YOLO.

    Only detections belonging to the ORIGINAL target
    class are considered.
    """

    results = model(
        frame,
        conf=YOLO_CONFIDENCE,
        verbose=False,
    )

    detections = results[0].boxes

    best_iou = 0.0
    best_detection = None

    if target_class_id is None:
        return best_iou, best_detection

    for box in detections:

        class_id = int(
            box.cls[0].item()
        )

        # VERY IMPORTANT:
        # Never consider another object class.
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

    return best_iou, best_detection


# ============================================================
# Main Application
# ============================================================

def main():

    # --------------------------------------------------------
    # Load YOLO
    # --------------------------------------------------------

    print("Loading YOLO model...")

    model = YOLO("yolo11n.pt")

    print("YOLO model loaded.")

    # --------------------------------------------------------
    # Open webcam
    # --------------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("Error: Could not open webcam.")

        return

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    tracker = None

    target_bbox = None

    target_histogram = None

    target_class_id = None
    target_class_name = None

    verification_failures = 0

    frame_count = 0

    tracking_status = "NO TARGET"

    verification_status = "SELECT TARGET"

    fps = 0.0

    previous_time = time.time()

    try:

        # ====================================================
        # Target Selection
        # ====================================================

        ret, frame = cap.read()

        if not ret:

            print("Error: Could not read first frame.")

            return

        print()
        print("Select the object you want to track.")
        print("Press SPACE or ENTER to confirm.")
        print("Press C to cancel.")

        user_roi = cv2.selectROI(
            "Select Target",
            frame,
            fromCenter=False,
            showCrosshair=True,
        )

        cv2.destroyWindow("Select Target")

        x, y, w, h = user_roi

        if w <= 0 or h <= 0:

            print("Error: No valid object selected.")

            return

        target_bbox = user_roi

        print(
            f"User selected target: {target_bbox}"
        )

        # ====================================================
        # Save Target Appearance
        # ====================================================

        target_histogram = calculate_appearance_histogram(
            frame,
            target_bbox,
        )

        if target_histogram is None:

            print(
                "Error: Could not create target appearance."
            )

            return

        print("Target appearance captured.")

        # ====================================================
        # Optional YOLO Target Identification
        # ====================================================

        (
            target_class_id,
            target_class_name,
            initial_iou,
        ) = identify_initial_target(
            model,
            frame,
            target_bbox,
        )

        if target_class_id is not None:

            print(
                f"YOLO identified selected target as: "
                f"{target_class_name}"
            )

            print(
                f"Initial IoU: {initial_iou:.2f}"
            )

        else:

            print(
                "YOLO could not identify the selected object."
            )

            print(
                "This is okay. "
                "CSRT + appearance verification will be used."
            )

        # ====================================================
        # Initialize CSRT
        # ====================================================

        tracker = create_csrt_tracker()

        tracker.init(
            frame,
            target_bbox,
        )

        print("CSRT tracker initialized.")

        tracking_status = "TRACKING"

        verification_status = "INITIALIZED"

        # ====================================================
        # Main Loop
        # ====================================================

        while True:

            ret, frame = cap.read()

            if not ret:

                print(
                    "Error: Could not read frame."
                )

                break

            frame_count += 1

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            current_time = time.time()

            elapsed = current_time - previous_time

            if elapsed > 0:

                fps = 1.0 / elapsed

            previous_time = current_time

            # ------------------------------------------------
            # CSRT Tracking
            # ------------------------------------------------

            success, tracker_bbox = tracker.update(frame)

            appearance_similarity = 0.0

            if success:

                tracker_bbox = clamp_bbox(
                    tracker_bbox,
                    frame.shape,
                )

                appearance_similarity = (
                    calculate_appearance_similarity(
                        frame,
                        tracker_bbox,
                        target_histogram,
                    )
                )

                # --------------------------------------------
                # Appearance verification
                # --------------------------------------------

                if (
                    appearance_similarity
                    >= APPEARANCE_THRESHOLD
                ):

                    tracking_status = "TRACKING"

                else:

                    verification_failures += 1

                    tracking_status = (
                        "TARGET UNCERTAIN"
                    )

                # Draw tracker box
                tx, ty, tw, th = tracker_bbox

                cv2.rectangle(
                    frame,
                    (tx, ty),
                    (tx + tw, ty + th),
                    (0, 255, 0),
                    2,
                )

            else:

                verification_failures += 1

                tracking_status = "TRACKING LOST"

            # ------------------------------------------------
            # YOLO verification
            # ------------------------------------------------

            if (
                frame_count % YOLO_INTERVAL == 0
                and success
                and target_class_id is not None
            ):

                (
                    yolo_iou,
                    yolo_detection,
                ) = verify_with_yolo(
                    model,
                    frame,
                    tracker_bbox,
                    target_class_id,
                )

                if yolo_detection is not None:

                    if yolo_iou >= IOU_THRESHOLD:

                        verification_status = (
                            f"YOLO VERIFIED "
                            f"IoU={yolo_iou:.2f}"
                        )

                    else:

                        verification_failures += 1

                        verification_status = (
                            f"YOLO UNCERTAIN "
                            f"IoU={yolo_iou:.2f}"
                        )

                else:

                    verification_status = (
                        "YOLO: TARGET NOT DETECTED"
                    )

            elif (
                frame_count % YOLO_INTERVAL == 0
                and target_class_id is None
            ):

                verification_status = (
                    "APPEARANCE VERIFICATION"
                )

            # ------------------------------------------------
            # Failure Handling
            # ------------------------------------------------

            if (
                verification_failures
                >= MAX_VERIFICATION_FAILURES
            ):

                tracking_status = "TARGET LOST"

                verification_status = (
                    "TARGET LOST - NO SWITCH"
                )

                print(
                    "Target verification failed."
                )

                print(
                    "Tracker will NOT switch "
                    "to another object."
                )



                success = False

            # ------------------------------------------------
            # Display information
            # ------------------------------------------------

            status_color = (
                (0, 255, 0)
                if tracking_status == "TRACKING"
                else (0, 0, 255)
            )

            cv2.putText(
                frame,
                tracking_status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                status_color,
                2,
            )

            cv2.putText(
                frame,
                verification_status,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            if target_class_name is not None:

                cv2.putText(
                    frame,
                    f"Target class: {target_class_name}",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 255),
                    2,
                )

            cv2.putText(
                frame,
                "R: reselect target | Q: quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            # ------------------------------------------------
            # Show frame
            # ------------------------------------------------

            cv2.imshow(
                "EyeGo Object Tracker",
                frame,
            )

            # ------------------------------------------------
            # Keyboard
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                break

            # -----------------------------------------------
            # Reselect target
            # -----------------------------------------------

            if key == ord("r"):

                print()
                print(
                    "Reselecting target..."
                )

                new_roi = cv2.selectROI(
                    "Select Target",
                    frame,
                    fromCenter=False,
                    showCrosshair=True,
                )

                cv2.destroyWindow(
                    "Select Target"
                )

                nx, ny, nw, nh = new_roi

                if nw > 0 and nh > 0:

                    target_bbox = new_roi

                    target_histogram = (
                        calculate_appearance_histogram(
                            frame,
                            target_bbox,
                        )
                    )

                    (
                        target_class_id,
                        target_class_name,
                        initial_iou,
                    ) = identify_initial_target(
                        model,
                        frame,
                        target_bbox,
                    )

                    tracker = create_csrt_tracker()

                    tracker.init(
                        frame,
                        target_bbox,
                    )

                    verification_failures = 0

                    tracking_status = "TRACKING"

                    verification_status = (
                        "TARGET RESELECTED"
                    )

                    print(
                        f"New target: {target_bbox}"
                    )

                    if target_class_name:

                        print(
                            f"Target class: "
                            f"{target_class_name}"
                        )

                else:

                    print(
                        "Invalid selection. "
                        "Keeping previous target."
                    )

    finally:

        cap.release()

        cv2.destroyAllWindows()

        print()
        print("Webcam released.")
        print("Application closed.")


if __name__ == "__main__":
    main()
