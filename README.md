# EyeGo Real-Time Object Tracker

A real-time single-object tracking system built with **OpenCV CSRT** and **YOLO-assisted verification and recovery**.

## Overview

The application allows a user to select an object from the first webcam frame using a bounding box.

The system then uses the **CSRT tracker** to follow the selected object in real time.

To improve tracking robustness, **YOLO11n** periodically verifies the CSRT tracking result by comparing the tracked bounding box with YOLO detections belonging to the originally identified object class.

If repeated verification failures occur, the system attempts to **recover the tracker using the best matching YOLO detection**.

## Features

* Manual target selection using a bounding box
* Real-time CSRT tracking
* YOLO11n target-class identification
* Periodic YOLO verification
* Class-aware YOLO verification
* IoU-based geometric validation
* Consecutive verification failure handling
* Automatic tracker recovery using YOLO detections
* Protection against unrelated object classes during verification
* OpenCV CSRT compatibility handling
* Lightweight real-time webcam processing

## Architecture

```text
                         Webcam
                            |
                            v
                       First Frame
                            |
                            v
                    User Selects Target
                            |
                            v
                           YOLO
                            |
                            v
                 Identify Target Class
                            |
                            v
                          CSRT
                            |
                            v
                  Frame-by-Frame Tracking
                            |
                            v
                   Periodic YOLO Check
                            |
                            v
              Filter by Target Class
                            |
                            v
                    Calculate IoU
                            |
                  +---------+---------+
                  |                   |
                  v                   v
              IoU >= 0.30         IoU < 0.30
                  |                   |
                  v                   v
              VERIFIED            UNCERTAIN
                  |                   |
                  |              Failure Count +1
                  |                   |
                  +---------+---------+
                            |
                            v
                  Repeated Failures?
                            |
                     +------+------+
                     |             |
                    No            Yes
                     |             |
                     v             v
                 Continue       Recovery
                                   |
                            +------+------+
                            |             |
                            v             v
                       YOLO Detection   No Detection
                            |             |
                            v             v
                     Reinitialize CSRT  Recovery Failed
                            |
                            v
                         Continue
```

## Why CSRT + YOLO?

**CSRT** is used as the primary tracker because it is designed for single-object tracking and can efficiently follow a user-selected target frame by frame.

However, a tracker can sometimes drift or lose the target. Therefore, **YOLO11n** is used as an auxiliary verification and recovery mechanism.

YOLO periodically detects objects in the current frame. The system then:

1. Filters detections using the target class identified during initialization.
2. Compares the YOLO detections with the current CSRT bounding box.
3. Calculates **Intersection over Union (IoU)**.
4. Uses the IoU result to determine whether the tracker is likely still following the intended object.
5. Attempts recovery if repeated verification failures occur.

This combination provides a balance between **fast tracking** and **periodic detection-based correction**.

## Target Identification

When the user selects the target in the first frame, YOLO is also executed on that frame.

The system compares the user's selected bounding box with YOLO detections and selects the detection with the highest IoU.

The corresponding YOLO class becomes the target class.

For example:

```text
User selects object
        |
        v
      YOLO
        |
        v
+-------------------+
| person            |
| bottle            |
| chair             |
+-------------------+
        |
        v
Highest IoU with
user selection
        |
        v
Target Class = person
```

If YOLO cannot identify the selected object, the system continues with **CSRT-only tracking**.

## YOLO Verification

YOLO does not run on every frame.

Instead, verification is performed every:

```text
10 frames
```

This reduces computational overhead while still providing periodic validation.

Only detections belonging to the **original target class** are considered.

For example, if the selected target was identified as `person`, detections belonging to other classes are ignored.

The system then calculates IoU between:

* The current CSRT bounding box
* The best matching YOLO detection

The current configuration uses:

```python
IOU_THRESHOLD = 0.30
```

An IoU greater than or equal to `0.30` is considered a successful verification.

## Tracker Recovery

If the system experiences repeated verification failures, it attempts to recover the CSRT tracker.

The current configuration allows:

```python
MAX_VERIFICATION_FAILURES = 3
```

When the threshold is reached:

1. The system searches for a valid YOLO detection belonging to the original target class.
2. The best matching detection is selected.
3. CSRT is reinitialized using that detection's bounding box.
4. The verification failure counter is reset.

Example:

```text
CSRT Tracking
      |
      v
Verification Failure
      |
      v
Failure Counter
      |
      v
3 Consecutive Failures
      |
      v
YOLO Detection
      |
      v
Valid Detection?
   /          \
 Yes           No
  |             |
  v             v
Recover      Recovery Failed
CSRT
```

If no valid YOLO detection is available, the system reports:

```text
RECOVERY FAILED: NO VALID DETECTION
```

## Technologies

* **Python 3.10+**
* **OpenCV**
* **OpenCV CSRT Tracker**
* **Ultralytics YOLO11n**
* **uv**

## Installation

### Requirements

* Python 3.10+
* Webcam
* Windows / Linux / macOS
* [uv](https://docs.astral.sh/uv/)

### Clone the Repository

```bash
git clone <https://github.com/OmarElsayed771/eyego-object-tracker.git>
cd EyeGo_Tracker
```

### Install Dependencies

The project uses `uv` for environment and dependency management.

```bash
uv sync
```

## Run

Start the application with:

```bash
uv run main.py
```

When the application starts:

1. The webcam is opened.
2. The first frame is displayed.
3. Select the object you want to track.
4. Press **ENTER** or **SPACE** to confirm.
5. CSRT starts tracking the selected object.
6. YOLO periodically verifies the tracking result.

## Controls

| Key | Action |
| --- | ------ |
| `Q` | Quit   |

During target selection:

| Key     | Action            |
| ------- | ----------------- |
| `ENTER` | Confirm selection |
| `SPACE` | Confirm selection |
| `C`     | Cancel selection  |

## Implementation Details

### 1. Target Selection

The first webcam frame is captured and displayed using OpenCV's ROI selector.

The user manually draws a bounding box around the object they want to track.

```python
user_roi = cv2.selectROI(
    "Select Target",
    frame,
    fromCenter=False,
    showCrosshair=True,
)
```

### 2. Target Class Identification

YOLO11n is executed on the first frame.

The system calculates IoU between the selected ROI and each YOLO detection.

The detection with the highest IoU determines the target class.

### 3. CSRT Tracking

The selected ROI is passed to the CSRT tracker.

```python
tracker = create_csrt_tracker()
tracker.init(frame, user_roi)
```

CSRT then provides an updated bounding box for each incoming webcam frame.

### 4. Periodic YOLO Verification

Every `YOLO_INTERVAL` frames, YOLO runs on the current frame.

The system ignores detections belonging to unrelated classes.

It then calculates IoU between the CSRT bounding box and the best matching YOLO detection.

### 5. Verification Failure Handling

When the IoU is below the configured threshold, the verification failure counter is increased.

When verification succeeds, the counter is reset.

This prevents a single temporary detection failure from immediately triggering recovery.

### 6. Tracker Recovery

After three consecutive verification failures, the system attempts to recover the tracker.

If a valid YOLO detection is available, CSRT is reinitialized using that detection's bounding box.

This allows the system to recover from temporary tracking drift or loss.

## Configuration

The main tracking parameters are defined at the beginning of `main.py`:

```python
YOLO_INTERVAL = 10
YOLO_CONFIDENCE = 0.40
IOU_THRESHOLD = 0.30
MAX_VERIFICATION_FAILURES = 3
```

| Parameter                   | Description                                      |
| --------------------------- | ------------------------------------------------ |
| `YOLO_INTERVAL`             | Number of frames between YOLO verification runs  |
| `YOLO_CONFIDENCE`           | Minimum YOLO detection confidence                |
| `IOU_THRESHOLD`             | Minimum IoU required for successful verification |
| `MAX_VERIFICATION_FAILURES` | Number of consecutive failures before recovery   |

## Project Structure

```text
EyeGo_Tracker/
│
├── main.py
├── association.py
├── yolo11n.pt
├── pyproject.toml
├── uv.lock
├── README.md
└── .gitignore
```

### File Responsibilities

| File             | Purpose                           |
| ---------------- | --------------------------------- |
| `main.py`        | Main webcam tracking application  |
| `association.py` | Bounding-box IoU calculation      |
| `yolo11n.pt`     | YOLO11n object detection model    |
| `pyproject.toml` | Project metadata and dependencies |
| `uv.lock`        | Locked dependency versions        |
| `README.md`      | Project documentation             |
| `.gitignore`     | Files excluded from Git           |

## Limitations

* YOLO verification depends on the classes supported by the YOLO11n model.
* Objects that YOLO cannot recognize can still be tracked by CSRT, but YOLO verification and recovery will not be available for them.
* CSRT can still lose the target under severe occlusion or rapid movement.
* Recovery depends on YOLO successfully detecting the original target class.
* The current implementation uses IoU as the main association method and does not perform advanced visual re-identification.


## License

This project was developed as part of the **EyeGo Real-Time Object Tracking Coding Task**.
