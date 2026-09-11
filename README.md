# EyeGo Real-Time Object Tracker

A real-time single-object tracking system built with **OpenCV CSRT** and **YOLO-assisted verification**.

## Overview

The application allows a user to select an arbitrary object from the first webcam frame and track that specific object in real time.

The selected object remains the target throughout the session. The system **does not switch to another object** if the original target disappears.

## Features

* Manual target selection using a bounding box
* Real-time CSRT tracking
* Target appearance verification
* Optional YOLO-based verification
* IoU-based geometric validation
* Target-class-aware YOLO verification
* Consecutive verification failure handling
* Target-lost state without switching identity
* FPS monitoring
* Target re-selection
* OpenCV compatibility handling
* Automated unit tests for bounding-box IoU

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
              +-------------+-------------+
              |                           |
              v                           v
       Target Appearance                YOLO
              |                           |
              |                    Optional Class
              |                     Identification
              |                           |
              +-------------+-------------+
                            |
                            v
                          CSRT
                            |
                            v
                  Frame-by-Frame Tracking
                            |
                            v
                  Periodic Verification
                            |
                    +-------+-------+
                    |               |
                    v               v
             Appearance           YOLO
              Similarity          + IoU
                    |               |
                    +-------+-------+
                            |
                            v
                    Target Validation
                            |
                    +-------+-------+
                    |               |
                    v               v
                  Valid          Repeated
                                 Failure
                    |               |
                    v               v
                Continue       TARGET LOST
```

## Why CSRT + YOLO?

**CSRT** is used as the primary tracker because the task requires tracking a single user-selected object in real time.

**YOLO** is used as an auxiliary detection and verification mechanism rather than as the primary tracker.

This separation is important because YOLO does not inherently know which instance of an object the user selected.

For example, if the user selects one person among several people, YOLO can detect the `person` class, but it does not automatically identify the exact person selected by the user.

Therefore, the **user's initial selection remains the source of target identity**.

## Target Identity

The system stores an appearance representation of the selected object and uses it to help determine whether the tracker is still following the original target.

If the target disappears, the system reports:

```text
TARGET LOST - NO SWITCH
```

instead of automatically switching to another visible object.

This prevents **identity switching**, which is especially important when multiple similar objects are present in the scene.

## Technologies

* Python 3.10+
* OpenCV
* OpenCV CSRT Tracker
* Ultralytics YOLO
* uv

## Installation

### Requirements

* Python 3.10+
* Webcam
* Windows / Linux / macOS
* [uv](https://docs.astral.sh/uv/)

### Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd EyeGo_Tracker
```

### Create Environment and Install Dependencies

The project uses `uv` to create and manage the virtual environment and install dependencies.

```bash
uv sync
```

## Run

Start the tracker with:

```bash
uv run main.py
```

During startup, draw a bounding box around the object you want to track and press **ENTER** or **SPACE** to confirm the selection.

## Controls

| Key | Action              |
| --- | ------------------- |
| `Q` | Quit                |
| `R` | Select a new target |

## Implementation Details

### 1. Target Selection

The first webcam frame is displayed and the user selects the target using OpenCV's ROI selector.

The selected bounding box defines the initial target identity.

### 2. CSRT Tracking

The **CSRT tracker** tracks the selected object frame by frame and provides an updated bounding box.

CSRT is responsible for the primary real-time tracking process.

### 3. Appearance Verification

An **HSV color histogram** is extracted from the initial target region.

The histogram is then compared with the currently tracked region to help detect potential tracking drift.

### 4. YOLO Verification

YOLO runs periodically rather than on every frame to reduce computational overhead.

When YOLO can identify the target class, its detections are compared with the current CSRT bounding box using **Intersection over Union (IoU)**.

The YOLO result is used as a verification signal, not as the source of target identity.

### 5. Target Loss

Several consecutive verification failures are required before the system declares the target lost.

When the target is lost, the system intentionally **does not switch to another object**.

The tracker enters the:

```text
TARGET LOST - NO SWITCH
```

state until the user manually selects a new target.


## Project Structure

```text
EyeGo_Tracker/
│
├── main.py
├── yolo11n.pt
├── pyproject.toml
├── uv.lock
├── README.md
├── .gitignore
```

## Limitations

* YOLO verification only works when the selected object belongs to a class supported by the YOLO model.
* Arbitrary objects that are not recognized by YOLO rely mainly on CSRT and appearance verification.
* Significant occlusion or complete disappearance of the target may cause the tracker to enter the target-lost state.
* Appearance verification based on color histograms may be affected by major lighting changes.


## License

This project was developed as part of the **EyeGo object tracking coding task**.
