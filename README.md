# EyeGo Real-Time Object Tracker

A real-time single-object tracking system built with OpenCV CSRT and YOLO-assisted verification.

## Overview

The application allows a user to select an arbitrary object from the first webcam frame and tracks that specific object in real time.

The selected object remains the target throughout the session. The system does not switch to another object if the original target disappears.

## Features

- Manual target selection using a bounding box
- Real-time CSRT tracking
- Target appearance verification
- Optional YOLO-based verification
- IoU-based geometric validation
- Target-class-aware YOLO verification
- Consecutive verification failure handling
- Target-lost state without switching identity
- FPS monitoring
- Target re-selection
- OpenCV compatibility handling
- Automated unit tests for bounding-box IoU

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
    +----------------------+
    |                      |
    v                      v
    Target Appearance       YOLO
    |                      |
    |                 Optional Class
    |                  Identification
    |                      |
    +----------+-----------+
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
            +-----+-----+
            |           |
            v           v
        Appearance     YOLO
        Similarity    + IoU
            |           |
            +-----+-----+
                |
                v
        Target Validation
                |
            +-----+-----+
            |           |
            v           v
        Valid       Repeated
                    Failure
            |           |
            v           v
        Continue   TARGET LOST




## Why CSRT + YOLO?

CSRT is used as the primary tracker because the task requires tracking a single user-selected object in real time.

YOLO is used as an auxiliary detector/verification mechanism rather than as the primary tracker.

This separation is important because YOLO does not inherently know which instance of an object the user selected.

For example, if the user selects one person among several people, YOLO can detect the class person, but it does not automatically identify the exact person selected by the user.

Therefore, the user's initial selection remains the source of target identity.

## Target Identity

The system stores an appearance representation of the selected object and uses it to help determine whether the tracker is still following the original target.

If the target disappears, the system reports:

TARGET LOST - NO SWITCH

instead of automatically switching to another visible object.

## Technologies
Python
OpenCV
OpenCV CSRT Tracker
Ultralytics YOLO
uv
Installation
Requirements
Python 3.10+
Webcam
Windows/Linux/macOS
uv



## Clone the repository
### Bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd EyeGo_Tracker
----------------------------------------------------------------
## Create environment and install dependencies
### Bash
uv sync
----------------------------------------------------------------
## Run
### Bash
uv run main.py
----------------------------------------------------------------
## Controls
### Key	Action
Q	Quit
R	Select a new target

During startup, draw a bounding box around the object you want to track and press ENTER or SPACE.

----------------------------------------------------------------
## Implementation Details
1. Target Selection

The first webcam frame is displayed and the user selects the target using OpenCV's ROI selector.

2. CSRT Tracking

CSRT tracks the selected object frame by frame and provides an updated bounding box.

3. Appearance Verification

An HSV color histogram is extracted from the initial target and compared with the current tracked region.

4. YOLO Verification

YOLO runs periodically rather than on every frame to reduce computational overhead.

When YOLO can identify the target class, its detections are compared with the CSRT bounding box using Intersection over Union (IoU).

5. Target Loss

Several consecutive verification failures are required before the system declares the target lost.

The system intentionally does not switch to another object when the selected target disappears.


----------------------------------------------------------------
## Project Structure
EyeGo_Tracker/
├── main.py
├── yolo11n.pt
├── pyproject.toml
├── uv.lock
├── README.md
└── .gitignore

