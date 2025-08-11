# OpenCV Beard Filter Camera

당신의 멋진 턱수염을 선택해보세요!

A real-time camera application that applies virtual beard filters using OpenCV.  
Supports 6 different beard styles, selectable during live video streaming.

![demo](video/beard1.gif)

## Features
- Real-time webcam video capture
- 6 beard styles (PNG overlay with alpha blending)
- Filter effects:
  - None
  - Grayscale
  - Gaussian Blur
  - Edge Detection (Canny)
  - Sepia Tone
  - HSV Saturation Adjustment
- Mouse click to place beard at specific position
- Adjustable beard size (`[` / `]`)
- Undo (`U`) and clear all (`C`) beard placements
- ESC key to exit and save the video recording

## Requirements
- Python 3.x
- OpenCV

## Installation
```
pip install opencv-python
```

## Usage
```
python main.py
```

## Controls
| Key                   | Action                          |
| --------------------- | ------------------------------- |
| 1 - 6                 | Select beard style              |
| N / G / B / E / S / H | Apply filter                    |
| Mouse Click           | Place beard at clicked position |
| `[` / `]`             | Decrease / Increase beard size  |
| U                     | Undo last beard                 |
| C                     | Clear all beards                |
| ESC                   | Exit and save video             |


