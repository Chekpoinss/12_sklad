# 12_sklad
# Wherehaus YOLO Detector

YOLO-based detector for warehouse sign recognition in a robotics/hackathon task.

The project detects 4 warehouse sign classes:
- fire
- flood
- medical
- industrial

It also supports a simple task-oriented inference pipeline for selecting a target warehouse and triggering a STOP condition when the target sign is detected confidently.

## Features

- frame extraction from input `.avi` videos
- dataset annotation workflow via Roboflow
- YOLO11 training with Ultralytics
- inference script for warehouse sign detection
- support for 4 sign classes
## Уставновка

python solve_wherehaus_yolo.py --weights runs\detect\train\weights\best.pt --debug
