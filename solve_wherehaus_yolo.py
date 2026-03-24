import argparse
import glob
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO


# Task IDs from the PDF:
# 1 – пожаротушение, 2 – наводнение, 3 – неотложная помощь, 4 – техногенная авария
ID_TO_NAME: Dict[int, str] = {
    1: "fire",
    2: "flood",
    3: "medical",
    4: "industrial",
}

DRAW_COLORS: Dict[int, Tuple[int, int, int]] = {
    1: (0, 0, 255),
    2: (255, 120, 0),
    3: (0, 140, 255),
    4: (0, 220, 220),
}

NAME_ALIASES: Dict[str, int] = {
    # English
    "fire": 1,
    "flood": 2,
    "medical": 3,
    "industrial": 4,
    "emergency": 3,
    "accident": 4,
    "tech": 4,
    # Possible Roboflow class names
    "fire_extinguishing": 1,
    "flooding": 2,
    "first_aid": 3,
    "technogenic_accident": 4,
    # Russian / translit / short labels
    "пожаротушение": 1,
    "наводнение": 2,
    "неотложная_помощь": 3,
    "неотложная помощь": 3,
    "техногенная_авария": 4,
    "техногенная авария": 4,
    "sklad_pozharotusheniya": 1,
    "sklad_pri_navodnenii": 2,
    "sklad_neotlozhnoj_pomoshhi": 3,
    "sklad_likvidacii_avarij": 4,
    # Numeric labels if dataset is named like 1/2/3/4
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wherehaus detector using a trained YOLO model."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="best.pt",
        help="Path to trained YOLO weights.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="",
        help="Path to one video. If omitted, all videos from ./Wherehaus or ./Day2/Wherehaus are used.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference image size.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device for inference: '', 'cpu', '0', '0,1', etc.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional output video path. If processing several videos, suffixes will be added.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Disable OpenCV window.",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=0.0,
        help="Optional delay before each video starts.",
    )
    parser.add_argument(
        "--stop-area",
        type=float,
        default=0.015,
        help="Minimal bbox/frame area ratio to consider the target 'close'.",
    )
    parser.add_argument(
        "--stop-streak",
        type=int,
        default=4,
        help="How many close frames in a row are needed to enable STOP.",
    )
    parser.add_argument(
        "--release-streak",
        type=int,
        default=6,
        help="How many far frames in a row are needed to disable STOP.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info for detections.",
    )
    return parser.parse_args()


def find_video_paths(single_video: str) -> List[str]:
    if single_video.strip():
        return [single_video.strip()]

    candidates = sorted(glob.glob(os.path.join("Wherehaus", "*.avi")))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join("Day2", "Wherehaus", "*.avi")))
    return candidates


def normalize_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def build_class_mapping(model: YOLO) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    names = model.names if hasattr(model, "names") else {}

    for raw_idx, raw_name in names.items():
        idx = int(raw_idx)
        norm = normalize_name(str(raw_name))
        if norm in NAME_ALIASES:
            mapping[idx] = NAME_ALIASES[norm]

    return mapping


def open_writer(save_path: str, video_path: str, width: int, height: int, fps: float, video_idx: int, total: int):
    if not save_path.strip():
        return None

    save_path = save_path.strip()
    base = Path(save_path)
    if total > 1:
        out_path = base.with_name(f"{base.stem}_{video_idx}{base.suffix or '.mp4'}")
    else:
        out_path = base if base.suffix else base.with_suffix(".mp4")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps if fps > 0 else 25.0, (width, height))
    return writer


def infer_task_id(class_mapping: Dict[int, int], cls_idx: int, fallback_name: str) -> Optional[int]:
    if cls_idx in class_mapping:
        return class_mapping[cls_idx]

    norm = normalize_name(fallback_name)
    return NAME_ALIASES.get(norm)


def process_video(
    model: YOLO,
    class_mapping: Dict[int, int],
    video_path: str,
    target_id: int,
    args: argparse.Namespace,
    video_idx: int,
    total_videos: int,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] cannot open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_area = max(1, width * height)

    writer = open_writer(args.save, video_path, width, height, fps, video_idx, total_videos)

    print(f"\n[INFO] processing: {video_path}")
    if args.warmup_seconds > 0:
        print(f"[INFO] starting in {args.warmup_seconds:.1f} sec...")
        time.sleep(args.warmup_seconds)

    stop_printed = False
    stop_active = False
    target_close_streak = 0
    target_far_streak = 0
    last_stop_anchor: Tuple[int, int] = (40, 60)
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        results = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device if args.device.strip() else None,
            verbose=False,
        )

        target_close_this_frame = False
        current_anchor = last_stop_anchor

        if results:
            res = results[0]
            boxes = getattr(res, "boxes", None)
            names = getattr(res, "names", model.names)

            if boxes is not None and boxes.xyxy is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
                clss = boxes.cls.cpu().numpy() if boxes.cls is not None else []

                for i, box in enumerate(xyxy):
                    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                    cls_idx = int(clss[i]) if len(clss) > i else -1
                    conf = float(confs[i]) if len(confs) > i else 0.0
                    raw_name = str(names.get(cls_idx, cls_idx))
                    task_id = infer_task_id(class_mapping, cls_idx, raw_name)

                    if task_id is None:
                        if args.debug:
                            print(
                                f"[DBG] f{frame_idx} unknown class idx={cls_idx} name={raw_name} conf={conf:.3f}"
                            )
                        continue

                    label = f"{task_id}:{ID_TO_NAME[task_id]} {conf:.2f}"
                    box_color = (0, 255, 255) if task_id == target_id else DRAW_COLORS[task_id]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(22, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        box_color,
                        2,
                        cv2.LINE_AA,
                    )

                    box_area = max(1, (x2 - x1) * (y2 - y1))
                    area_ratio = box_area / float(frame_area)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    inside_center = int(0.10 * width) <= cx <= int(0.90 * width)
                    inside_vertical = int(0.15 * height) <= cy <= int(0.90 * height)
                    no_edge_cut = x1 > 3 and x2 < width - 3 and y1 > 3 and y2 < height - 3
                    close_enough = (
                        area_ratio > args.stop_area
                        and inside_center
                        and inside_vertical
                        and no_edge_cut
                    )

                    if args.debug:
                        print(
                            f"[DBG] f{frame_idx} cls={task_id}:{ID_TO_NAME[task_id]} "
                            f"raw={raw_name} conf={conf:.3f} box=({x1},{y1},{x2},{y2}) "
                            f"area_ratio={area_ratio:.4f} close={int(close_enough)}"
                        )

                    if task_id == target_id and close_enough:
                        target_close_this_frame = True
                        current_anchor = (x1, min(height - 10, y2 + 30))

        if target_close_this_frame:
            target_close_streak += 1
            target_far_streak = 0
            last_stop_anchor = current_anchor
        else:
            target_close_streak = max(0, target_close_streak - 1)
            target_far_streak += 1

        if not stop_active and target_close_streak >= args.stop_streak:
            stop_active = True
            if not stop_printed:
                print(f"[STOP] target {target_id} at frame {frame_idx}")
                stop_printed = True

        if stop_active and target_far_streak >= args.release_streak:
            stop_active = False

        if stop_active:
            cv2.putText(
                frame,
                "STOP",
                last_stop_anchor,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        if writer is not None:
            writer.write(frame)

        if not args.no_window:
            cv2.namedWindow("Wherehaus YOLO", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Wherehaus YOLO", width, height)
            cv2.imshow("Wherehaus YOLO", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_window:
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()

    try:
        target_id = int(
            input(
                "Введите целевой склад: 1-пожаротушение, 2-наводнение, 3-неотложная помощь, 4-техногенная авария: "
            ).strip()
        )
    except ValueError:
        print("[ERROR] enter integer from 1 to 4")
        return

    if target_id not in ID_TO_NAME:
        print("[ERROR] target must be in [1..4]")
        return

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        print(f"[ERROR] weights file not found: {weights_path}")
        return

    videos = find_video_paths(args.video)
    if not videos:
        print("[ERROR] no videos found in ./Wherehaus or ./Day2/Wherehaus")
        return

    model = YOLO(str(weights_path))
    class_mapping = build_class_mapping(model)
    print(f"[INFO] loaded model: {weights_path}")
    print(f"[INFO] model names: {model.names}")
    print(f"[INFO] class mapping: {class_mapping}")

    for idx, video_path in enumerate(videos, start=1):
        process_video(
            model=model,
            class_mapping=class_mapping,
            video_path=video_path,
            target_id=target_id,
            args=args,
            video_idx=idx,
            total_videos=len(videos),
        )


if __name__ == "__main__":
    main()
