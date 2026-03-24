import argparse
import glob
import os
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from Wherehaus videos for Roboflow labeling.")
    parser.add_argument("--input-dir", type=str, default="Wherehaus", help="Folder with input videos.")
    parser.add_argument("--pattern", type=str, default="*.avi", help="Glob pattern for videos.")
    parser.add_argument("--output-dir", type=str, default="dataset/images", help="Where to save extracted frames.")
    parser.add_argument("--every", type=int, default=10, help="Save every N-th frame.")
    parser.add_argument("--max-per-video", type=int, default=250, help="Cap extracted frames per video.")
    parser.add_argument("--blur-threshold", type=float, default=45.0, help="Skip very blurry frames below this Laplacian variance.")
    return parser.parse_args()


def blur_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not videos and args.input_dir == "Wherehaus":
        videos = sorted(glob.glob(os.path.join("Day2", "Wherehaus", args.pattern)))

    if not videos:
        print("[ERROR] no videos found")
        return

    total_saved = 0
    for video_path in videos:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARN] cannot open {video_path}")
            continue

        stem = Path(video_path).stem
        frame_idx = 0
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            if frame_idx % max(1, args.every) != 0:
                continue
            if saved >= args.max_per_video:
                break
            if blur_score(frame) < args.blur_threshold:
                continue

            out_name = f"{stem}_f{frame_idx:06d}.jpg"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), frame)
            saved += 1
            total_saved += 1

        cap.release()
        print(f"[INFO] {video_path}: saved {saved} frames")

    print(f"[DONE] total saved: {total_saved}")


if __name__ == "__main__":
    main()
