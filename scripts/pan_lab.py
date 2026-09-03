#!/usr/bin/env python3
"""Fast visual test bench for crop-pan behavior.

This intentionally bypasses scene detection and YOLO. It exercises the same
crop and interpolation helpers as production, making pan tuning a seconds-long
local loop instead of a full model-assisted render.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from main import calculate_crop_box_for_center, interpolate_pan_x  # noqa: E402


def parse_durations(value):
    durations = []
    for raw in value.split(","):
        duration = float(raw.strip())
        if duration < 0:
            raise argparse.ArgumentTypeError("pan durations must be non-negative")
        durations.append(duration)
    if not durations:
        raise argparse.ArgumentTypeError("provide at least one pan duration")
    return durations


def open_writer(path, fps, size):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create {path}. Verify the OpenCV FFmpeg backend is available.")
    return writer


def generate_synthetic_source(path, width=1280, height=720, fps=30,
                              duration_sec=6):
    """Generate a stable two-subject shot with enough detail to judge panning."""
    writer = open_writer(path, fps, (width, height))
    total_frames = int(fps * duration_sec)
    for frame_number in range(total_frames):
        frame = np.full((height, width, 3), (28, 31, 38), dtype=np.uint8)
        for x in range(0, width, 80):
            cv2.line(frame, (x, 0), (x, height), (48, 52, 61), 1)
        for y in range(0, height, 80):
            cv2.line(frame, (0, y), (width, y), (48, 52, 61), 1)

        # Static framing makes unwanted crop jumps immediately visible.
        for center_x, color, label in [
            (width // 4, (72, 145, 255), "LEFT"),
            (width * 3 // 4, (118, 210, 112), "RIGHT"),
        ]:
            cv2.circle(frame, (center_x, height // 3), 75, color, -1)
            cv2.rectangle(
                frame,
                (center_x - 105, height // 3 + 75),
                (center_x + 105, height - 70),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (center_x - 72, height - 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (15, 18, 22),
                3,
                cv2.LINE_AA,
            )

        seconds = frame_number / fps
        cv2.putText(
            frame,
            f"Synthetic pan fixture  t={seconds:04.2f}s",
            (35, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (235, 238, 244),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()


def read_video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if width <= 0 or height <= 0 or fps <= 0 or total_frames <= 0:
        raise RuntimeError(f"Invalid video metadata for {path}")
    return width, height, fps, total_frames


def render_variant(source_path, output_path, duration_sec, from_center,
                   to_center, boundary_frame, start_frame, end_frame):
    width, height, fps, _ = read_video_info(source_path)
    crop_width = int(height * 9 / 16)
    if crop_width % 2:
        crop_width += 1
    output_height = height if height % 2 == 0 else height + 1
    output_width = crop_width
    from_x = calculate_crop_box_for_center(
        from_center, width, height, crop_width)[0]
    to_x = calculate_crop_box_for_center(
        to_center, width, height, crop_width)[0]
    duration_frames = max(1, int(round(duration_sec * fps)))

    cap = cv2.VideoCapture(str(source_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = open_writer(output_path, fps, (output_width, output_height))
    positions = []

    for source_frame in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        transition_offset = source_frame - boundary_frame
        if transition_offset < 0:
            crop_x = from_x
        elif duration_sec == 0 or transition_offset >= duration_frames:
            crop_x = to_x
        else:
            crop_x = interpolate_pan_x(
                from_x, to_x, transition_offset, duration_frames)

        cropped = frame[:, crop_x:crop_x + crop_width]
        if output_height != height:
            cropped = cv2.copyMakeBorder(
                cropped, 0, output_height - height, 0, 0,
                cv2.BORDER_REPLICATE)
        cv2.putText(
            cropped,
            f"pan={duration_sec:.2f}s  crop_x={crop_x}",
            (14, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(cropped)
        positions.append(crop_x)

    cap.release()
    writer.release()
    return positions, fps, (output_width, output_height)


def create_comparison(variant_paths, output_path, fps):
    captures = [cv2.VideoCapture(str(path)) for path in variant_paths]
    if not all(cap.isOpened() for cap in captures):
        raise RuntimeError("Could not reopen rendered variants")

    count = len(captures)
    columns = min(2, count)
    rows = math.ceil(count / columns)
    cell_height = 360
    first_width = int(captures[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    first_height = int(captures[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    cell_width = int(round(cell_height * first_width / first_height))
    writer = open_writer(
        output_path, fps, (cell_width * columns, cell_height * rows))

    while True:
        frames = []
        any_frame = False
        for cap in captures:
            ok, frame = cap.read()
            if ok:
                any_frame = True
                frames.append(cv2.resize(frame, (cell_width, cell_height)))
            else:
                frames.append(np.zeros(
                    (cell_height, cell_width, 3), dtype=np.uint8))
        if not any_frame:
            break
        canvas = np.zeros(
            (cell_height * rows, cell_width * columns, 3), dtype=np.uint8)
        for index, frame in enumerate(frames):
            row, column = divmod(index, columns)
            canvas[
                row * cell_height:(row + 1) * cell_height,
                column * cell_width:(column + 1) * cell_width,
            ] = frame
        writer.write(canvas)

    for cap in captures:
        cap.release()
    writer.release()


def read_frame(path, frame_number):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def create_contact_sheet(variants, output_path, fps, boundary_local_frame):
    offsets_sec = [-0.1, 0.0, 0.1, 0.2, 0.4, 0.6]
    cell_height = 240
    sample = read_frame(variants[0][1], 0)
    if sample is None:
        raise RuntimeError("Could not read rendered output for contact sheet")
    cell_width = int(round(cell_height * sample.shape[1] / sample.shape[0]))
    label_height = 34
    sheet = np.zeros(
        ((cell_height + label_height) * len(variants),
         cell_width * len(offsets_sec), 3),
        dtype=np.uint8,
    )

    for row, (duration, path) in enumerate(variants):
        for column, offset in enumerate(offsets_sec):
            frame_number = boundary_local_frame + int(round(offset * fps))
            frame = read_frame(path, frame_number)
            if frame is None:
                continue
            frame = cv2.resize(frame, (cell_width, cell_height))
            y = row * (cell_height + label_height)
            x = column * cell_width
            sheet[y:y + cell_height, x:x + cell_width] = frame
            cv2.putText(
                sheet,
                f"{duration:.2f}s / {offset:+.1f}s",
                (x + 8, y + cell_height + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (235, 238, 244),
                1,
                cv2.LINE_AA,
            )
    cv2.imwrite(str(output_path), sheet)


def main():
    parser = argparse.ArgumentParser(
        description="Render side-by-side pan variants without YOLO.")
    parser.add_argument(
        "--input", type=Path,
        help="Optional real source clip. Omit to generate a synthetic fixture.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("pan-lab-output"))
    parser.add_argument(
        "--boundary-sec", type=float,
        help="Source timestamp where the crop target changes. Defaults to midpoint.")
    parser.add_argument(
        "--from-x", type=float,
        help="Starting crop center in source pixels. Defaults to 25%% width.")
    parser.add_argument(
        "--to-x", type=float,
        help="Ending crop center in source pixels. Defaults to 75%% width.")
    parser.add_argument(
        "--durations", type=parse_durations,
        default=parse_durations("0,0.25,0.4,0.65"),
        help="Comma-separated pan durations (default: 0,0.25,0.4,0.65).")
    parser.add_argument(
        "--window-sec", type=float, default=1.5,
        help="Seconds retained before and after the boundary for real clips.")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.input.resolve() if args.input else output_dir / "source.mp4"
    if args.input is None:
        generate_synthetic_source(source_path)

    width, height, fps, total_frames = read_video_info(source_path)
    total_seconds = total_frames / fps
    boundary_sec = (
        args.boundary_sec if args.boundary_sec is not None else total_seconds / 2)
    if not 0 < boundary_sec < total_seconds:
        raise ValueError(
            f"--boundary-sec must be inside the clip (0..{total_seconds:.2f})")
    boundary_frame = int(round(boundary_sec * fps))
    from_center = args.from_x if args.from_x is not None else width * 0.25
    to_center = args.to_x if args.to_x is not None else width * 0.75

    if args.input:
        start_frame = max(0, boundary_frame - int(args.window_sec * fps))
        end_frame = min(
            total_frames, boundary_frame + int(args.window_sec * fps))
    else:
        start_frame, end_frame = 0, total_frames
    boundary_local_frame = boundary_frame - start_frame

    variants = []
    report = {
        "source": str(source_path),
        "sourceSize": [width, height],
        "fps": fps,
        "sourceBoundarySec": boundary_sec,
        "fromCenterX": from_center,
        "toCenterX": to_center,
        "variants": {},
    }
    for duration in args.durations:
        label = str(duration).replace(".", "_")
        output_path = output_dir / f"pan_{label}s.mp4"
        positions, _, _ = render_variant(
            source_path,
            output_path,
            duration,
            from_center,
            to_center,
            boundary_frame,
            start_frame,
            end_frame,
        )
        variants.append((duration, output_path))
        report["variants"][str(duration)] = {
            "video": str(output_path),
            "cropXByFrame": positions,
        }

    comparison_path = output_dir / "comparison.mp4"
    contact_sheet_path = output_dir / "contact-sheet.jpg"
    create_comparison([path for _, path in variants], comparison_path, fps)
    create_contact_sheet(
        variants, contact_sheet_path, fps, boundary_local_frame)
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Source:        {source_path}")
    print(f"Comparison:    {comparison_path}")
    print(f"Contact sheet: {contact_sheet_path}")
    print(f"Trajectory:    {report_path}")


if __name__ == "__main__":
    main()
