#!/usr/bin/env python3
"""Fast visual test bench for crop-pan behavior.

This bypasses scene detection and YOLO, but everything after that point is
the production code path: `plan_pan_transitions` decides where pans happen
and `render_output_frame` produces every output pixel, exactly as the
`autocrop` CLI does. A pan that looks right here is the pan that ships.

Two modes:

  * Synthetic / hand-specified: a two-scene TRACK->TRACK plan built from
    --from-x / --to-x / --boundary-sec (default: generated fixture).
  * Replay: --plan <file> written by `autocrop --plan-json`, re-planned for
    each requested --durations value and rendered around a chosen boundary of
    the real clip given by --input.
"""

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import main as autocrop  # noqa: E402


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


def two_scene_plan(from_center, to_center, boundary_frame, total_frames,
                   height):
    """Hand-built TRACK->TRACK plan in the same shape `cli()` produces."""
    return [
        {
            "start_frame": 0,
            "end_frame": boundary_frame,
            "strategy": "TRACK",
            "target_box": [from_center, 0, from_center, height],
        },
        {
            "start_frame": boundary_frame,
            "end_frame": total_frames,
            "strategy": "TRACK",
            "target_box": [to_center, 0, to_center, height],
        },
    ]


def load_plan(path):
    payload = json.loads(Path(path).read_text())
    ratio = payload.get("ratio", "9:16")
    w, h = ratio.split(":")
    autocrop.ASPECT_RATIO = int(w) / int(h)
    return payload


def pick_boundary_frame(scenes):
    """First planned pan, else first TRACK->TRACK boundary, else first cut."""
    for kind in ("pan", "hold"):
        for scene in scenes[1:]:
            if scene.get("boundary_kind") == kind:
                return scene["start_frame"]
    for previous, scene in zip(scenes, scenes[1:]):
        if previous["strategy"] == "TRACK" and scene["strategy"] == "TRACK":
            return scene["start_frame"]
    if len(scenes) > 1:
        return scenes[1]["start_frame"]
    raise ValueError("Plan has a single scene; nothing to pan between.")


def render_variant(source_path, output_path, scenes, duration_sec,
                   width, height, fps, start_frame, end_frame):
    """Re-plan `scenes` for one pan duration and render via production code."""
    plan = copy.deepcopy(scenes)
    autocrop.plan_pan_transitions(
        None, plan, width, height, fps, pan_duration=duration_sec)
    output_width, output_height = autocrop.compute_output_size(height)

    cap = cv2.VideoCapture(str(source_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = open_writer(output_path, fps, (output_width, output_height))
    positions = []
    index = autocrop.scene_index_for_frame(plan, start_frame, 0)

    for frame_number in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        index = autocrop.scene_index_for_frame(plan, frame_number, index)
        scene = plan[index]
        strategy, crop_box = autocrop.resolve_frame_crop(
            scene, frame_number, width, height)
        output = autocrop.render_output_frame(
            frame, scene, frame_number, width, height,
            output_width, output_height)
        crop_x = crop_box[0] if crop_box else None
        label = (
            f"pan={duration_sec:.2f}s  {strategy.lower()}  "
            f"x={'-' if crop_x is None else crop_x}"
        )
        cv2.putText(output, label, (14, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(output)
        positions.append(crop_x)

    cap.release()
    writer.release()
    return positions, autocrop.summarize_pan_plan(plan)


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
        "--plan", type=Path,
        help="Plan JSON from `autocrop --plan-json`. Requires --input. Replays "
             "the real scene plan through the production render path so no "
             "scene detection or YOLO is needed.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("pan-lab-output"))
    parser.add_argument(
        "--boundary-sec", type=float,
        help="Source timestamp where the crop target changes. Defaults to the "
             "midpoint, or to the first planned pan when --plan is given.")
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

    if args.plan:
        if not args.input:
            raise ValueError("--plan requires --input (the clip the plan was made from)")
        plan_payload = load_plan(args.plan)
        scenes = plan_payload["scenes"]
        if [plan_payload.get("width"), plan_payload.get("height")] != [width, height]:
            raise ValueError(
                f"Plan is for {plan_payload.get('width')}x{plan_payload.get('height')} "
                f"but --input is {width}x{height}")
        if args.boundary_sec is not None:
            boundary_frame = int(round(args.boundary_sec * fps))
        else:
            boundary_frame = pick_boundary_frame(scenes)
        boundary_sec = boundary_frame / fps
        from_center = to_center = None
    else:
        boundary_sec = (
            args.boundary_sec if args.boundary_sec is not None else total_seconds / 2)
        boundary_frame = int(round(boundary_sec * fps))
        from_center = args.from_x if args.from_x is not None else width * 0.25
        to_center = args.to_x if args.to_x is not None else width * 0.75
        scenes = two_scene_plan(
            from_center, to_center, boundary_frame, total_frames, height)

    if not 0 < boundary_frame < total_frames:
        raise ValueError(
            f"boundary must be inside the clip (0..{total_seconds:.2f}s)")

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
        "plan": str(args.plan.resolve()) if args.plan else None,
        "sourceSize": [width, height],
        "fps": fps,
        "sourceBoundarySec": boundary_sec,
        "fromCenterX": from_center,
        "toCenterX": to_center,
        "renderedFrames": [start_frame, end_frame],
        "variants": {},
    }
    for duration in args.durations:
        label = str(duration).replace(".", "_")
        output_path = output_dir / f"pan_{label}s.mp4"
        positions, summary = render_variant(
            source_path,
            output_path,
            scenes,
            duration,
            width,
            height,
            fps,
            start_frame,
            end_frame,
        )
        variants.append((duration, output_path))
        report["variants"][str(duration)] = {
            "video": str(output_path),
            "planSummary": summary,
            "cropXByFrame": positions,
        }
        print(f"pan={duration:.2f}s  planned {summary['pan']} pan / "
              f"{summary['hold']} hold / {summary['layout_switch']} layout-switch "
              f"over {summary['track_to_track']} TRACK->TRACK boundaries")

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
