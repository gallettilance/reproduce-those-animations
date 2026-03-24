import argparse
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading

import cv2
import numpy as np

cv2.setUseOptimized(True)

_SENTINEL = object()


def parse_rgb_hex(s):
    s = s.strip().lstrip("#")
    if len(s) == 8:
        s = s[:6]
    if len(s) != 6:
        raise argparse.ArgumentTypeError(
            f"hex color must be 6 digits (optional # prefix, optional 2-digit alpha), got {s!r}"
        )
    try:
        return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid hex color: {s!r}") from e


def rgb_to_bgr(rgb):
    r, g, b = rgb
    return (b, g, r)


def hsv_range_from_rgb_hex(hex_str, tol_h, tol_s, tol_v):
    """OpenCV HSV: H 0–179, S and V 0–255."""
    r, g, b = parse_rgb_hex(hex_str)
    bgr = np.uint8([[[b, g, r]]])
    h, s, v = (int(x) for x in cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0])

    if s <= 40:
        lower = [0, max(0, s - tol_s), max(0, v - tol_v)]
        upper = [179, min(255, s + tol_s), min(255, v + tol_v)]
    else:
        lower = [max(0, h - tol_h), max(0, s - tol_s), max(0, v - tol_v)]
        upper = [min(179, h + tol_h), min(255, s + tol_s), min(255, v + tol_v)]

    return lower, upper


def _decode_fourcc(code):
    code = int(code)
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


def _fourcc_int(tag_str):
    return cv2.VideoWriter_fourcc(*tag_str)


def _writer_fourcc_for_output(output_path, source_fourcc_int):
    """Pick a container-friendly fourcc. MP4 rejects the raw 'h264' tag some decoders report."""
    ext = os.path.splitext(output_path)[1].lower()

    if ext in (".mp4", ".m4v", ".mov"):
        # Matches FFmpeg/OpenCV behavior for H.264 in MP4 (avoids h264→avc1 fallback warning).
        return _fourcc_int("avc1")
    return source_fourcc_int


def _parse_fraction_fps(s):
    if not s or s == "0/0" or s == "N/A":
        return None
    if "/" in s:
        a, _, b = s.partition("/")
        try:
            x, y = float(a), float(b)
            return x / y if y else None
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _ffprobe_video_timing(path):
    """Return duration (seconds), frame count, and frame rates from ffprobe, or None."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(r.stdout or "{}")
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError):
        return None
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    if not streams:
        return None
    st = streams[0]

    dur_raw = st.get("duration")
    if dur_raw in (None, "N/A"):
        dur_raw = fmt.get("duration")
    duration = None
    if dur_raw not in (None, "N/A"):
        try:
            duration = float(dur_raw)
        except (TypeError, ValueError):
            pass

    nb_raw = st.get("nb_frames")
    nb_frames = None
    if nb_raw not in (None, "N/A"):
        try:
            nb_frames = int(nb_raw)
        except (TypeError, ValueError):
            pass

    return {
        "duration": duration,
        "nb_frames": nb_frames,
        "avg_fps": _parse_fraction_fps(st.get("avg_frame_rate")),
        "r_fps": _parse_fraction_fps(st.get("r_frame_rate")),
    }


def _choose_writer_fps(video_path, cap, fps_override=None):
    """
    FPS for VideoWriter: OpenCV's CAP_PROP_FPS is often rounded or wrong (duration drift).
    Prefer ffprobe (nb_frames/duration, then avg_frame_rate).
    """
    if fps_override is not None and fps_override > 1e-6:
        return float(fps_override), "(--fps)"

    meta = _ffprobe_video_timing(video_path)
    fps_cv = cap.get(cv2.CAP_PROP_FPS)

    if meta:
        d = meta.get("duration")
        nb = meta.get("nb_frames")
        avg = meta.get("avg_fps")
        rf = meta.get("r_fps")

        if nb and d and d > 1e-6:
            from_nb = nb / d
            if from_nb > 1e-3:
                if avg and abs(from_nb - avg) / max(avg, 0.01) <= 0.03:
                    return avg, "ffprobe (avg_frame_rate)"
                return from_nb, "ffprobe (nb_frames/duration)"

        if avg and avg > 1e-3:
            return avg, "ffprobe (avg_frame_rate)"
        if rf and rf > 1e-3:
            return rf, "ffprobe (r_frame_rate)"

    if fps_cv and fps_cv > 1e-3:
        return float(fps_cv), "OpenCV CAP_PROP_FPS"

    return 30.0, "default 30 (no reliable fps found)"


def _container_can_carry_muxed_audio(output_path):
    ext = os.path.splitext(output_path)[1].lower()
    return ext in (".mp4", ".m4v", ".mov", ".mkv", ".avi")


def _mux_video_and_optional_audio(
    video_only_path, audio_source_path, output_path, quiet=False
):
    """Mux re-encoded video with audio from the original file (stream copy)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False, "ffmpeg not found in PATH; cannot attach audio"

    loglevel = "error" if quiet else "warning"
    cmd = [
        ffmpeg,
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        loglevel,
        "-i",
        video_only_path,
        "-i",
        audio_source_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        output_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        return True, None

    # Older ffmpeg without optional maps, or other map issues: video-only to output.
    fallback = [
        ffmpeg,
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        loglevel,
        "-i",
        video_only_path,
        "-c:v",
        "copy",
        "-an",
        output_path,
    ]
    r2 = subprocess.run(fallback, capture_output=True, text=True)
    if r2.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()
        err2 = (r2.stderr or r2.stdout or "").strip()
        detail = err or err2 or "ffmpeg mux and fallback both failed"
        return False, detail

    return (
        True,
        "Video saved without copied audio (first ffmpeg pass failed; "
        "upgrade ffmpeg or check that the source has an audio stream).",
    )


def _replace_color_inplace(frame, lower_color, upper_color, repl):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    frame[mask > 0] = repl


def _progress_printer(total_frames, quiet):
    """Thread-safe progress lines; returns (on_frame(frame_idx), get_last_frame_idx())."""
    lock = threading.Lock()
    report_every = max(1, total_frames // 20) if total_frames else 100
    next_at = 1
    last_idx = 0

    def on_frame(frame_idx):
        nonlocal next_at, last_idx
        if quiet:
            last_idx = frame_idx
            return
        with lock:
            last_idx = frame_idx
            if frame_idx == next_at or (
                total_frames is not None and frame_idx == total_frames
            ):
                if total_frames:
                    pct = 100.0 * frame_idx / total_frames
                    print(f"  {frame_idx}/{total_frames} frames ({pct:.1f}%)")
                else:
                    print(f"  {frame_idx} frames…")
                next_at = frame_idx + report_every

    def get_last():
        with lock:
            return last_idx

    return on_frame, get_last


def _run_pipelined(
    cap,
    out,
    lower_color,
    upper_color,
    repl,
    total_frames,
    quiet,
    queue_size,
):
    q_read = queue.Queue(maxsize=queue_size)
    q_write = queue.Queue(maxsize=queue_size)
    on_frame, get_last = _progress_printer(total_frames, quiet)

    def reader():
        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                idx += 1
                q_read.put((idx, frame))
        finally:
            q_read.put(_SENTINEL)

    def processor():
        while True:
            item = q_read.get()
            if item is _SENTINEL:
                q_write.put(_SENTINEL)
                break
            idx, frame = item
            _replace_color_inplace(frame, lower_color, upper_color, repl)
            q_write.put((idx, frame))

    def writer():
        while True:
            item = q_write.get()
            if item is _SENTINEL:
                break
            idx, frame = item
            out.write(frame)
            on_frame(idx)

    t_r = threading.Thread(target=reader, name="reader")
    t_p = threading.Thread(target=processor, name="processor")
    t_w = threading.Thread(target=writer, name="writer")
    t_r.start()
    t_p.start()
    t_w.start()
    t_r.join()
    t_p.join()
    t_w.join()
    return get_last()


def change_color(
    video_path,
    output_path,
    target_color,
    replacement_bgr,
    quiet=False,
    pipeline=False,
    queue_size=2,
    copy_audio=True,
    fps_override=None,
):
    if not quiet:
        print(f"Input:  {video_path}")
        print(f"Output: {output_path}")

    use_tmp = (
        copy_audio
        and _container_can_carry_muxed_audio(output_path)
        and shutil.which("ffmpeg") is not None
    )
    tmp_video_path = None
    if use_tmp:
        out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
        fd, tmp_video_path = tempfile.mkstemp(
            suffix=".mp4", prefix="change_color_vid_", dir=out_dir
        )
        os.close(fd)
        video_writer_path = tmp_video_path
        if not quiet:
            print(
                "Writing video-only to a temp file, then muxing audio with ffmpeg "
                f"({os.path.basename(tmp_video_path)})."
            )
    else:
        video_writer_path = output_path
        if copy_audio and _container_can_carry_muxed_audio(output_path) and not quiet:
            print(
                "Note: install ffmpeg and ensure it is on PATH to copy audio into the output; "
                "otherwise the result is video-only (OpenCV does not write audio)."
            )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open input video: {video_path}")

    source_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    writer_fourcc = _writer_fourcc_for_output(output_path, source_fourcc)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cv = cap.get(cv2.CAP_PROP_FPS)
    frame_rate, fps_source = _choose_writer_fps(video_path, cap, fps_override)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None

    if not quiet:
        cv_line = (
            f"OpenCV reported fps: {fps_cv:.6g}"
            if fps_cv and fps_cv > 1e-3
            else "OpenCV reported fps: (none)"
        )
        print(
            f"Video: {frame_width}x{frame_height}; writer fps: {frame_rate:.6g} ({fps_source})"
        )
        print(f"  {cv_line}")
        print(f"Input fourcc: {_decode_fourcc(source_fourcc)!r}")
        if writer_fourcc != source_fourcc:
            print(
                f"Writer fourcc: {_decode_fourcc(writer_fourcc)!r} "
                f"(chosen for {os.path.splitext(output_path)[1] or 'output'} container)"
            )
        else:
            print(f"Writer fourcc: {_decode_fourcc(writer_fourcc)!r}")
        if total_frames is not None:
            print(f"Reported frame count: {total_frames}")
        else:
            print("Frame count: unknown (progress by frame number only)")

    out = cv2.VideoWriter(
        video_writer_path, writer_fourcc, frame_rate, (frame_width, frame_height)
    )
    if not out.isOpened():
        cap.release()
        if tmp_video_path:
            try:
                os.remove(tmp_video_path)
            except OSError:
                pass
        raise SystemExit(
            f"Could not open VideoWriter for {video_writer_path!r} "
            f"(codec {_decode_fourcc(writer_fourcc)!r}, {frame_width}x{frame_height} @ {frame_rate} fps)."
        )

    lower_color = np.array(target_color[0], dtype=np.uint8)
    upper_color = np.array(target_color[1], dtype=np.uint8)
    repl = np.array(replacement_bgr, dtype=np.uint8)

    if not quiet:
        print(f"HSV range: lower {target_color[0]}  upper {target_color[1]}")
        print(f"Replacement BGR: {list(replacement_bgr)}")
        if pipeline:
            print(f"Pipeline: on (decode / process / encode overlapped, queue={queue_size})")
        print("Processing frames…")

    if pipeline:
        frame_idx = _run_pipelined(
            cap,
            out,
            lower_color,
            upper_color,
            repl,
            total_frames,
            quiet,
            max(1, queue_size),
        )
    else:
        frame_idx = 0
        report_every = max(1, total_frames // 20) if total_frames else 100
        next_report_at = 1

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if not quiet:
                if frame_idx == next_report_at or (
                    total_frames is not None and frame_idx == total_frames
                ):
                    if total_frames:
                        pct = 100.0 * frame_idx / total_frames
                        print(f"  {frame_idx}/{total_frames} frames ({pct:.1f}%)")
                    else:
                        print(f"  {frame_idx} frames…")
                    next_report_at = frame_idx + report_every

            _replace_color_inplace(frame, lower_color, upper_color, repl)
            out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if use_tmp and tmp_video_path:
        if not quiet:
            print("Muxing audio from source into final file…")
        ok, warn = _mux_video_and_optional_audio(
            tmp_video_path, video_path, output_path, quiet=quiet
        )
        if not ok:
            try:
                shutil.copyfile(tmp_video_path, output_path)
            except OSError as e:
                try:
                    os.remove(tmp_video_path)
                except OSError:
                    pass
                raise SystemExit(
                    f"ffmpeg failed and could not copy temp video to output: {warn} ({e})"
                ) from e
            try:
                os.remove(tmp_video_path)
            except OSError:
                pass
            raise SystemExit(
                f"ffmpeg could not mux audio. Video-only file written to {output_path!r}. "
                f"Detail: {warn}"
            )
        try:
            os.remove(tmp_video_path)
        except OSError:
            pass
        if warn and not quiet:
            print(f"Warning: {warn}")

    if not quiet:
        dur_s = frame_idx / frame_rate if frame_rate > 1e-9 else 0.0
        print(
            f"Done. Wrote {frame_idx} frames to {output_path} "
            f"(~{dur_s:.3f}s at {frame_rate:.6g} fps)."
        )


if __name__ == "__main__":
    default_lower = [0, 0, 200]
    default_upper = [179, 50, 255]
    default_repl_bgr = [102, 220, 255]

    parser = argparse.ArgumentParser(
        description="Replace a color range in each frame of a video (HSV mask, BGR replacement)."
    )
    parser.add_argument("input", type=str, help="Input video path")
    parser.add_argument("output", type=str, help="Output video path")
    parser.add_argument(
        "--target-hex",
        type=str,
        metavar="RRGGBB",
        help="Color to match (e.g. b7b7b7 or #b7b7b7ff). Implies full H range when saturation is low (grays).",
    )
    parser.add_argument(
        "--replace-hex",
        type=str,
        metavar="RRGGBB",
        help="Replacement color as hex (e.g. d9d2e9). Written as BGR to frames.",
    )
    parser.add_argument(
        "--tol-h",
        type=int,
        default=10,
        metavar="N",
        help="Hue half-width (0–179) when saturation is not low (default: 10)",
    )
    parser.add_argument(
        "--tol-s",
        type=int,
        default=40,
        metavar="N",
        help="Saturation half-width (default: 40)",
    )
    parser.add_argument(
        "--tol-v",
        type=int,
        default=25,
        metavar="N",
        help="Value half-width (default: 25)",
    )
    parser.add_argument(
        "--target-lower",
        type=int,
        nargs=3,
        metavar=("H", "S", "V"),
        help="Manual HSV lower bound (overrides --target-hex if both given; use with --target-upper)",
    )
    parser.add_argument(
        "--target-upper",
        type=int,
        nargs=3,
        metavar=("H", "S", "V"),
        help="Manual HSV upper bound",
    )
    parser.add_argument(
        "--replacement-bgr",
        type=int,
        nargs=3,
        metavar=("B", "G", "R"),
        help="Manual replacement B,G,R (overrides --replace-hex if both given)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress and configuration messages",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Overlap decode, color pass, and encode on threads (often faster on large files)",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=2,
        metavar="N",
        help="Frames buffered between pipeline stages (default: 2; higher uses more RAM)",
    )
    parser.add_argument(
        "--no-copy-audio",
        action="store_true",
        help="Do not mux audio from the input (video-only; writes directly to the output path)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        metavar="RATE",
        help="Force output frame rate (fixes duration drift if ffprobe/OpenCV guess is wrong)",
    )

    args = parser.parse_args()

    has_hex = args.target_hex is not None or args.replace_hex is not None
    if has_hex:
        if args.target_hex is None or args.replace_hex is None:
            parser.error("--target-hex and --replace-hex must be used together")
        target = hsv_range_from_rgb_hex(args.target_hex, args.tol_h, args.tol_s, args.tol_v)
        replacement_bgr = list(rgb_to_bgr(parse_rgb_hex(args.replace_hex)))
    else:
        target = (default_lower, default_upper)
        replacement_bgr = default_repl_bgr

    if args.target_lower is not None and args.target_upper is not None:
        target = (list(args.target_lower), list(args.target_upper))
    elif args.target_lower is not None or args.target_upper is not None:
        parser.error("--target-lower and --target-upper must be used together")

    if args.replacement_bgr is not None:
        replacement_bgr = list(args.replacement_bgr)

    if not args.quiet:
        print("Color replacement settings (resolved):")
        if args.target_hex is not None:
            print(f"  target hex: {args.target_hex!r}  replace hex: {args.replace_hex!r}")
            print(f"  HSV tolerances: h={args.tol_h} s={args.tol_s} v={args.tol_v}")
        elif args.target_lower is not None:
            print("  manual HSV bounds (--target-lower / --target-upper)")
        else:
            print("  default HSV mask and replacement (no hex / manual bounds)")
        print()

    change_color(
        args.input,
        args.output,
        target,
        replacement_bgr,
        quiet=args.quiet,
        pipeline=args.pipeline,
        queue_size=args.queue_size,
        copy_audio=not args.no_copy_audio,
        fps_override=args.fps,
    )
