import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image


REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ROOT_DIR = REPO_DIR / 'data' / 'FaceVid-Forensics'
DEFAULT_MANIFEST = DEFAULT_ROOT_DIR / 'manifests' / 'train.json'
VIDEO_SUFFIXES = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Decode videos listed by a FaceVid-Forensics manifest into the frames/ tree expected by training.')
    parser.add_argument('manifest', nargs='?', default=DEFAULT_MANIFEST, type=Path,
                        help=f'Input JSON manifest. Default: {DEFAULT_MANIFEST}')
    parser.add_argument('--root-dir', default=DEFAULT_ROOT_DIR, type=Path,
                        help=f'Dataset root containing videos/. Default: {DEFAULT_ROOT_DIR}')
    parser.add_argument('--workers', default=32, type=int,
                        help='Number of parallel decoding workers. Default: 32.')
    parser.add_argument('--backend', choices=['auto', 'opencv', 'decord'], default='decord',
                        help='Video decoding backend. Default: decord.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Regenerate frames even when the output directory already has enough PNG files.')
    parser.add_argument('--limit', default=None, type=int, help='Decode only the first N manifest rows.')
    return parser.parse_args()


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as file:
        rows = json.load(file)
    if not isinstance(rows, list):
        raise ValueError(f'Manifest must be a JSON array: {path}')
    return rows


def video_path_from_row(root_dir: Path, row: Dict[str, Any]) -> Path:
    relative_video = Path(row['video_path'])
    if relative_video.is_absolute():
        return relative_video
    if not relative_video.parts or relative_video.parts[0] != 'videos':
        raise ValueError(f'Expected video_path under videos/, got: {relative_video}')
    return root_dir / relative_video


def frames_dir_from_row(root_dir: Path, row: Dict[str, Any]) -> Path:
    relative_video = Path(row['video_path'])
    if relative_video.is_absolute():
        if relative_video.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f'Unsupported absolute video path: {relative_video}')
        return root_dir / 'frames' / relative_video.stem
    if not relative_video.parts or relative_video.parts[0] != 'videos':
        raise ValueError(f'Expected video_path under videos/, got: {relative_video}')
    return root_dir / Path('frames', *relative_video.parts[1:])


def clear_existing_pngs(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for path in output_dir.glob('*.png'):
        path.unlink()


def save_frame(frame: Image.Image, output_dir: Path, index: int) -> None:
    frame.save(output_dir / f'{index:06d}.png')


def decode_with_opencv(video_path: Path, output_dir: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f'OpenCV failed to open video: {video_path}')
    saved = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            save_frame(Image.fromarray(frame), output_dir, saved)
            saved += 1
    finally:
        capture.release()
    if saved <= 0:
        raise ValueError(f'OpenCV decoded no frames: {video_path}')
    return saved


def decode_with_decord(video_path: Path, output_dir: Path) -> int:
    from decord import VideoReader, cpu

    reader = VideoReader(str(video_path), ctx=cpu(0))
    for index in range(len(reader)):
        save_frame(Image.fromarray(reader[index].asnumpy()), output_dir, index)
    if len(reader) <= 0:
        raise ValueError(f'Decord decoded no frames: {video_path}')
    return len(reader)


def video_frame_count(video_path: Path, backend: str) -> int:
    if backend == 'decord':
        from decord import VideoReader, cpu

        return len(VideoReader(str(video_path), ctx=cpu(0)))

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f'OpenCV failed to open video: {video_path}')
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if count <= 0:
        raise ValueError(f'OpenCV reported no frames: {video_path}')
    return count


def choose_backend(requested: str) -> str:
    if requested != 'auto':
        return requested
    try:
        import cv2

        return 'opencv'
    except Exception:
        try:
            import decord

            return 'decord'
        except Exception as exc:
            raise RuntimeError('Neither opencv-python nor decord is available for video decoding.') from exc


def decode_one(root_dir: Path, row: Dict[str, Any], backend: str, overwrite: bool) -> str:
    video_path = video_path_from_row(root_dir, row)
    output_dir = frames_dir_from_row(root_dir, row)
    if not video_path.is_file():
        raise FileNotFoundError(f'Video file does not exist: {video_path}')
    count = video_frame_count(video_path, backend)
    if not overwrite and len(list(output_dir.glob('*.png'))) >= count:
        return f'skip {row["video_path"]}'

    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        clear_existing_pngs(output_dir)
    saved = decode_with_decord(video_path, output_dir) if backend == 'decord' else decode_with_opencv(video_path, output_dir)
    return f'decoded {row["video_path"]}: {saved} frames -> {output_dir}'


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir.resolve()
    rows = load_manifest(args.manifest.resolve())
    if args.limit is not None:
        rows = rows[:args.limit]
    backend = choose_backend(args.backend)

    print(f'Manifest: {args.manifest.resolve()}')
    print(f'Root dir: {root_dir}')
    print(f'Backend: {backend}')
    print(f'Videos: {len(rows)}; mode: decode all frames; size: original')

    done = 0
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(decode_one, root_dir, row, backend, args.overwrite)
            for row in rows
        ]
        for future in as_completed(futures):
            try:
                message = future.result()
                done += 1
                if done == 1 or done % 100 == 0:
                    print(f'[{done}/{len(rows)}] {message}')
            except Exception as exc:
                errors.append(str(exc))
                print(f'[error] {exc}')

    print(f'Finished: {done}/{len(rows)} videos processed, {len(errors)} errors.')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
