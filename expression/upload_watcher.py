"""
Uploaded media watcher — scans expression_video/uploaded/ for user media files.
Polling-based, no external dependencies beyond Python stdlib.

SSH users can scp/sftp video or image files into the uploaded/ directory.
File names (without extension) become expression names usable via the existing
POST /expression/{name} API.
"""
import os
import threading
import time
from typing import Dict


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

UPLOADED_EXPRESSIONS: Dict[str, str] = {}
_lock = threading.Lock()


def is_image_path(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in IMAGE_EXTENSIONS


def _media_extension(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def scan_uploaded_dir(base_dir: str, uploaded_subdir: str = "uploaded") -> int:
    """
    Rescan the uploaded directory and update UPLOADED_EXPRESSIONS in-place.
    Returns number of files found.
    """
    upload_dir = os.path.join(base_dir, uploaded_subdir)
    os.makedirs(upload_dir, exist_ok=True)

    found: Dict[str, str] = {}
    try:
        for fname in os.listdir(upload_dir):
            fpath = os.path.join(upload_dir, fname)
            if not os.path.isfile(fpath):
                continue
            ext = _media_extension(fname)
            if ext not in IMAGE_EXTENSIONS and ext not in VIDEO_EXTENSIONS:
                continue
            name = os.path.splitext(fname)[0]
            found[name] = fpath
    except OSError:
        pass

    with _lock:
        UPLOADED_EXPRESSIONS.clear()
        UPLOADED_EXPRESSIONS.update(found)
    return len(found)


def start_watcher(base_dir: str, interval: float = 2.0) -> threading.Thread:
    """
    Start a daemon thread that periodically rescans the uploaded directory.
    Returns the thread for reference.
    """

    def _loop():
        count = scan_uploaded_dir(base_dir)
        print(f"[upload_watcher] initial scan: {count} file(s) in uploaded/")
        while True:
            time.sleep(interval)
            try:
                count = scan_uploaded_dir(base_dir)
            except Exception as e:
                print(f"[upload_watcher] scan error: {e}")

    t = threading.Thread(target=_loop, name="upload-watcher", daemon=True)
    t.start()
    return t
