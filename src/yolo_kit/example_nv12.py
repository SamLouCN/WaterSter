"""ctypes example for tightly packed NV12 frames from front_cam.get_nv12()."""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys
import tempfile

import numpy as np


class YoloDetection(C.Structure):
    _fields_ = [
        ("class_id", C.c_int32),
        ("confidence", C.c_float),
        ("x1", C.c_float),
        ("y1", C.c_float),
        ("x2", C.c_float),
        ("y2", C.c_float),
    ]


def load_library(path):
    lib = C.CDLL(str(Path(path).resolve()))
    lib.yolo_init.argtypes = [C.c_char_p]
    lib.yolo_init.restype = C.c_int
    lib.yolo_process_nv12.argtypes = [
        C.POINTER(C.c_uint8), C.c_size_t, C.c_int, C.c_int,
        C.POINTER(YoloDetection), C.c_int, C.POINTER(C.c_int),
    ]
    lib.yolo_process_nv12.restype = C.c_int
    lib.yolo_cleanup.argtypes = []
    lib.yolo_cleanup.restype = C.c_int
    return lib


def process_frame(lib, nv12, capacity=100):
    """Return copied Python values. Library calls must be serialized."""
    if nv12.dtype != np.uint8 or nv12.ndim != 2:
        raise ValueError("Expected a two-dimensional uint8 NV12 array")
    rows, width = nv12.shape
    if rows <= 0 or rows % 3 or width <= 0 or width % 2:
        raise ValueError("Expected packed NV12 shape (height * 3 // 2, width)")
    height = rows * 2 // 3
    if capacity < 0:
        raise ValueError("capacity must not be negative")
    frame = np.ascontiguousarray(nv12)
    results = (YoloDetection * capacity)()
    count = C.c_int(0)
    ret = lib.yolo_process_nv12(
        frame.ctypes.data_as(C.POINTER(C.c_uint8)), frame.nbytes,
        width, height, results, capacity, C.byref(count),
    )
    if ret != 0:
        raise RuntimeError("YOLO processing failed; see C++ stderr")
    if not 0 <= count.value <= capacity:
        raise RuntimeError("Invalid detection count from library")
    detections = []
    for i in range(count.value):
        d = results[i]
        detections.append({
            "class_id": int(d.class_id),
            "confidence": float(d.confidence),
            "xyxy": (float(d.x1), float(d.y1), float(d.x2), float(d.y2)),
            "center": ((d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2),
            "wh": (d.x2 - d.x1, d.y2 - d.y1),
        })
    return detections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default=str(Path(__file__).parent / "build/libyolo_detect.so"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--nv12", required=True, help="Raw, tightly packed NV12 file")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--score-format", required=True, choices=["logits", "probability"],
                        help="Must match the verified model export contract")
    parser.add_argument("--box-format", required=True,
                        choices=["ltrb_grid", "ltrb_pixels", "xyxy_pixels", "cxcywh_pixels"],
                        help="Must match the verified model export contract")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        parser.error("width and height must be positive and even")
    frame = np.fromfile(args.nv12, dtype=np.uint8)
    if frame.size != args.width * args.height * 3 // 2:
        parser.error("NV12 file size does not match width and height")
    frame = frame.reshape(args.height * 3 // 2, args.width)
    config = json.loads(Path(args.config).read_text())
    config.update(score_format=args.score_format, box_format=args.box_format)
    lib = load_library(args.library)
    # Explicit CLI choices apply to a temporary config; original JSON is preserved.
    with tempfile.TemporaryDirectory(prefix="waterster_yolo_") as tmp:
        config_path = Path(tmp) / "config.json"
        config_path.write_text(json.dumps(config))
        if lib.yolo_init(str(config_path).encode()) != 0:
            raise RuntimeError("YOLO initialization failed; see C++ stderr")
    try:
        for detection in process_frame(lib, frame):
            print(detection)
    finally:
        if lib.yolo_cleanup() != 0:
            print("YOLO cleanup incomplete; resources retained for retry", file=sys.stderr)
            raise RuntimeError("YOLO cleanup failed")


if __name__ == "__main__":
    main()
