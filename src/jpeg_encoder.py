"""JPEG encoding on the X5 JPU via WaterSter's libmultimedia.so.

One worker owns each encoder. Copy its returned array before the next encode.
"""
import ctypes as C
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# OpenCV I420 uses limited-range video YUV; JPEG/JPU expects full-range YCbCr.
_LUMA = np.clip(np.rint((np.arange(256) - 16) * 255 / 219), 0, 255).astype(np.uint8)
_CHROMA = np.clip(np.rint((np.arange(256) - 128) * 255 / 224 + 128), 0, 255).astype(np.uint8)


def bgr_to_jpeg_i420(image):
    i420 = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
    height = image.shape[0]
    cv2.LUT(i420[:height], _LUMA, dst=i420[:height])
    cv2.LUT(i420[height:], _CHROMA, dst=i420[height:])
    return i420


class HardwareJpegEncoder:
    backend = 'x5-jpu/libmultimedia.so'

    def __init__(self, width, height, quality=85):
        self.handle = None
        if (width <= 0 or height <= 0 or width % 16 or height % 8 or
                width > 8192 or height > 8192 or not 1 <= quality <= 100):
            raise ValueError('JPU requires width aligned to 16, height to 8, dimensions <=8192, quality 1..100')
        source = ROOT / 'src/jpeg_encoder.c'
        library = ROOT / 'lib/libwaterster_jpeg.so'
        vendor = ROOT / 'lib/libmultimedia.so'
        dependencies = [source, vendor, ROOT / 'include/hb_media_codec.h',
                        ROOT / 'include/hb_media_basic_types.h']
        for path in dependencies:
            if not path.is_file():
                raise RuntimeError(f'Required JPEG encoder file missing: {path}')
        if not library.exists() or any(p.stat().st_mtime > library.stat().st_mtime
                                       for p in dependencies):
            temporary = library.with_name(f'.{library.name}.{os.getpid()}.tmp')
            try:
                subprocess.run([
                    'cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-shared', '-fPIC',
                    '-I', str(ROOT / 'include'), str(source), str(vendor),
                    '-Wl,-rpath,$ORIGIN', '-o', str(temporary),
                ], check=True, capture_output=True, text=True, timeout=60)
                temporary.replace(library)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f'JPEG encoder build failed: {exc.stderr}') from exc
            finally:
                temporary.unlink(missing_ok=True)
        self.vendor = C.CDLL(str(vendor), mode=C.RTLD_GLOBAL)
        self.lib = C.CDLL(str(library))
        self.lib.jpeg_encoder_open.argtypes = [C.c_int, C.c_int, C.c_int, C.POINTER(C.c_int)]
        self.lib.jpeg_encoder_open.restype = C.c_void_p
        self.lib.jpeg_encoder_encode.argtypes = [
            C.c_void_p, C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)]
        self.lib.jpeg_encoder_encode.restype = C.c_int
        self.lib.jpeg_encoder_close.argtypes = [C.c_void_p]
        self.lib.jpeg_encoder_close.restype = None
        self.width, self.height = width, height
        self.output = np.empty(width * height * 3 + 65536, np.uint8)
        error = C.c_int()
        self.handle = self.lib.jpeg_encoder_open(width, height, quality, C.byref(error))
        if not self.handle:
            raise RuntimeError(f'X5 JPEG encoder initialization failed: {error.value}')

    def encode(self, i420):
        if not self.handle:
            raise RuntimeError('JPEG encoder is closed')
        if (i420.dtype != np.uint8 or not i420.flags.c_contiguous or
                i420.shape != (self.height * 3 // 2, self.width)):
            raise ValueError('Expected contiguous uint8 I420 matching encoder dimensions')
        written = C.c_size_t()
        rc = self.lib.jpeg_encoder_encode(
            self.handle, C.c_void_p(i420.ctypes.data), i420.nbytes,
            C.c_void_p(self.output.ctypes.data), self.output.nbytes, C.byref(written))
        if rc:
            self.close()
            raise RuntimeError(f'X5 JPEG encode failed: {rc}')
        return self.output[:written.value]

    def close(self):
        if self.handle:
            self.lib.jpeg_encoder_close(self.handle)
            self.handle = None
