"""USB MJPEG capture, original JPEG publishing and X5 hardware NV12 decoding.

Web preview uses the camera JPEG bytes unchanged. Hardware decoding uses only
WaterSter/lib/libmultimedia.so; get_nv12() supplies a copy for later algorithms.
Run web_server.py to start the camera and the HTTP stream together.
"""
import ctypes as C
from collections import deque
import logging
import os
from pathlib import Path
import subprocess
import threading
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


class HardwareDecoder:
    def __init__(self, width, height):
        source = ROOT / 'src/mjpeg_decoder.c'
        library = ROOT / 'lib/libfront_mjpeg.so'
        vendor = ROOT / 'lib/libmultimedia.so'
        headers = [ROOT / 'include/hb_media_codec.h',
                   ROOT / 'include/hb_media_basic_types.h']
        for path in [source, vendor, *headers]:
            if not path.is_file():
                raise RuntimeError(f'Required decoder file missing: {path}')
        if not library.exists() or any(p.stat().st_mtime > library.stat().st_mtime
                                       for p in [source, vendor, *headers]):
            temporary = library.with_name(f'.{library.name}.{os.getpid()}.tmp')
            try:
                subprocess.run([
                    'cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-shared', '-fPIC',
                    '-I', str(ROOT / 'include'), str(source), str(vendor),
                    '-Wl,-rpath,$ORIGIN', '-o', str(temporary),
                ], check=True, capture_output=True, text=True, timeout=60)
                temporary.replace(library)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f'Decoder build failed: {exc.stderr}') from exc
            finally:
                temporary.unlink(missing_ok=True)
        # Load the supplied vendor library explicitly before loading our ABI bridge.
        self.vendor = C.CDLL(str(vendor), mode=C.RTLD_GLOBAL)
        self.lib = C.CDLL(str(library))
        self.lib.front_decoder_open.argtypes = [C.c_int, C.c_int, C.POINTER(C.c_int)]
        self.lib.front_decoder_open.restype = C.c_void_p
        self.lib.front_decoder_decode.argtypes = [
            C.c_void_p, C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t]
        self.lib.front_decoder_decode.restype = C.c_int
        self.lib.front_decoder_close.argtypes = [C.c_void_p]
        self.lib.front_decoder_close.restype = None
        error = C.c_int()
        self.handle = self.lib.front_decoder_open(width, height, C.byref(error))
        if not self.handle:
            raise RuntimeError(f'X5 decoder initialization failed: {error.value}')
        self.image = np.empty((height * 3 // 2, width), dtype=np.uint8)

    def decode(self, jpeg):
        rc = self.lib.front_decoder_decode(
            self.handle, C.c_char_p(jpeg), len(jpeg),
            C.c_void_p(self.image.ctypes.data), self.image.nbytes)
        if rc:
            raise RuntimeError(f'X5 JPEG decode failed: {rc}')
        return self.image

    def close(self):
        if self.handle:
            self.lib.front_decoder_close(self.handle)
            self.handle = None


class FrontCamera:
    def __init__(self, device='/dev/video0', width=640, height=480, fps=60,
                 hardware_decode=True):
        self.device, self.width, self.height, self.fps = device, width, height, fps
        self.hardware_decode = hardware_decode
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.thread = None
        self.jpeg = self.nv12 = None
        self.sequence = 0
        self.last_frame_at = self.started_at = 0.0
        self.capture_fps = 0.0
        self.frame_times = deque(maxlen=1000)
        self.actual = {}
        self.error = None
        self.cap = self.decoder = None

    def start(self):
        try:
            self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                raise RuntimeError(f'Cannot open {self.device}')
            for prop, value in [
                (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')),
                (cv2.CAP_PROP_FRAME_WIDTH, self.width),
                (cv2.CAP_PROP_FRAME_HEIGHT, self.height),
                (cv2.CAP_PROP_FPS, self.fps),
                (cv2.CAP_PROP_CONVERT_RGB, 0),
                # Keep two V4L2 buffers queued; one halves FPS on this UVC camera.
                (cv2.CAP_PROP_BUFFERSIZE, 2),
            ]:
                if not self.cap.set(prop, value):
                    LOG.warning('Camera did not accept property %s=%s', prop, value)
            fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            fmt = ''.join(chr((fourcc >> (8*i)) & 255) for i in range(4))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.actual = dict(device=self.device, width=self.width, height=self.height,
                               format=fmt, requested_fps=self.fps,
                               negotiated_fps=self.cap.get(cv2.CAP_PROP_FPS),
                               hardware_decode=self.hardware_decode)
            if fmt != 'MJPG':
                raise RuntimeError(f'Camera negotiated {fmt!r}, expected MJPG')
            if self.hardware_decode:
                self.decoder = HardwareDecoder(self.width, self.height)
            self.started_at = time.monotonic()
            self.thread = threading.Thread(target=self._capture, name='front-camera', daemon=True)
            self.thread.start()
            with self.condition:
                self.condition.wait_for(lambda: self.jpeg is not None or self.error is not None, 10)
                if self.jpeg is None:
                    raise RuntimeError(self.error or 'Timed out waiting for camera frame')
            LOG.info('Camera ready: %s', self.actual)
            return self
        except BaseException:
            self.close()
            raise

    def _capture(self):
        try:
            while not self.stop_event.is_set():
                ok, raw = self.cap.read()
                if not ok or raw is None:
                    raise RuntimeError('Camera read failed or device disconnected')
                if raw.dtype != np.uint8 or raw.ndim != 2 or raw.shape[0] != 1:
                    raise RuntimeError('V4L2 returned decoded pixels instead of raw JPEG')
                jpeg = raw.tobytes()
                if not jpeg.startswith(b'\xff\xd8'):
                    raise RuntimeError('Camera frame has no JPEG SOI marker')
                end = jpeg.rfind(b'\xff\xd9')
                if end < 0:
                    LOG.warning('Dropping incomplete JPEG frame')
                    continue
                jpeg = jpeg[:end + 2]
                # Keep the original JPEG for HTTP; never decode/re-encode preview.
                with self.condition:
                    if self.decoder:
                        self.nv12 = self.decoder.decode(jpeg)
                    self.jpeg = jpeg
                    self.sequence += 1
                    self.last_frame_at = time.monotonic()
                    self.frame_times.append(self.last_frame_at)
                    while len(self.frame_times) > 2 and self.frame_times[0] < self.last_frame_at - 2:
                        self.frame_times.popleft()
                    elapsed = self.last_frame_at - self.frame_times[0]
                    self.capture_fps = ((len(self.frame_times) - 1) / elapsed
                                        if elapsed > 0 else 0.0)
                    self.condition.notify_all()
        except Exception as exc:
            LOG.exception('Camera stopped')
            with self.condition:
                self.error = str(exc)
        finally:
            self.stop_event.set()
            self.cap.release()
            if self.decoder:
                self.decoder.close()
            with self.condition:
                self.condition.notify_all()

    def wait_frame(self, previous=0, timeout=3):
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence > previous or self.stop_event.is_set(), timeout)
            if self.sequence > previous:
                return self.sequence, self.jpeg
            return None

    def get_nv12(self):
        """Return (sequence, NV12 copy); safe to retain after the next capture."""
        with self.condition:
            return self.sequence, None if self.nv12 is None else self.nv12.copy()

    def status(self):
        with self.condition:
            age = time.monotonic()-self.last_frame_at if self.last_frame_at else None
            fresh = age is not None and age < 3 and not self.stop_event.is_set()
            return dict(self.actual, frames=self.sequence,
                        capture_fps=round(self.capture_fps, 2) if fresh else 0.0,
                        frame_age_seconds=None if age is None else round(age, 3),
                        running=not self.stop_event.is_set(), error=self.error)

    def close(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                LOG.warning('Camera driver is still waiting for a frame')
        else:
            if self.cap:
                self.cap.release()
            if self.decoder:
                self.decoder.close()
