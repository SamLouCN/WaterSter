"""HTTP backend publishing YOLO-annotated MJPEG from a shared camera.

Start: python3 src/yolo_test_task --host 0.0.0.0 --port 8080
Page: http://192.168.127.10:8080/ (or the existing Nginx entry on port 80)
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
from pathlib import Path
import socket
import time
from urllib.parse import parse_qs, urlsplit
import uuid

from yolo_test_task import YoloTestTask
from yolo_test_task.timing import elapsed_ms

LOG = logging.getLogger(__name__)


class CameraServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, camera, preview_enabled=True):
        self.camera = camera
        self.preview = YoloTestTask(camera, preview_enabled=preview_enabled)
        super().__init__(server_address, Handler)

    def serve_forever(self, poll_interval=.5):
        try:
            self.preview.start()
            self.preview_started_at = time.monotonic()
            super().serve_forever(poll_interval=poll_interval)
        finally:
            self.preview.close()

    def server_close(self):
        try:
            self.preview.close()
        finally:
            super().server_close()

    def service_actions(self):
        # Let the task launcher restart a failed decoder in a fresh process.
        # Do not take the camera lock here: a stuck decoder may hold that lock.
        camera = self.camera
        preview = self.preview
        error = camera.error or preview.error
        if not error and (camera.stop_event.is_set() or preview.stop_event.is_set()):
            error = 'Camera or YOLO worker stopped'
        now = time.monotonic()
        if not error and camera.last_frame_at and now - camera.last_frame_at > 10:
            error = 'Camera has not published a frame for 10 seconds'
        last_preview = preview.last_frame_at or getattr(self, 'preview_started_at', now)
        if not error and now - last_preview > 10:
            error = 'YOLO has not published a frame for 10 seconds'
        if error:
            preview.timing.write('backend_restart_required', error=error)
            raise RuntimeError(error)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            self.connection.settimeout(3)
            if urlsplit(self.path).path != '/client-metrics':
                self.send_error(404)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096:
                    self.send_error(413 if length > 4096 else 400)
                    return
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('Expected an object')
                session = payload.get('session')
                if not isinstance(session, str) or not 1 <= len(session) <= 80:
                    raise ValueError('Invalid session')
                fields = {}
                for key in ('sequence', 'displayed_frames', 'dropped_frames',
                            'reconnects', 'display_fps', 'decode_ms', 'draw_ms',
                            'display_idle_ms'):
                    value = payload.get(key)
                    if value is None and key == 'display_idle_ms':
                        fields[key] = None
                        continue
                    if (type(value) not in (int, float) or not math.isfinite(value)
                            or not 0 <= value <= 1e15):
                        raise ValueError('Invalid numeric metric')
                    if key in ('sequence', 'displayed_frames', 'dropped_frames', 'reconnects') and int(value) != value:
                        raise ValueError('Invalid frame counter')
                    fields[key] = value
                frame_run = payload.get('frame_run', '')
                if not isinstance(frame_run, str) or len(frame_run) > 80:
                    raise ValueError('Invalid frame run')
                error = payload.get('error', '')
                if not isinstance(error, str) or len(error) > 300:
                    raise ValueError('Invalid error description')
            except (ValueError, OverflowError):
                self.send_error(400, 'Invalid client metrics')
                return
            self.server.preview.timing.write(
                'client_frame', session=session, frame_run=frame_run,
                client=self.client_address[0], error=error,
                user_agent=self.headers.get('User-Agent', '')[:300], **fields)
            self.send_response(204)
            self.send_header('Content-Length', '0')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def do_GET(self):
        try:
            self._do_GET()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            # Page reloads and disconnected clients may interrupt any response.
            pass

    def _do_GET(self):
        self.connection.settimeout(5)
        path = urlsplit(self.path).path
        if path in ('/', '/index.html'):
            try:
                data = (Path(__file__).resolve().parents[1] / 'web/index.html').read_bytes()
            except OSError:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)
            return
        if path == '/detections':
            result = self.server.preview.result()
            age = result['age_seconds']
            ready = (result['running'] and not result['error']
                     and result.get('sequence', 0) > 0 and age is not None and age < 3)
            data = json.dumps(result).encode('utf-8')
            self.send_response(200 if ready else 503)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)
            return
        if path in ('/frame.jpg', '/stream.mjpg') and not self.server.preview.preview_enabled:
            self.send_error(503, 'JPEG preview disabled by --detections-only; use /detections')
            return
        if path == '/health':
            status = self.server.preview.status()
            age = status['frame_age_seconds']
            healthy = (status['running'] and not status['error']
                       and status['processed_frames'] > 0 and age is not None and age < 3)
            data = json.dumps(status).encode('utf-8')
            self.send_response(200 if healthy else 503)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)
            return
        if path == '/frame.jpg':
            preview = self.server.preview
            packet = preview.wait_frame(include_timing=True)
            if packet is None or preview.stop_event.is_set():
                self.send_error(503, 'YOLO preview unavailable')
                return
            sequence, jpeg, metadata = packet
            started = time.monotonic()
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(jpeg)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Frame-Sequence', str(sequence))
            self.send_header('X-Run-Id', preview.timing.run_id)
            self.end_headers()
            self.wfile.write(jpeg)
            finished = time.monotonic()
            preview.timing.write('snapshot_frame', sequence=sequence,
                                 source_sequence=metadata['source_sequence'],
                                 socket_write_ms=elapsed_ms(started, finished),
                                 source_to_write_done_ms=elapsed_ms(metadata['source_ready_at'], finished))
            return
        if path != '/stream.mjpg':
            self.send_error(404)
            return
        try:
            fps = float(parse_qs(urlsplit(self.path).query).get('fps', ['30'])[0])
            if not 1 <= fps <= 60:
                raise ValueError('fps outside 1..60')
        except ValueError:
            self.send_error(400, 'fps must be between 1 and 60')
            return
        self.connection.settimeout(2)
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
        preview = self.server.preview
        timing = preview.timing
        stream_id = uuid.uuid4().hex[:12]
        client = f'{self.client_address[0]}:{self.client_address[1]}'
        timing.write('stream_open', stream=stream_id, client=client, fps_limit=fps)
        wait_started = time.monotonic()
        packet = preview.wait_frame(include_timing=True)
        packet_at = time.monotonic()
        wait_ms = elapsed_ms(wait_started, packet_at)
        if packet is None or preview.stop_event.is_set():
            timing.write('stream_close', stream=stream_id, client=client,
                         reason='preview_unavailable', sent_frames=0)
            self.send_error(503, 'YOLO preview unavailable')
            return
        previous = 0
        sent_frames = 0
        close_reason = 'preview_stopped'
        pending = None
        send_started = None
        try:
            headers_at = time.monotonic()
            self.send_response(200)
            # A fetch reader parses the framing itself. Generic binary MIME
            # avoids browsers handing multipart/x-mixed-replace to their
            # image-document loader instead of exposing a readable body.
            fetch_transport = parse_qs(urlsplit(self.path).query).get('transport') == ['fetch']
            self.send_header('Content-Type', 'application/octet-stream' if fetch_transport
                             else 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            self.send_header('X-Accel-Buffering', 'no')
            self.end_headers()
            timing.write('stream_headers', stream=stream_id, client=client,
                         headers_ms=elapsed_ms(headers_at, time.monotonic()))
            while packet is not None and not preview.stop_event.is_set():
                sequence, jpeg, metadata = packet
                pending = dict(stream=stream_id, client=client, sequence=sequence,
                               source_sequence=metadata['source_sequence'],
                               output_skipped=max(0, sequence - previous - 1) if previous else 0,
                               frame_wait_ms=wait_ms, jpeg_bytes=len(jpeg))
                build_at = time.monotonic()
                payload = (
                    b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
                    str(len(jpeg)).encode() + b'\r\nX-Frame-Sequence: ' +
                    str(sequence).encode() + b'\r\nX-Source-Sequence: ' +
                    str(metadata['source_sequence']).encode() + b'\r\nX-Run-Id: ' +
                    timing.run_id.encode() + b'\r\n\r\n' + jpeg + b'\r\n')
                send_started = time.monotonic()
                self.wfile.write(payload)
                write_done = time.monotonic()
                self.wfile.flush()
                flushed_at = time.monotonic()
                timing.write(
                    'http_frame', **pending, wire_bytes=len(payload),
                    packet_build_ms=elapsed_ms(build_at, send_started),
                    publish_to_acquire_ms=elapsed_ms(metadata['published_at'], packet_at),
                    publish_to_send_start_ms=elapsed_ms(metadata['published_at'], send_started),
                    socket_write_ms=elapsed_ms(send_started, write_done),
                    socket_flush_ms=elapsed_ms(write_done, flushed_at),
                    http_active_ms=elapsed_ms(packet_at, flushed_at),
                    source_to_publish_ms=metadata['source_to_publish_ms'],
                    source_to_write_done_ms=elapsed_ms(metadata['source_ready_at'], flushed_at))
                sent_frames += 1
                previous = sequence
                pending = None
                send_started = None
                # Pace only this client. Fetch the latest frame after pacing so
                # output never queues old JPEGs while inference continues.
                preview.stop_event.wait(max(0.0, 1.0 / fps - (flushed_at - build_at)))
                wait_started = time.monotonic()
                packet = preview.wait_frame(sequence, include_timing=True)
                packet_at = time.monotonic()
                wait_ms = elapsed_ms(wait_started, packet_at)
                if packet is None and not preview.stop_event.is_set():
                    close_reason = 'frame_wait_timeout'
        except OSError as exc:
            close_reason = type(exc).__name__
            fields = pending or dict(stream=stream_id, client=client)
            timing.write('http_error', **fields, error=str(exc),
                         socket_elapsed_ms=(elapsed_ms(send_started, time.monotonic())
                                            if send_started is not None else None))
        finally:
            timing.write('stream_close', stream=stream_id, client=client,
                         reason=close_reason, sent_frames=sent_frames)

    def log_message(self, fmt, *args):
        LOG.info('%s %s', self.client_address[0], fmt % args)


if __name__ == '__main__':
    raise SystemExit('Start with: python3 src/yolo_test_task --host 0.0.0.0 --port 8080')
