"""WaterSter application entry point and shared-resource lifecycle.

Start: python3 src/main.py (from the project root)
Page:  http://192.168.127.10/
Starts Nginx and a supervised YOLO camera backend.
"""
import argparse
import logging
import os
import shutil
import subprocess
import signal
import threading

from front_cam import FrontCamera
from web_server import CameraServer

LOG = logging.getLogger(__name__)


def start_nginx():
    """Use the installed Nginx configuration and the device's existing sudo policy."""
    nginx = shutil.which('nginx') or '/usr/sbin/nginx'
    systemctl = shutil.which('systemctl') or '/usr/bin/systemctl'
    privilege = [] if os.geteuid() == 0 else ['sudo', '-n']
    try:
        subprocess.run([*privilege, nginx, '-t'], check=True)
        active = subprocess.run([systemctl, 'is-active', '--quiet', 'nginx'],
                                check=False).returncode == 0
        subprocess.run([*privilege, systemctl, 'reload' if active else 'start', 'nginx'],
                       check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError('Could not start Nginx; check its config and sudo permissions') from exc
    LOG.info('Nginx ready: http://192.168.127.10/')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1',
                        help='HTTP backend listen address (default: behind Nginx)')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--device', default='/dev/video0')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--no-hw-decode', action='store_true',
                        help='Unsupported by the YOLO Web service; hardware decode is required')
    parser.add_argument('--no-nginx', action='store_true',
                        help='Start only the backend; use --host 0.0.0.0 for direct access')
    parser.add_argument('--detections-only', action='store_true',
                        help='Publish coordinates without drawing or JPEG preview encoding')
    args = parser.parse_args()
    if min(args.width, args.height, args.fps) <= 0 or not 1 <= args.port <= 65535:
        parser.error('Invalid frame dimensions, frame rate, or port')
    if not args.detections_only and (args.width % 16 or args.height % 8
                                     or max(args.width, args.height) > 8192):
        parser.error('JPU preview requires width aligned to 16, height to 8, dimensions <=8192')
    if args.no_hw_decode:
        parser.error('The YOLO Web service requires hardware decoding; omit --no-hw-decode')
    if not args.no_nginx and (args.port != 8080 or args.host not in ('127.0.0.1', '0.0.0.0')):
        parser.error('Nginx proxies to 127.0.0.1:8080; use --no-nginx for a custom backend address')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    if os.environ.get('WATERSTER_YOLO_WORKER') != '1':
        if not args.no_nginx:
            start_nginx()
        from yolo_test_task.__main__ import supervise
        return supervise()

    # Creating the object does not open the camera. Share this same instance
    # with future task pipelines as well as the HTTP server.
    camera = FrontCamera(args.device, args.width, args.height, args.fps,
                         hardware_decode=not args.no_hw_decode)
    # Bind before starting capture so a port conflict never opens the camera.
    server = CameraServer((args.host, args.port), camera,
                          preview_enabled=not args.detections_only)

    def stop(signum, _frame):
        LOG.info('Stopping on signal %s', signum)
        # shutdown() must run outside the thread executing serve_forever().
        threading.Thread(target=server.shutdown, daemon=True).start()

    try:
        camera.start()
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        LOG.info('YOLO Web service listening on %s:%d', args.host, args.port)
        if args.no_nginx:
            page_host = '192.168.127.10' if args.host == '0.0.0.0' else args.host
            LOG.info('Page: http://%s:%d/', page_host, args.port)
        else:
            LOG.info('Page: http://192.168.127.10/; health: http://192.168.127.10/health')
        server.serve_forever(poll_interval=.25)
    finally:
        try:
            camera.close()
        finally:
            server.server_close()


if __name__ == '__main__':
    raise SystemExit(main())
