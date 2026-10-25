"""WaterSter application entry point and shared-resource lifecycle.

Start: python3 src/main.py
Page:  http://192.168.127.10/
"""
import argparse
import logging
import signal
import threading

from front_cam import FrontCamera
from web_server import CameraServer

LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1',
                        help='Internal HTTP backend address used by Nginx')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--device', default='/dev/video0')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--no-hw-decode', action='store_true',
                        help='Preview only: forward camera JPEG without producing NV12')
    args = parser.parse_args()
    if min(args.width, args.height, args.fps) <= 0 or not 1 <= args.port <= 65535:
        parser.error('Invalid frame dimensions, frame rate, or port')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # Creating the object does not open the camera. Share this same instance
    # with future task pipelines as well as the HTTP server.
    camera = FrontCamera(args.device, args.width, args.height, args.fps,
                         hardware_decode=not args.no_hw_decode)
    # Bind before starting capture so a port conflict never opens the camera.
    server = CameraServer((args.host, args.port), camera)

    def stop(signum, _frame):
        LOG.info('Stopping on signal %s', signum)
        # shutdown() must run outside the thread executing serve_forever().
        threading.Thread(target=server.shutdown, daemon=True).start()

    try:
        camera.start()
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        LOG.info('MJPEG backend listening on %s:%d; open the Nginx page on port 80',
                 args.host, args.port)
        server.serve_forever(poll_interval=.25)
    finally:
        try:
            camera.close()
        finally:
            server.server_close()


if __name__ == '__main__':
    main()
