import djitellopy
from collections import deque
from threading import Lock, Thread
from settings import settings as S
import av
import numpy as np

# override entire function
def tello_bgfr_init_override(self, tello, address, with_queue = False, maxsize = 32):
    print('!running monkeypatched BackgroundFrameRead init')
    self.address = address
    self.lock = Lock()
    self.frame = np.zeros([300, 400, 3], dtype=np.uint8)
    self.frames = deque([], maxsize)
    self.with_queue = with_queue

    try:
        djitellopy.Tello.LOGGER.debug('trying to grab video frames...')
        self.container = av.open(self.address, timeout=(30, 30))
    except av.error.ExitError:
        raise djitellopy.TelloException('Failed to grab video frames from video stream')

    self.stopped = False
    self.worker = Thread(target=self.update_frame, args=(), daemon=True)
djitellopy.BackgroundFrameRead.__init__ = tello_bgfr_init_override


# override definitions
_orig_send_command = djitellopy.Tello.send_command_with_return
def _patched_send_command(self, command, timeout=None):
    # force your patched value if None or original default
    if timeout is None or timeout == djitellopy.Tello.RESPONSE_TIMEOUT:
        timeout = S.tello_response_timeout
    return _orig_send_command(self, command, timeout=timeout)
djitellopy.Tello.send_command_with_return = _patched_send_command

_orig_control_command = djitellopy.Tello.send_control_command
def _patched_control_command(self, command, timeout=None):
    if timeout is None or timeout == djitellopy.Tello.RESPONSE_TIMEOUT:
        timeout = S.tello_response_timeout
    return _orig_control_command(self, command, timeout=timeout)
djitellopy.Tello.send_control_command = _patched_control_command

_orig_init = djitellopy.Tello.__init__
def _patched_init_command(self, host=S.tello_ip, retry_count=S.tello_retry_count, vs_udp=S.tello_vs_port):
    return _orig_init(self, host=host, retry_count=retry_count, vs_udp=vs_udp)
djitellopy.Tello.__init__ = _patched_init_command