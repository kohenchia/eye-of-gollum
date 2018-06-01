"""Enables graceful killing by intercepting signals.

Code adopted from:
https://stackoverflow.com/questions/18499497/how-to-process-sigterm-signal-gracefully

"""
import signal
import time


class GracefulKiller:
    """
    Class that captures process signals.
    """
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self,signum, frame):
        """
        Signal handler function
        """
        self.kill_now = True