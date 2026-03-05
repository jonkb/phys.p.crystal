""" Utility functions
"""
import time
from datetime import datetime

def isonow():
    """ Return the current date and time in ISO format, suitable
    for filenames: YYYYMMDD-HHMMSS
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def tic():
    """ Start timing. Returns a list of times with one entry.
    I wrote these tic & toc functions for NMLAB
    See https://github.com/jonkb/NMLab/blob/main/src/util.py
    """
    times = []
    times.append(time.time())
    return times

def toc(times, msg=None, total=False):
    """ Log the current time in the times list.
    If msg is provided, print out a message with the time.
      the string f" time: {t}" will be appended to the message.
    If total, then print out the elapsed time since the start.
      Else, print out only the last time interval.
    """
    times.append(time.time())
    if msg is not None:
        t = times[-1] - times[0] if total else times[-1] - times[-2]
        print(f"{msg} time: {t:.6f} s")


