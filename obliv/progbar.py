#!/usr/bin/env python3

"""A progress bar for Python that looks nice with unicode characters."""

import math
import subprocess
import sys

class ProgressBar:
    """A Progress Bar instance, which handles counting up to an arbitrary value.

    This class is a context manager, so you can use it in a with statement.
    This makes initialization and "cleanup" of the display seamless.
    
    On initialization, you set the maximum value of the counter (default 100).
    The += operator is overloaded for convenient updating.

    Note, your program should not do any other i/o to the screen during the progress
    bar output.

    Example usage:

        with ProgressBar(5050) as pb:
            for i in range(100, 0, -1):
                time.sleep(.1)
                pb += i
    """

    def __init__(self, maxval=100, basic=False, dest=sys.stderr):
        """Create a new progress bar.

        The maxval is how much to count up to.
        basic is a flag that, if true, won't use unicode but numeric percentages.
        dest is the file-like object where the progress bar should be printed.
        """

        self.maxval = maxval
        self.dest = dest
        self.val = 0
        self.redraw_at = 0
        self.partial = [' ']
        for code in range(0x258f, 0x2587, -1):
            self.partial.append(chr(code))
        self.gran = len(self.partial) - 1
        self.active = False
        self.basic = basic

    def start(self):
        if self.active:
            raise RuntimeError("Progress bar already started")
        print(file=self.dest)
        self.active = True
        self.redraw()

    def finish(self):
        if not self.active:
            raise RuntimeError("Progress bar not started; can't be stopped.")
        self.redraw()
        self.active = False
        print('\n', file=self.dest)

    def update(self, newval):
        if not self.val <= newval <= self.maxval:
            raise ValueError("invalid progbar value: {} not between {} and {}"
                    .format(newval, self.val, self.maxval))
        self.val = newval
        if self.active and self.val >= self.redraw_at:
            self.redraw()

    def __iadd__(self, increment):
        """Overloads the += operator."""
        self.update(self.val + increment)
        return self

    def __int__(self):
        """Overloads int() conversion."""
        return self.val

    def __enter__(self):
        """Called at the beginning of a with statement."""
        self.start()
        return self

    def __exit__(self, typ, val, cb):
        """Called at the end of a with statement"""
        self.finish()
        return (typ is None)

    def redraw(self):
        if self.basic:
            percent = self.val * 100 // self.maxval
            print('\rProgress: {}%'.format(percent), end="", file=self.dest)
            self.redraw_at = max(self.val+1, ((percent+1)*self.maxval + 99) // 100)
        else:
            assert self.active
            total_width = int(subprocess.check_output(['stty', 'size']).split()[1])
            total_blocks = total_width - 6
            total_subblocks = total_blocks * self.gran
            assert 0 <= self.val <= self.maxval
            percent = self.val * 100 // self.maxval
            subblocks = self.val * total_subblocks // self.maxval
            nfull, remain = divmod(subblocks, self.gran)
            line = "\r\u2592"
            line += self.partial[-1] * nfull
            if remain:
                line += self.partial[remain]
                line += self.partial[0] * (total_blocks - nfull - 1)
            else:
                line += self.partial[0] * (total_blocks - nfull)
            line += "\u2592{:>3}%".format(percent)
            print(line, end="", file=self.dest)
            self.redraw_at = min(
                ((percent+1)*self.maxval + 99) // 100,
                ((subblocks+1)*self.maxval + total_subblocks-1) // total_subblocks
                )
            assert self.redraw_at > self.val
