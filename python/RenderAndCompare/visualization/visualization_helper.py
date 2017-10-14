"""
Visualization Helper
"""

import cv2
import numpy as np


def draw_bbx2d(image, bbx, color=(0, 255, 0), thickness=1):
    """Draw a 2d bbx on image"""
    cv2.rectangle(image,
                  tuple(np.floor(bbx[:2]).astype(int)),
                  tuple(np.ceil(bbx[2:]).astype(int)),
                  color, thickness, cv2.LINE_AA)


class WaitKeyNavigator(object):
    """Convenience class to help navigate opencv windows"""

    def __init__(self, low, high=None, fwd=True, paused=True):
        """
        Args:
            low (int): lowest (signed) integer of the range (unless high=None, in which case 
                        this parameter is one above the highest such integer).
            high (high): If provided, one above the largest (signed) integer of the range 
                        (see above for behavior if high=None).
        """
        if high:
            self.index_range = [low, high - 1]
        else:
            self.index_range = [0, low - 1]
        assert self.index_range[1] >= self.index_range[0], "Bad range {}".format(self.index_range)
        self.index = self.index_range[0]
        self.fwd = fwd
        self.paused = paused

    def print_key_map(self):
        """prints a helpful key map"""
        print "---------------KeyMap-----------------"
        print "Press p to toggle pause"
        print "Press a/s/left/down to move to previous frame"
        print "Press a/s/left/down to move to previous frame"
        print "Press w/d/right/up to move to next frame"
        print "Press ESC or q to quit"

    def process_key(self):
        """Process key map"""
        key = cv2.waitKey(not self.paused)
        if key == 27 or key == ord('q'):
            cv2.destroyAllWindows()
            return True
        elif key in [82, 83, 100, 119, 61, 43]:
            self.fwd = True
        elif key in [81, 84, 97, 115, 45]:
            self.fwd = False
        elif key == ord('p'):
            self.paused = not self.paused
        self.index = self.index + 1 if self.fwd else self.index - 1
        self.index = max(self.index_range[0], min(self.index, self.index_range[1]))
        return False
