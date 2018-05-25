import cv2
import config
import re

debug = config.debug


def helper_imshow(name, image):
    cv2.imshow(name, image)


def helper_imwait():
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def helper_showwait(name, image):
    if debug:
        helper_imshow(name, image)
        helper_imwait()

def is_letters(value):
    return bool(re.match('^[A-Z]+$', value))

def is_digits(value):
    return bool(re.match('^[0-9]+$', value))
