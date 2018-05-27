# import the necessary packages
import os
import argparse
import numpy as np
import cv2
import imutils

from transform import four_point_transform
import pyzbar.pyzbar as pyzbar

from database import get_assets, update
from utils import helper_showwait, is_digits, is_letters, validate


class Packet():
    pass


def decode(im):
    # Find barcodes and QR codes
    decodedObjects = pyzbar.decode(im)

    p = Packet()
    for obj in decodedObjects:
        p.Type = obj.type
        p.value = obj.data
    return p


def capture_frame(image):
    # load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # show the original image and the edge detected image
    helper_showwait("Image", image)
    helper_showwait("Edged", edged)
    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(
        edged.copy(),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    # show the contour (outline) of the piece of paper
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    helper_showwait("Outline", image)

    # apply the four point transform to obtain a top-down
    # view of the original image
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    # show the original and scanned images
    scanned = imutils.resize(warped, height=200)
    helper_showwait("Scanned", scanned)
    return scanned


def skew(image):
    # convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # the cv2.minAreaRect function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)

    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # draw the correction angle on the image so we can validate it
    cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the output image
    print("[INFO] angle: {:.3f}".format(angle))
    helper_showwait("Input", image)
    helper_showwait("Rotated", rotated)


def bar_code_detect(image):
    # convert the image to grayscale
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = image

    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # blur and threshold the image
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    # find the contours in the thresholded image
    (_, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # if no contours were found, return None
    if len(cnts) == 0:
        return None

    # otherwise, sort the contours by area and compute the rotated
    # bounding box of the largest contour
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))

    # return the bounding box of the barcode
    return box


def capture_video(video):
    text = None
    camera = cv2.VideoCapture(r'%s' % (video))
    # keep looping over the frames
    while True:
        # grab the current frame
        (grabbed, frame) = camera.read()
        # check to see if we have reached the end of the
        # video
        if not grabbed:
            break

        try:
            packet = decode(capture_frame(frame)).value
        except BaseException:
            packet = None

        if packet:
            break

    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()
    return packet


def process(path=None):
    if path:
        print('Processing from filesystem ...')
        for name in os.listdir(path):
            video = os.path.join(path, name)
            string = capture_video(video)
            print(string)
    else:
        print('Processing from database ....')
        for asset in get_assets():
            video = os.path.join(asset.path, asset.name)
            string = capture_video(video)
            update(asset.id, string)
            print(string)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-f",
        "--process_from",
        required=True,
        help="Process from , database or filesystem")
    ap.add_argument(
        "-p",
        "--path",
        required=False,
        help="Source files or folder path")
    args = vars(ap.parse_args())

    process_from = args['process_from']
    path = args['path']
    if process_from == 'database':
        process()
    elif process_from == 'filesystem' and path:
        process(path=args['path'])
    else:
        raise ValueError('Valid option not provided')
