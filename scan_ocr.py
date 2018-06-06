# import the necessary packages
import os
import uuid
import numpy as np
import argparse
import cv2
import imutils
import config
from utils import helper_showwait, is_digits, is_letters, validate, os_type
from database import get_assets, update, failure

from transform import four_point_transform
from skimage.filters import threshold_local

from PIL import Image
import pytesseract

if os_type() == 'nt':
    pytesseract.pytesseract.tesseract_cmd = config.tesseract_command_line

debug = config.debug


def capture_frame(image):
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

    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 35, offset=10, method="gaussian")
    warped = (warped > T).astype("uint8") * 255

    # show the original and scanned images
    warped = imutils.resize(warped.copy(), height=500)
    helper_showwait("warped", warped)
    #
    clean_img, chars = extract_characters(warped)
    #
    if len(clean_img) > 0:
        clone_imgs = clone(clean_img)
        for clone_img in clone_imgs:
            #print('Text from OCR - %s'%ocr_text(img))
            string = validate(ocr_text(clone_img))
            if string:
                break
        return string


def clone(clean):
    results = []
    blur_points = [5, 7, 9, 11]
    for iteration in range(3, 11):
        for p in blur_points:
            clean = cv2.GaussianBlur(clean, (p, p), 0)
            kernel = np.ones((2, 2), np.uint8)
            erode = cv2.erode(clean, kernel, iterations=iteration)
            results.append(erode)
    return results


def highlight_characters(img, chars):
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for bbox, char_img in chars:
        x, y, w, h = bbox
        # cv2.rectangle(output_img,(x,y),(x+w,y+h),255,1)
        output_img[y:y + h, x:x + w]
    helper_showwait('Characters Highlighted', output_img)
    return output_img


def extract_characters(img):
    bw_image = cv2.bitwise_not(img)
    contours = cv2.findContours(
        bw_image,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE)[1]

    char_mask = np.zeros_like(img)
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        center = (x + w / 2, y + h / 2)
        # if (area > 700) and (area < 1200):
        
        if (w > 40 and w < 90) and (h > 50 and h < 100) and (y > 300):
            x, y, w, h = x - 4, y - 4, w + 8, h + 8
            bounding_boxes.append((center, (x, y, w, h)))
            cv2.rectangle(char_mask, (x, y), (x + w, y + h), 255, -1)

    helper_showwait('Region of interest', char_mask)
    clean = cv2.bitwise_not(
        cv2.bitwise_and(
            char_mask,
            char_mask,
            mask=bw_image))
    bounding_boxes = sorted(bounding_boxes, key=lambda item: item[0][0])

    characters = []
    for center, bbox in bounding_boxes:
        x, y, w, h = bbox
        char_image = clean[y:y + h, x:x + w]
        characters.append((bbox, char_image))

    clean = cv2.threshold(
        clean,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    return clean, characters


def ocr_text(image):
    filename = "{}.jpg".format(str(uuid.uuid1()))
    cv2.imwrite(filename, image)
    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    return text

"""
def capture_video(video):
    camera = cv2.VideoCapture(r'%s' % (video))
    # keep looping over the frames

    camera.set(1, 2)
    # grab the current frame
    (grabbed, frame) = camera.read()

    try:
        packet = capture_frame(frame)
    except BaseException:
        packet = None

    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()
    return packet
"""


def capture_video(video):
    camera = cv2.VideoCapture(r'%s' % (video))
    # keep looping over the frames

    #camera.set(1, 2)
    count = 0
    while True:
        count = count + 1
        if count > config.maximum_frames:
            break
        # grab the current frame
        (grabbed, frame) = camera.read()

        # check to see if we have reached the end of the
        # video
        if not grabbed:
            break

        try:
            packet = capture_frame(frame)
        except:
            packet = None
        if packet:
            break

    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()
    return packet


import datetime
def process(path=None):
    if path:
        print('Processing from filesystem ...')
        print('Starts at : %s'%str(datetime.datetime.now()))
        for name in os.listdir(path):
            video = os.path.join(path, name)
            if video.endswith('.pdf'):
                continue
            if (not video.endswith('.md')):
                print('Processing video file - %s'%(video))
                string = capture_video(video)
                if string:
                    print(string)
                else:
                    print('Failed 1')
            else:
                print('Failed 2')
        print('Ends at : %s'%str(datetime.datetime.now()))
        return string
    else:
        print('Processing from database ....')
        print('Starts at : %s'%str(datetime.datetime.now()))
        assets = get_assets()
        if assets:
            for asset in get_assets():
                video = os.path.join(asset.path, asset.name)
                if video.endswith('.pdf'):
                    continue                
                if (not video.endswith('.md')):
                    print('Processing video file - %s'%(video))
                    string = capture_video(video)
                    if string:
                        update(asset.id, string)
                        print(string)
                        return string
                    else:
                        print('Failed 1')
                        failure(asset.id)
                else:
                    print('Failed 2')
                    failure(asset.id)
        print('Ends at : %s'%str(datetime.datetime.now()))

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
