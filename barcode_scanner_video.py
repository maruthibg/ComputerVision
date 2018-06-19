# USAGE
# python barcode_scanner_video.py

# import the necessary packages
from imutils.video import VideoStream
from transform import four_point_transform
from pyzbar import pyzbar
import argparse
import datetime
import imutils
import time
import cv2
from config import debug

# barcodes found thus far
found = set()


def process(vs):
    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it to
        # have a maximum width of 400 pixels
        (grabbed, frame) = vs.read()
        # check to see if we have reached the end of the
        # video
        if not grabbed:
            break
        
        # find the barcodes in the frame and decode each of the barcodes
        barcodes = pyzbar.decode(frame)

        # loop over the detected barcodes
        for barcode in barcodes:
            # extract the bounding box location of the barcode and draw
            # the bounding box surrounding the barcode on the image
            (x, y, w, h) = barcode.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # the barcode data is a bytes object so if we want to draw it
            # on our output image we need to convert it to a string first
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type

            # draw the barcode data and barcode type on the image
            text = "{} ({})".format(barcodeData, barcodeType)
            if debug:
                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow("Barcode Scanner", frame)
                key = cv2.waitKey(1) & 0xFF
    
                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break                
            else:
                return text

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str,
                    help="path to input video containing barcodes")
    args = vars(ap.parse_args())
    # initialize the video stream and allow the camera sensor to warm up
    if debug:
        print("[INFO] starting video stream...")
    src = r'%s' % (args["input"])
    vs = cv2.VideoCapture(src)
    time.sleep(2.0)
    text = process(vs)
    print(text)
    if debug:
        print("[INFO] cleaning up...")
    cv2.destroyAllWindows()
    vs.release()
