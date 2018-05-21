# import the necessary packages
import os
import uuid
import numpy as np
import argparse
import cv2
import imutils

#from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local

from PIL import Image
import pytesseract
#pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'

path = r'd:\PROJECTS\scanner\videos'
debug = False

def helper_imshow(name, image):
    cv2.imshow(name, image)

def helper_imwait():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def helper_showwait(name, image):
    if debug:
        helper_imshow(name, image)
        helper_imwait()

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
	return warped

def detect(image):
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height = 500)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # show the original image and the edge detected image
    print "STEP 1: Edge Detection"
    helper_showwait("Image", image)
    helper_showwait("Edged", edged)

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

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
    print("STEP 2: Find contours of paper")
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    helper_showwait("Outline", image)

    # apply the four point transform to obtain a top-down
    # view of the original image
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 35, offset = 10, method = "gaussian")
    warped = (warped > T).astype("uint8") * 255

    # show the original and scanned images
    print("STEP 3: Apply perspective transform")
    warped = imutils.resize(warped.copy(), width=500, height = 500)
    helper_showwait("warped", warped)
    clean_img, chars = extract_characters(warped)
    output_img = highlight_characters(clean_img, chars)
    return clean_img

def highlight_characters(img, chars):
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for bbox, char_img in chars:
        x,y,w,h = bbox
        #cv2.rectangle(output_img,(x,y),(x+w,y+h),255,1)
        output_img[y:y+h,x:x+w]
    helper_showwait('Characters Highlighted', output_img)
    return output_img

def extract_characters(img):
    bw_image = cv2.bitwise_not(img)
    contours = cv2.findContours(bw_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

    char_mask = np.zeros_like(img)
    bounding_boxes = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        area = w * h
        center = (x + w/2, y + h/2)
        #if (area > 700) and (area < 1200):
        if (w > 20 and w < 40) and (h > 30 and h < 40):
            x,y,w,h = x-4, y-4, w+8, h+8
            bounding_boxes.append((center, (x,y,w,h)))
            cv2.rectangle(char_mask,(x,y),(x+w,y+h),255,-1)

    helper_showwait('Roi', char_mask)
    clean = cv2.bitwise_not(cv2.bitwise_and(char_mask, char_mask, mask = bw_image))
    bounding_boxes = sorted(bounding_boxes, key=lambda item: item[0][0])  

    characters = []
    for center, bbox in bounding_boxes:
        x,y,w,h = bbox
        char_image = clean[y:y+h,x:x+w]
        characters.append((bbox, char_image))

    clean = cv2.threshold(clean, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    clean = cv2.GaussianBlur(clean,(5,5),0)
    #clean= cv2.Canny(clean, 75, 200)
    helper_showwait('Clean Image', clean)
    return clean, characters

def image_to_text(image):
    filename = "{}.jpg".format(str(uuid.uuid1()))
    cv2.imwrite(filename, image)
    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    return text

def feeder(video):
    text = None
    camera = cv2.VideoCapture(r'%s'%(video))
    # keep looping over the frames
    while True:
        # grab the current frame
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width=800)

        # check to see if we have reached the end of the
        # video
        if not grabbed:
            break

        clean_img = detect(frame)
        if len(clean_img) > 0:
            clean_img = imutils.resize(clean_img, width=400, height = 1200)
            text = image_to_text(clean_img)
            if text:
                break

    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()
    return text

def process():
    for v in os.listdir(path):
        new_path = os.path.join(path, v)
        text = feeder(new_path)
        print text


def process_image():
    images = os.listdir(path)
    for image in images:
        new_path = os.path.join(path, image)
        image = detect(cv2.imread(r'%s'%(new_path)))
        img = imutils.resize(image, width=300, height = 1200)
        helper_showwait('last image', img)
        print(image_to_text(img))

#SqlLite part


if __name__ == '__main__':
    process()
