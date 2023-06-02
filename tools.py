"""here we are collection our tools"""

import cv2
import numpy as np 

def read_video_frames(video_path):
    """function to read in video frame by frame and save it in a list with an index"""
    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Check if the video capture is successfully opened
    if not cap.isOpened():
        print("Error opening video file.")
        return

    frame_list = []  # List to store the frames
    index = 0

    while cap.isOpened():
        # Read the next frame
        ret, frame = cap.read()

        # If the frame is not successfully read, the video has ended
        if not ret:
            break

        frame_list.append((index, frame))  # Append the frame along with its index
        index += 1

    # Release the VideoCapture object
    cap.release()

    return frame_list


#TODO CONVERT FRAMES TO GRAYSCALE 


def calc_line_koords(x1,y1,x2,y2, limits: list): 
    '''
        function to calaculate line between two points
        returns the koordinates to plot the line 
    '''
    m = (y2-y1)/(x2-x1)
    b = y1 - m * x1
    gerade = list(map(lambda y: int((y-b)/m), limits))
    koords = [[gerade[0], limits[0]], [gerade[1], limits[1]]]
    return koords

def find_center_plot(img):
    '''
    finds the center of a image with white contour on black 
    returns the new image and koords of the center 
    '''
    m = cv2.moments(img)
    cX = int(m["m10"] / m["m00"])
    cY = int(m["m01"] / m["m00"])
    center_koords =  [cX,cY]
    img_res = cv2.circle(img, (cX, cY),100, [0,0,255], cv2.FILLED)
    return img_res, center_koords

def find_max_contour(img):
    """
        function to find the biggest contour (area)
        return new binary image with filled contour 
    """
    cnts, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(np.shape(img), dtype='uint8')
    area = []
    for cnt in cnts:
        area.append(cv2.contourArea(cnt))
    c_idx = area.index(max(area))
    ret_img = cv2.drawContours(out, cnts, c_idx, [255,255,255], cv2.FILLED)
    return ret_img


def thresh_gauss(img):
    """
    makes gauss and auto threshold
    """
    #convert to grayscale 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gauss 
    gauss = cv2.GaussianBlur(img, None, 10)
    t, seg = cv2.threshold(gauss,200,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #streifen = 200
    return seg
