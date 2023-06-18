"""here we are collection our tools"""

import cv2
import numpy as np 
import matplotlib.pyplot as plt
import pyttsx3
import sys

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

def user_input():
    """
    reads in the video file
    user gives video file over path and decide if the line is black or white

    return:
        video   cv2.VideoCapture Object
        line_color  string W or B 
    """
    while True:
        try:
            path = input("please enter the path to the video, exit with q: ")
            if path == "q":
                break
            video = cv2.VideoCapture(path)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) # for automatic sopping
            if not video.isOpened():
                print("wrong path please try again")
            else:
                print("")
                break
        except:
            print("")
    if path == "q":
        sys.exit()

    while True:
        line_color = input("Please enter the color of the line. Choose between black(b) or white(w): ").upper()
        
        if line_color == "W" or line_color == "B":
            print("You did a great Job!")
            break  # Exit the loop if the input is correct
        else:
            print("Are you serious?")
    return video, line_color, total_frames

class Frame:
    def __init__(self, frame):
        self.height = frame.shape[0]
        self.width = frame.shape[1]
        self.show = frame.copy()
        

#TODO CONVERT FRAMES TO GRAYSCALE 


def calc_line_koords(x1,y1,x2,y2, limits: list): 
    '''
        function to calaculate line between two points
        returns the koordinates to plot the line 
    '''
    #to get NO ZeroDivisionError
    if x2-x1 == 0: 
        resx = 1
    else: 
        resx = x2-x1
    m = (y2-y1)/(resx)
    b = y1 - m * x1
    gerade = list(map(lambda y: int((y-b)/m), limits))
    koords = [[gerade[0], limits[0]], [gerade[1], limits[1]]]
    return koords

def find_center_plot(img):
    '''
    finds the center of a image with white contour on black 
    returns the new image and koords(x/y) of the center 
    '''
    m = cv2.moments(img)
    try:
        cX = int(m["m10"] / m["m00"])
        cY = int(m["m01"] / m["m00"])
    except ZeroDivisionError:
        print("zero division error")
        return img, [int(val/2) for val in list(img.shape[:2])] # use koords of center from image 
    center_koords =  [cX,cY]
    img_res = cv2.circle(img, (cX, cY),10, [0,0,255], cv2.FILLED)
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

def put_text_image(img, text:str):
    """
    puts the text in the middel of an image
    returns the new image 
    """
    # Define the text and its properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 4
    color = (255, 0, 255)  # Green color in BGR format

    # Get the text size
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    return cv2.putText(img, text, (int(img.shape[1]/2-text_width/2), int(img.shape[0]/2-text_height/2)), font, font_scale, color, thickness)

def draw_seg_orientationline(original_image, seg_image, color=(0, 255, 255), alpha=0.25):
    '''
        Draws the segmented areas into the original picture as a yellow transparent area.
     TODO better description
        Input: original_image, segmented_image, Color (default: yellow), Transparency (default: 0.25)
        Output: Picture with blue transparent orientation lines
    '''
    # Create a copy of the original image
    overlay = original_image.copy()

    # Set the color of the area in the overlay image to the specified color (default: yellow)
    overlay[np.where(seg_image)] = color

    # Add the overlay image to the original image with transparency
    img_out = cv2.addWeighted(original_image, 1-alpha, overlay, alpha, 0)

    return img_out

def audio_output(value, standard1_str_= str("Eine Kreuzung wurde dedektiert!"), standard2_str_= str("Sie haben folgende Abbiegemöglichkeiten:")):
    '''
        Outputs a given Value(Numbers and Letters possible) as audio
    '''
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # Convert the new value to a readable string
    value_str = str(value)

    # Configure voice properties
    engine.setProperty('rate', 150)  # Speed of the speech output
    engine.setProperty('volume', 0.9)  # Volume of the speech output

    # Output the standartised first words
    engine.say(standard1_str_)
    engine.say(standard2_str_)
    # Output the new value as audio
    engine.say(value_str)
    engine.runAndWait()


def seg_orientation_lines(image, color, percentage_white=0.2, percentage_black=0.235, region=1/5):
    '''
    Segments orientation lines from an input image.
    Output: bw_image with contours of orientation lines
    '''

    def grayscale_values(image, y_position):
        '''
            returns a list with all gray_values found along a horizontal line

            Args:
                image: Imput image (grayscale)
                y_position: y_koordinat of the horizontal line
        '''
        # Initialize x-axis and grayscale values
        x_values = np.arange(image.shape[1])
        gray_values = []

        # Collect grayscale values along the line
        for x in x_values:
            gray_values.append(image[y_position, x])

        return gray_values

    def find_largest_component(image):
        '''
        Find all connected components and search for the largest one.
        Output: Image with the largest component
        '''
        # Find all connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

        # Find the largest connected component
        largest_component_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

        # Create an image that contains only the largest connected component
        largest_component_image = np.zeros_like(image)
        largest_component_image[labels == largest_component_label] = 255

        return largest_component_image


    def seg_dilate_largest_comp(image, thresh, iterations):
        '''
        Applies dilation to the segmented image and returns the largest component.

        Args:
            image: Input image (grayscale).
            thresh: Threshold value for binarization.
            iterations: Number of iterations for dilation.

        Output: Image with the largest component after dilation.
        '''

        # Threshold the image to create a binary segmentation
        t, seg = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)

        # Apply dilation to the segmented image
        img_dilate = cv2.dilate(seg.astype('uint8'), np.ones((3, 3)), iterations=iterations)

        # Find the largest connected component in the dilated image
        img_largest = find_largest_component(img_dilate)

        return img_largest


    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.gray()

    # Apply Gaussian blur for better results when finding the threshold
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Choose color of the orientation lines: "W" for white or "B" for black
    if color == "W":
        img_b_or_w = img_blur
        percentage = percentage_white        #Passes the appropriate value to percentage
        seg_min = 24000                      #Sets limit value (Pixels of largest connected componend) when to choose tresh_1
        iterations_1 = 7                     #Iterations for first round of dilate                  
        iterations_2 = 9                     #Iterations for erode and second dilate
    else:
        if color == "B":
            img_b_or_w = 255 - img_blur      #invert picture for next steps
            percentage = percentage_black    #Passes the appropriate value to percentage
            seg_min = 31000                  #Sets limit value (Pixels of largest connected componend) when to choose tresh_1
            iterations_1 = 4                 #Iterations for first round of dilate
            iterations_2 = 5                 #Iterations for erode and second dilate
        else:
            raise ValueError("Invalid color selection. Valid options are B for Black or W for White")

    # Darken the top region of the image
    h = img_b_or_w.shape[0]
    top_region = img_b_or_w[0:int(h * region), :]  # Select the top region (default 1/5) of the image

    # Darken the selected region
    top_region = np.zeros_like(top_region)

    # Add the darkened region back to the original image
    img_b_or_w[0:int(h * region), :] = top_region

    #Write all gray values along a horizontal line in all_grayscale_values
    all_grayscale_values = []

    for y in range(int(h / 2), int(h), int(h / 8)):              #Search only in the lower half of the picture and do it on 4 to 5 points max. (distance between them h/8)
        gray_values = grayscale_values(img_b_or_w, y)
        all_grayscale_values.extend(gray_values)

    # Threshold 2, which usually provides better results
    max_gray = max(all_grayscale_values)
    threshold_2 = int(max_gray - (max_gray * percentage))

    # Threshold 1, which is used when threshold 2 reaches its limits
    sorted_values = sorted(all_grayscale_values)  # Sort the grayscale values in ascending order
    num_values = len(sorted_values)  # Number of grayscale values

    # Upper possible percentage starting from the maximum grayscale value
    index_1 = int(num_values * (1 - percentage))  # Index for the selected percentage
    threshold_1 = sorted_values[index_1]

    # First steps of dilate, find_largest_component, erode, ...
    # After execution, check if the result is sufficient, otherwise repeat the procedure with threshold 1
    img_largest = seg_dilate_largest_comp(img_b_or_w, thresh=threshold_2, iterations=iterations_1)

    # If the segmented area is smaller than the minimum threshold, perform the process again with threshold 1 instead of threshold 2
    white_pixels = cv2.countNonZero(img_largest)
    if white_pixels <= seg_min:
        img_largest_correct = seg_dilate_largest_comp(img_b_or_w, thresh=threshold_1, iterations=iterations_1)
    else:
        img_largest_correct = img_largest

    # Remove any falsely connected components by applying erode, find_largest_component, dilate again
    img_erode = cv2.erode(img_largest_correct.astype('uint8'), np.ones((3, 3)), iterations=iterations_2)

    img_largest_2 = find_largest_component(img_erode)

    img_out = cv2.dilate(img_largest_2.astype('uint8'), np.ones((3, 3)), iterations=int(iterations_2 * 1.5))


    return img_out