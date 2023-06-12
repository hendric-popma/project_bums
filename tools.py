"""here we are collection our tools"""

import cv2
import numpy as np 
import matplotlib.pyplot as plt

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
    font_scale = 3.0
    thickness = 10
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


def seg_orientation_lines(image, color):
    '''
    segmentiert aus einem Imput Bild die Ortientierungslinien
    Output: bw_image with contours of orientation lines
    '''

    def grayscale_values(image, y_position):
        # Linienbild erstellen
        #line_image = create_horizontal_line(image, y_position)

        # x-Achse und Grauwerte initialisieren
        x_values = np.arange(image.shape[1])
        gray_values = []

        # Grauwerte entlang der Linie sammeln
        for x in x_values:
            gray_values.append(image[y_position, x])

        return gray_values, x_values
    

    def find_largest_component (image):
        '''
        find all connected Components, search for the largest
        Output: Image with the largest Component
        '''
        # Finde alle zusammenhängenden Elemente
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

        # Finde das größte zusammenhängende Element
        largest_component_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

        # Erstelle ein Bild, das nur das größte zusammenhängende Element enthält
        largest_component_image = np.zeros_like(image)
        largest_component_image[labels == largest_component_label] = 255

        return largest_component_image
    

    def seg_dilate_lagrestcomp (image, thresh):
        
        t, seg = cv2.threshold(image,thresh,255,cv2.THRESH_BINARY)

        #segmentiert das größe zusammenhängende Objekt
        # Erstelle ein Bild, das nur das größte zusammenhängende Element enthält
        if color == "W":
            iterations_1 = 7
        else:
            if color == "B":
                iterations_1 = 4
            else:
                raise ValueError("Invalid color selection. Valid options are B for Black or W for White")


        img_dilate = cv2.dilate(seg.astype('uint8'), np.ones((3,3)), iterations = iterations_1)
        #plt.imshow(seg)

        img_largest = find_largest_component(img_dilate)
        #plt.imshow(img_largest)

        return img_largest

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.gray()

    

    #Gauss für bessere Ergebnisse bei finde thresh
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)

    #color = "w"   #Farbe der Orientierungslinie wählen: w für weiß und s für schwarz
    #img_b_or_w = white_or_black(img_blur, color)
    if color == "W":
        img_b_or_w = img_blur
        percentage = 0.2
        seg_min = 24000
    else:
        if color == "B":
            img_b_or_w = 255 - img_blur
            percentage = 0.235
            seg_min = 31000
        else:
            raise ValueError("Invalid color selection. Valid options are B for Black or W for White")
        
    
    #Oberes vietrl des Bildes schwärzen
    h = img_b_or_w.shape[0]
    #w = img_b_or_w.shape[1]
    region = 1/5
    top_region = img_b_or_w[0:int(h*region), :]  # Obere 1/4 des Bildes auswählen

    # Schwärze den ausgewählten Bereich
    top_region = np.zeros_like(top_region)

    # Füge den geschwärzten Bereich wieder in das ursprüngliche Bild ein
    #img_b_or_w_new = np.copy(img_b_or_w)
    img_b_or_w[0:int(h*region), :] = top_region


    #thresh_1, thresh_2 = find_thresh(img_b_or_w, 0.13)  # oberen möglichen 13% ab maximalem Grauwert
    #h = img_b_or_w.shape[1]
    all_grayscale_values = []
    
    for y in range(int(h/2), int(h), int(h/8)):
        gray_values, x_values = grayscale_values(img_b_or_w, y)
        all_grayscale_values.extend(gray_values)


    max_gray = max(all_grayscale_values)
    
    sorted_values = sorted(all_grayscale_values)  # Sortiere die Grauwerte aufsteigend
    num_values = len(sorted_values)  # Anzahl der Grauwerte

    # oberen möglichen 13% ab maximum Grauwert
    #percentage = 0.13
    thresh_percent = 1-percentage
    
    index_1 = int(num_values * thresh_percent)  # Index für den 13% Punkt
    threshold_2 = int(max_gray - (max_gray*percentage))

    threshold_1 = sorted_values[index_1]



    img_largest = seg_dilate_lagrestcomp(img_b_or_w, thresh = threshold_2)

    #TODO: Wenn seg Fläche kleiner Grenzwert ( ca. 30.000), dann nochmal mit thresh_1 statt thresh_2
    white_pixels = cv2.countNonZero(img_largest)
    if white_pixels <= seg_min:
        img_largest_correct = seg_dilate_lagrestcomp(img_b_or_w, thresh=threshold_1)
    else:
        img_largest_correct = img_largest


    #Um evtl. falsch verbundene Komponenten zu entfernen nocheinmal erode, find_largest_component, dilate
    #iterations_2 = 9
    if color == "W":
        iterations_2 = 9
    else:
        if color == "B":
            iterations_2 = 5
        else:
            raise ValueError("Invalid color selection. Valid options are B for Black or W for White")

    img_erode = cv2.erode(img_largest_correct.astype('uint8'), np.ones((3,3)), iterations = iterations_2)
    #plt.imshow(img_erode)

    img_largest_2 = find_largest_component(img_erode)
    #plt.imshow(img_largest_2)

    img_out = cv2.dilate(img_largest_2.astype('uint8'), np.ones((3,3)), iterations = int(iterations_2*1.5))

    return img_out