import numpy as np
import cv2
import matplotlib.pyplot as plt 
import pyttsx3


#%%
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

#%%
# Plot erstellen
def plot_values (gray_values, x_values):
    plt.plot(x_values, gray_values)
    plt.title('Grauwerte entlang der Suchlinie')
    plt.xlabel('x-Achse')
    plt.ylabel('Grauwert')
    plt.show()

#%%
def white_or_black (image, color):
     
    if color == "W":
        img_out = image
    else:
        if color == "B":
            img_out = 255 - image
        else:
            raise ValueError("Invalid color selection. Valid options are B for Black or W for White")

    return img_out

#%%
def find_thresh (image, persentage):

    h = image.shape[0]
    all_grayscale_values = []
    #x_values = []
    
    for y in range(int(h/2), int(h), int(h/8)):
        gray_values, x_values = grayscale_values(image, y)
        all_grayscale_values.extend(gray_values)
        
    
    print(all_grayscale_values)
    max_gray = max(all_grayscale_values)
    
    sorted_values = sorted(all_grayscale_values)  # Sortiere die Grauwerte aufsteigend
    num_values = len(sorted_values)  # Anzahl der Grauwerte

    thresh_percent = 1-persentage
    
    index_1 = int(num_values * thresh_percent)  # Index für den 10% Punkt
    threshold_2 = int(max_gray - (max_gray*persentage))

    print(index_1)
   
    threshold_1 = sorted_values[index_1]

    return threshold_1, threshold_2

#%%
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

#%%
def seg_largest_component(image):
    '''
    finds the orientation lines and segment them
    Outout: Image withe segmentated lines
    '''
    img_dilate = cv2.dilate(image.astype('uint8'), np.ones((3,3)), iterations=7)
    #plt.imshow(img_dilate)

    img_largest = find_largest_component(img_dilate)
    #plt.imshow(img_largest)

    #Um evtl. falsch verbundene Komponenten zu entfernen nocheinmal erode, find_largest_component, dilate
    iterations_2 = 8

    img_erode = cv2.erode(img_largest.astype('uint8'), np.ones((3,3)), iterations=iterations_2)
    #plt.imshow(img_erode)

    img_largest_2 = find_largest_component(img_erode)
    #plt.imshow(img_largest_2)

    img_out = cv2.dilate(img_largest_2.astype('uint8'), np.ones((3,3)), iterations= iterations_2*2)

    return img_out

#%%
def seg_orientation_lines_first_version(image, color, percentage_white = 0.2, percentage_black = 0.235, region = 1/5):
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
    

    def seg_dilate_lagrestcomp (image, thresh, iterations):
        
        t, seg = cv2.threshold(image,thresh,255,cv2.THRESH_BINARY)

        img_dilate = cv2.dilate(seg.astype('uint8'), np.ones((3,3)), iterations = iterations)

        img_largest = find_largest_component(img_dilate)

        return img_largest


    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.gray()

    #Gauss für bessere Ergebnisse bei finde thresh
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)

    #Farbe der Orientierungslinie wählen: w für weiß und s für schwarz
    if color == "W":
        img_b_or_w = img_blur
        percentage = percentage_white
        seg_min = 24000
        iterations_1 = 7
        iterations_2 = 9
    else:
        if color == "B":
            img_b_or_w = 255 - img_blur
            percentage = percentage_black
            seg_min = 31000
            iterations_1 = 4
            iterations_2 = 5
        else:
            raise ValueError("Invalid color selection. Valid options are B for Black or W for White")
        
    
    #Oberes vietrl des Bildes schwärzen
    h = img_b_or_w.shape[0]
    #w = img_b_or_w.shape[1]
    print(h)

    top_region = img_b_or_w[0:int(h*region), :]  # Obere 1/5 des Bildes auswählen

    # Schwärze den ausgewählten Bereich
    top_region = np.zeros_like(top_region)

    # Füge den geschwärzten Bereich wieder in das ursprüngliche Bild ein
    img_b_or_w[0:int(h*region), :] = top_region

    all_grayscale_values = []
    
    for y in range(int(h/2), int(h), int(h/8)):
        gray_values, x_values = grayscale_values(img_b_or_w, y)
        all_grayscale_values.extend(gray_values)

    print(all_grayscale_values)

    #threshold 2, which brings better results in most of the time
    max_gray = max(all_grayscale_values)
    threshold_2 = int(max_gray - (max_gray*percentage))

    #threshold 1, which is better when threshold 2 comes on his limits
    sorted_values = sorted(all_grayscale_values)  # Sortiere die Grauwerte aufsteigend
    num_values = len(sorted_values)  # Anzahl der Grauwerte

    # oberen möglichen percentage ab maximum Grauwert
    index_1 = int(num_values * (1-percentage))  # Index für augewählter percentage
    threshold_1 = sorted_values[index_1]

    print("1:", threshold_1)
    print("2:", threshold_2)

    #fist steps of dilate, find_largest_component, erod, ... 
    #after it is executed checks if result is sufficient, otherwise same procedure with thresh1
    img_largest = seg_dilate_lagrestcomp(img_b_or_w, thresh = threshold_2, iterations = iterations_1)

    #Wenn seg Fläche kleiner Grenzwert, dann nochmal mit thresh_1 statt thresh_2
    white_pixels = cv2.countNonZero(img_largest)
    if white_pixels <= seg_min:
        img_largest_correct = seg_dilate_lagrestcomp(img_b_or_w, thresh = threshold_1, iterations = iterations_1)
    else:
        img_largest_correct = img_largest

    #Um evtl. falsch verbundene Komponenten zu entfernen nocheinmal erode, find_largest_component, dilate
    img_erode = cv2.erode(img_largest_correct.astype('uint8'), np.ones((3,3)), iterations = iterations_2)

    img_largest_2 = find_largest_component(img_erode)

    img_out = cv2.dilate(img_largest_2.astype('uint8'), np.ones((3,3)), iterations = int(iterations_2*1.5))

    return img_out, region

#%%
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

#%%
def draw_seg_orientationline(original_image, seg_image, color=(255, 255, 0), alpha=0.25):
    '''
        Draws the segmented areas into the original picture as a yellow transparent area.

        Args:
            original_image: The original input image.
            seg_image: The segmented image.
            color: The color of the overlay area (default: yellow).
            alpha: The transparency of the overlay area (default: 0.25).

        Output: an image with the segmented areas drawn as a transparent overlay.
    '''
    # Create a copy of the original image
    overlay = original_image.copy()

    # Convert the segmented image to a 3-channel RGB image
    segmented_image_rgb = cv2.cvtColor(seg_image, cv2.COLOR_GRAY2RGB)

    # Set the color of the area in the overlay image to the specified color (default: yellow)
    overlay[np.where(seg_image)] = color

    # Add the overlay image to the original image with transparency
    img_out = cv2.addWeighted(original_image, 1 - alpha, overlay, alpha, 0)

    return img_out


#%%
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


#%%
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

    #%%
    def nearest_line(koords, image_width):
        '''
            Looks for the section in which the koords are located,
            allowing us to determine if we are already on the orientation line
            or if the nearest line is to the left or right of us.

            Args: 
                koords: Koordinates of the segmentated center of the lower section
                image_width: Width of the original used image or frame

            Output: "On line" , "nearest line is left" or "nearest line is  rigth""first - Kopie.ipynb"
        '''
        x_value = koords[0]
        sections = image_width*(1/8)

        if 2*sections < x_value < 6*sections:
            location = "on line"

        else:
            if x_value <= 2*sections:
                location = "nearset line is left"

            if x_value >= 6*sections:
                location = "nearest line is right"

        return location
