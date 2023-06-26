"""here we are collection our tools"""
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import pyttsx3
import sys
import re

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
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) # for automatic stopping
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

def audio_output(value):
    '''
    Outputs a given Value(Numbers and Letters possible) as audio
    '''
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # Convert the new value to a readable string
    output_str = re.sub(r'\d+', '', str(value))

    # Configure voice properties
    engine.setProperty('rate', 180)  # Speed of the speech output
    engine.setProperty('volume', 0.9)  # Volume of the speech output

    #Check whether string for a turn or not?
    if output_str == "straight" or output_str == "no orientation line" or output_str == "nearest line is left " or output_str == "nearest line is right ":
        engine.say(output_str)
    else:
        final_output_str = "walking options: "+ output_str 
        engine.say(final_output_str)

    engine.runAndWait()

class FrameObject:

    def __init__(self, frame):
        self.img = frame
        self.height = frame.shape[0]
        self.width = frame.shape[1]
        self.show = frame.copy()
        self.only_straight = []
        self.line = 0 # used later
    
    def get_frame_img(self):
        return self.frame
    
    def canny_zero_line(self):
        self.canny = cv2.Canny(self.line, 0, 0)
        return self.canny

    def calc_line_koords(self, x1,y1,x2,y2): 
        '''
            function to calaculate line between two points
            returns the koordinates to plot the line 
        '''
        limits = [0,self.height]
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

    def find_center_plot(self, img):
        '''
        finds the center of a image with white contour on black 
        returns the new image and koords(x/y) of the center 
        '''
        m = cv2.moments(img)
        try:
            cX = int(m["m10"] / m["m00"])
            cY = int(m["m01"] / m["m00"])
        except ZeroDivisionError:
            #print("zero division error")
            # TODO make logging
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

    def text_in_frame(self, text:str):
        """
        puts the text in the middel of an image
        returns the new image 
        """
        # Define the text and its properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2 #bevor 1.5
        thickness = 3
        color = (255, 0, 255)  # Green color in BGR format

        # Get the text size
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        self.show = cv2.putText(self.show, text, (int(self.width/2-text_width/2), int(self.height/2-text_height/2)), font, font_scale, color, thickness)
        return self.show
    
    def overlay_segmentation(self, seg_image, color=(0, 255, 255), alpha=0.25):
        '''
        Draws the segmented areas into the original picture as a yellow transparent area.
    
        Args: original_image, segmented_image, Color (default: yellow), Transparency of the area (default: 0.25)
        Output: Picture with yellow transparent orientation lines
        '''
        # Create a copy of the original image
        overlay = self.show.copy()

        # Set the color of the area in the overlay image to the specified color (default: yellow)
        overlay[np.where(seg_image)] = color

        # Add the overlay image to the original image with transparency
        self.show = cv2.addWeighted(self.show, 1-alpha, overlay, alpha, 0)

        return self.show
    

    def build_center_line(self,img_cont):
        self.koords = []
        cut = int(6/9 * self.height) # bevor 3/5
        black_up = img_cont[:cut,:]
        black_down = img_cont[cut:,:]

        #find center of both halfs
        c_black_up, koords_up = self.find_center_plot(black_up)
        c_black_down, koords_down = self.find_center_plot(black_down)
        self.koords_down = koords_down
        # if koords are center koords take the koords from the "run" before
        
        if len(self.koords) > 0: 
            if koords_up == [int(val/2) for val in list(img_cont.shape[:2])]: #frame 
                koords_up = self.koords[0]
            if koords_down == [int(val/2) for val in list(img_cont.shape[:2])]:
                koords_down = self.koords[1]
        
        
        #concatenate both halfs two full picture again
        c_new = np.concatenate((c_black_up, c_black_down))
        # make line through middle 
        self.koords = self.calc_line_koords(koords_up[0], koords_up[1], koords_down[0], koords_down[1]+cut)
        self.line = cv2.line(c_new, self.koords[0], self.koords[1], [0,0,0], 7)
        self.show = cv2.line(self.show, self.koords[0], self.koords[1], [0,0,0], 7)
        return self.koords
    

    def make_block_over_center_line(self):
        """
        check for zeroes at bottom of frame to find line from canny 
        """
        img_cut = self.line.copy()
        canny = self.canny_zero_line()
        one = canny[self.height-10,:] > 0
        #xposition where img is white at bottom
        on = np.where(one==True)

        # TODO make smarter  
        # make black line over the big white straight line (vertical)
        # if no line at bottom go up until line was found   
        dist_min = self.height*0.1
        # check that are enough space to the edge of frame
        if len(on[0]) >= 2 and on[0][0] > dist_min and on[0][-1] < (self.height-dist_min):
            #make black in line 
            #dist = (on[0][0]-10)-(on[0][-1]+10)
            #img_cut = cv2.line(self.line, self.koords[0], self.koords[1], [0,0,0], abs(dist))
            img_cut[:,on[0][0]-10:on[0][-1]+10] = 0 #bevor 5 ->10

        else: 
            offset_line_canny = int(self.height * 0.2) #for horizontal video offset 0.8
            ycheck = self.height-offset_line_canny
            # put y position up as long there is no white space found
            while len(on[0]) == 0:
                #ycheck cannot be smaller than = otherwise it isnt in the picture range of y-axis
                if ycheck <=0:
                    return
                one = self.canny[ycheck,:] > 0
                on = np.where(one==True)
                ycheck = ycheck - offset_line_canny  
            img_cut[:,on[0][0]-10:on[0][-1]+10] = 0 # bevor 5 -> 10
            #dist = (on[0][0]-10)-(on[0][-1]+10)
            #img_cut = cv2.line(self.line, self.koords[0], self.koords[1], [0,0,0], abs(dist))
            
        return img_cut #cv2.cvtColor(img_cut, cv2.COLOR_BGR2GRAY)
            

class FindDirection(FrameObject):
    def __init__(self):
        self.output = [0]
        self.dist = [0]
        self.only_straight = []
        self.koords_hor = []
        self.res_text = None
        self.wait = None
        self.save_cnts = None
        self.counter = 0
        self.ubers =    {
            0 : "straight",
            1 : "left", 
            2 : "right",
            3: "left and right",
            4 : "no orientation line",
            11 : "only left", 
            22: "only right", 
            33 : "only left and right",
            111 : "nearest line is left",
            222 : "nearest line is right", 
        }

    def get_values_from_frame_object(self,frame:FrameObject):
        #TODO rename
        self.frame = frame
        self.koords = frame.koords

    def add_one_counter(self):
        self.counter = self.counter+1
    
    def find_nearest_line(self, img_cont):
        '''
        Looks for the section in which the koords are located,
        allowing us to determine if we are already on the orientation line
        or if the nearest line is to the left or right of us.

        Args: 
            koords: Koordinates of the segmentated center of the lower section
            seg_image: The segmented image of the orientation line.

        Output: None for "On line" , 111 for "nearest line is left" or 222 for "nearest line is rigth"
        '''
        if self.frame.koords_down[0] is None:
            return
        x_value = self.frame.koords_down[0]
        
        w = self.frame.width
        h = self.frame.height
        white_pixels = 0
        for y in range(h-10,h):
            white_line = np.count_nonzero(img_cont[y] == 255)
            white_pixels = white_pixels + white_line
        if white_pixels <= 210:   # when True possibility is heigh, that the line comes from on sid into the picture
            sections = w*(1/7)

            if 2*sections < x_value < 5*sections:
                # we are on the line
                self.output.append(0)
            elif x_value <= 2*sections:
                #nearset line is left
                self.output.append(111)
            elif x_value >= 5*sections:
                #nearest line is right
                self.output.append(222)
        else: 
            self.output.append(0)
        return
    

    def wait_for_complete_contour(self, img):
        """
        delay the direction-decision until contours are complete in image
        """
        if self.wait is None: 
            self.wait = 0
        if self.save_cnts is None:
            self.save_cnts

        self.cnts,_ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # when cnts come in picture wait bevor findcontours so that contour is in complete picture solved
        if len(self.cnts) == 0 and self.save_cnts == 0: 
            self.wait = 0
        if len(self.cnts) > 0 and self.save_cnts == 0: 
            self.wait = self.wait + 1
        if self.wait >= 1: # count up if bigger then one
            self.wait = self.wait +1
        self.save_cnts = len(self.cnts) #save for next while step
        return self.cnts, self.wait 

    def check_where_to_go(self):

        if self.wait is None:
            print("use fkt waiting before")
            return

        # check if we are on the line or not
        if self.output[-1] == 111 or self.output[-1] == 222:
            return
        else:
            # check if you cannot go straight
            one2 = self.frame.canny[int(1/3 * self.frame.height),:] > 0
            one3 = self.frame.canny[int(0.5 * self.frame.height),:] > 0

            #xposition where img is white/black at top
            on2 = np.where(one2==True)
            on3 = np.where(one3==True)
            if len(on2[0]) < 2 and len(on3[0] < 2): #TODO maybe <=2
                self.only_straight.append(1)  
            else:
                self.only_straight.append(0)

            if self.wait <2:
                # check for zeroes at bottom of frame to find line from canny 
                if self.only_straight[-1] == 1 and self.only_straight[-2] == 1:# and self.only_straight[-3] == 1:
                    self.output.append(4)
                else:
                    self.output.append(0)
                self.dist.append("")
            if self.wait > 2:
                
                cnts_idx = []
                for i in range(0,len(self.cnts)):
                    # Calculate the bounding rectangle of the contour
                    x, y, w, h = cv2.boundingRect(self.cnts[i])
                    if w/h > 0.85: # bevor it was 1
                        self.frame.show = cv2.drawContours(self.frame.show, self.cnts, i, [255,0,0], 2)
                        cnts_idx.append(i)
                

                self.koords_hor = []      
                for i in cnts_idx:
                    _,cent = self.find_center_plot(self.cnts[i])
                    self.frame.show = cv2.circle(self.frame.show,cent,10, [255,0,0], cv2.FILLED)
                    self.koords_hor.append(cent)
                #check if koords_hor ist left or right from the middle or both 
                if len(self.koords_hor) == 1:
                    if self.koords_hor[0][0] < (self.koords[0][0]+self.koords[1][0])/2: #TODO maybe take only the lower koords
                        #left is one
                        self.dist.append(self.frame.height-self.koords_hor[0][1])
                        if self.only_straight[-1] == 1 and self.only_straight[-2] == 1 and self.only_straight[-3] == 1:
                            self.output.append(11)
                        else:
                            self.output.append(1)
                        
                    if self.koords_hor[0][0] > (self.koords[0][0]+self.koords[1][0])/2:
                        #rigth is two
                        self.dist.append(self.frame.height-self.koords_hor[0][1])
                        if self.only_straight[-1] == 1 and self.only_straight[-2] == 1 and self.only_straight[-3] == 1:
                            self.output.append(22)
                        else:
                            self.output.append(2)
                elif len(self.koords_hor) == 2:
        #TODO check position of the cnts 
                    if self.koords_hor[0][0] < (self.koords[0][0]+self.koords[1][0])/2 and self.koords_hor[1][0] > (self.koords[0][0]+self.koords[1][0])/2:
                        #left and right is 3
                        self.dist.append(self.frame.height-self.koords_hor[0][1])
                        if self.only_straight[-1] == 1 and self.only_straight[-2] == 1 and self.only_straight[-3] == 1:
                            self.output.append(33)
                        else:
                            self.output.append(3)
                    elif self.koords_hor[1][0] < (self.koords[0][0]+self.koords[1][0])/2 and self.koords_hor[0][0] > (self.koords[0][0]+self.koords[1][0])/2:
                        #left and right is 3
                        self.output.append(3)
                        self.dist.append(self.frame.height-self.koords_hor[0][1])
                        if self.only_straight[-1] == 1 and self.only_straight[-2] == 1 and self.only_straight[-3] == 1:
                            self.output.append(33)
                        else:
                            self.output.append(3)
            return self.output, self.dist
    
    
    def smooth_output(self):

        if self.res_text is None:
            self.res_text = ""

        if self.counter%5 == 0:
            #print(f"modulo {self.counter}")

            if len(self.output) > 3:
                if self.output[-1] == self.output[-2]:
                    self.res_text = self.ubers[self.output[-1]]
                elif self.output[-1] != self.output[-2]: 
                    self.res_text = self.ubers[self.output[-3]]
                elif self.output[-1] == self.output[-3]:
                    self.res_text = self.ubers[self.output[-1]]
        #when straight then no distance output
        #self.res_text = self.ubers[self.output[-1]]
        self.add_one_counter() 
        if self.res_text == "straight" or self.res_text == "no orientation line" or self.res_text == "nearset line is left" or self.res_text=="nearset line is right":
            return self.res_text
        else:
            return self.res_text+ " " + str(self.dist[-1])

    
    def get_ubers(self):
        return self.ubers


class Segmentation:
    
    def __init__(self, frame:FrameObject, line_color):
        self.img = frame.img
        self.frame = frame
        self.line_color = line_color
        
    def grayscale_values(self,img, y_position):
        '''
        returns a list with all gray_values found along a horizontal line

        Args:
            image: Imput image (grayscale)
            y_position: y_koordinat of the horizontal line
        '''
        # Initialize x-axis and grayscale values
        x_values = np.arange(img.shape[1])
        gray_values = []

        # Collect grayscale values along the line
        for x in x_values:
            gray_values.append(img[y_position, x])

        return gray_values

    def find_largest_component(self, img):
        '''
        Find all connected components and search for the largest one.
        Output: Image with the largest component
        '''
        # Find all connected components
        _ , labels, stats, _ = cv2.connectedComponentsWithStats(img)

        # Find the largest connected component
        largest_component_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

        # Create an image that contains only the largest connected component
        largest_component_image = np.zeros_like(img)
        largest_component_image[labels == largest_component_label] = 255

        return largest_component_image

    def seg_dilate_largest_comp(self,img, thresh, iterations):
        '''
        Applies dilation to the segmented image and returns the largest component.

        Args:
            image: Input image (grayscale).
            thresh: Threshold value for binarization.
            iterations: Number of iterations for dilation.

        Output: Image with the largest component after dilation.
        '''

        # Threshold the image to create a binary segmentation
        t, seg = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

        # Apply dilation to the segmented image
        img_dilate = cv2.dilate(seg.astype('uint8'), np.ones((3, 3)), iterations=iterations)

        # Find the largest connected component in the dilated image
        img_largest = self.find_largest_component(img_dilate)

        return img_largest

    def get_orientation_lines(self, percentage_white=0.2, percentage_black=0.235, region=1/5):
        '''
        Segments orientation lines from an input image.
        Output: bw_image with contours of orientation lines
        '''

        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur for better results when finding the threshold
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

        # Choose color of the orientation lines: "W" for white or "B" for black
        if self.line_color == "W":
            img_b_or_w = img_blur
            percentage = percentage_white        #Passes the appropriate value to percentage
            seg_min = 24000                      #Sets limit value (Pixels of largest connected componend) when to choose tresh_1
            iterations_1 = 7                     #Iterations for first round of dilate                  
            iterations_2 = 9                     #Iterations for erode and second dilate
        else:
            if self.line_color == "B":
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
            gray_values = self.grayscale_values(img_b_or_w, y)
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
        img_largest = self.seg_dilate_largest_comp(img_b_or_w, thresh=threshold_2, iterations=iterations_1)

        # If the segmented area is smaller than the minimum threshold, perform the process again with threshold 1 instead of threshold 2
        white_pixels = cv2.countNonZero(img_largest)
        if white_pixels <= seg_min:
            img_largest_correct = self.seg_dilate_largest_comp(img_b_or_w, thresh=threshold_1, iterations=iterations_1)
        else:
            img_largest_correct = img_largest

        # Remove any falsely connected components by applying erode, find_largest_component, dilate again
        img_erode = cv2.erode(img_largest_correct.astype('uint8'), np.ones((3, 3)), iterations=iterations_2)

        img_largest_2 = self.find_largest_component(img_erode)

        img_out = cv2.dilate(img_largest_2.astype('uint8'), np.ones((3, 3)), iterations=int(iterations_2 * 1.5))


        return img_out