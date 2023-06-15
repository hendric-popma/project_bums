import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from tools import *
plt.gray()

# Open the video file
video = cv2.VideoCapture("/Users/hendricpopma/Documents/Uni/Uni_6_Sem/Bums/test_videos/kempten1.MOV")
#video = cv2.VideoCapture("/Users/hendricpopma/Documents/Uni/Uni_6_Sem/Bums/test_videos/muenchen2.mp4")

# Check if the video file was successfully opened
if not video.isOpened():
    print("Error opening video file.")


# declare some list and start values which are needed
c = 0 #TODO not used delete at end
save_cnts = 0
wait = 0
output = []
dist = []
koords = []
only_straight = []
# Read and process the video frames
while video.isOpened():# and c < 300:
    # Read a single frame
    ret, frame = video.read()
    frame_show = frame.copy()
    # Check if the frame was successfully read
    if not ret:
        break
    
    # get sizes of the frame 
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # segmentation of the frame, to get black and white picture with the lines
    img_cont = seg_orientation_lines(frame, "W")
    
    # cut picture to find two center points to draw the line 
    cut = int(3/5 * frame_height)
    black_up = img_cont[:cut,:]
    black_down = img_cont[cut:,:]

    #find center of both halfs
    c_black_up, koords_up = find_center_plot(black_up)
    c_black_down, koords_down = find_center_plot(black_down)
    # if koords are center koords take the koords from the "run" before
    
    if len(koords) > 0: #TODO check if necessary #otherwise error in first run
        if koords_up == [int(val/2) for val in list(frame.shape[:2])]:
            koords_up = koords[0]
        if koords_down == [int(val/2) for val in list(frame.shape[:2])]:
            koords_down = koords[1]
    
    
    #concatenate both halfs two full picture again
    c_new = np.concatenate((c_black_up, c_black_down))
    # make line through middle 
    koords = calc_line_koords(koords_up[0], koords_up[1], koords_down[0], koords_down[1]+cut, [0,frame_height])
    img_line = cv2.line(c_new, koords[0], koords[1], [0,0,0], 20) # not used
    frame_show = cv2.line(frame_show, koords[0], koords[1], [0,0,0], 20)
    canny = cv2.Canny(img_line, 0, 0)
    
    img_cut = img_line.copy() #TODO change too frame_show
    
    # check for zeroes at bottom of frame to find line from canny 
    one = canny[frame_height-10,:] > 0
    #xposition where img is white at bottom
    on = np.where(one==True)

    # TODO make smarter  
    # make black line over the big white straight line (vertical)
    # if no line at bottom go up until line was found   
    dist_min = frame_height*0.1
    # check that are enough space to the edge of frame
    if len(on[0]) >= 2 and on[0][0] > dist_min and on[0][-1] < (frame_height-dist_min):
        #make black in line 
        img_cut[:,on[0][0]-10:on[0][-1]+10] = 0

    else: 
        offset_line_canny = int(frame_height * 0.2) #for horizontal video offset 0.8
        ycheck = frame_height-offset_line_canny
        # put y position up as long there is no white space found
        while len(on[0]) == 0:
            one = canny[ycheck,:] > 0
            on = np.where(one==True)
            ycheck = ycheck - offset_line_canny  
        img_cut[:,on[0][0]-10:on[0][-1]+10] = 0

    # find contours after making the straight line black
    # with these contours we check where we can go
    cnts,_ = cv2.findContours(img_cut, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # when cnts come in picture wait bevor findcontours so that contour is in complete picture solved
    img_cnts = cv2.drawContours(img_cut,cnts, -1, [255,0,0], 10)
    if len(cnts) == 0 and save_cnts == 0: 
        wait = 0
    if len(cnts) > 0 and save_cnts == 0: 
        wait = wait + 1
    if wait >= 1: # count up if bigger then one
        wait = wait +1 
    save_cnts = len(cnts) #save for next while step
    
    # check if you cannot go straight
    one2 = canny[int(1/3 * frame_height),:] > 0
    one3 = canny[int(0.5 * frame_height),:] > 0

    #xposition where img is white at bottom
    on2 = np.where(one2==True)
    on3 = np.where(one3==True)
    if len(on2[0]) < 2 and len(on3[0] < 2): 
        print("straight not possible")
        only_straight.append(1)
    else:
        only_straight.append(0)



    new = cv2.merge((img_line, img_line, img_line)) # TODO not used
    if wait <2:
        # check for zeroes at bottom of frame to find line from canny 
        if only_straight[-1] == 1 and only_straight[-2] == 1 and only_straight[-3] == 1:
            res_text = "no line"
            output.append(4)
        else:
            output.append(0)
            res_text = "straight"
        dist.append("")
    if wait > 2:
        
        cnts_idx = []
        for i in range(0,len(cnts)):
            # Calculate the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(cnts[i])
            if w/h > 1:
                new = cv2.drawContours(new, cnts, i, [255,0,0], 2)
                frame_show = cv2.drawContours(frame_show, cnts, i, [255,0,0], 2)
                cnts_idx.append(i)
        

        # TODO hold contours to y value and then search again need pixel change per frame or all 5 frames 

        koords_hor = []        
        for i in cnts_idx:
            _,cent = find_center_plot(cnts[i])
            new = cv2.circle(new,cent,10, [255,0,0], cv2.FILLED)
            frame_show = cv2.circle(frame_show,cent,10, [255,0,0], cv2.FILLED)
            koords_hor.append(cent)
        
        #check if koords_hor ist left or right from the middle or both 
        if len(koords_hor) == 1:
            if koords_hor[0][0] < (koords[0][0]+koords[1][0])/2:
                res_text = "left"
                #left is one
                dist.append(frame_height-koords_hor[0][1])
                if only_straight[-1] == 1 and only_straight[-2] == 1 and only_straight[-3] == 1:
                    res_text = "only left"
                    output.append(11)
                else:
                    output.append(1)
                
            if koords_hor[0][0] > (koords[0][0]+koords[1][0])/2:
                res_text = "right"
                #rigth is two
                dist.append(frame_height-koords_hor[0][1])
                if only_straight[-1] == 1 and only_straight[-2] == 1 and only_straight[-3] == 1:
                    res_text = "only right"
                    output.append(22)
                else:
                    output.append(2)
        elif len(koords_hor) == 2:
#TODO check position of the cnts 
            if koords_hor[0][0] < (koords[0][0]+koords[1][0])/2 and koords_hor[1][0] > (koords[0][0]+koords[1][0])/2:
                res_text = "left and right"
                #left and right is 3
                dist.append(frame_height-koords_hor[0][1])
                if only_straight[-1] == 1 and only_straight[-2] == 1 and only_straight[-3] == 1:
                    res_text = "only left and right"
                    output.append(33)
                else:
                    output.append(3)
            elif koords_hor[1][0] < (koords[0][0]+koords[1][0])/2 and koords_hor[0][0] > (koords[0][0]+koords[1][0])/2:
                res_text = "left and right"
                #left and right is 3
                output.append(3)
                dist.append(frame_height-koords_hor[0][1])
                if only_straight[-1] == 1 and only_straight[-2] == 1 and only_straight[-3] == 1:
                    res_text = "only left and right"
                    output.append(33)
                else:
                    output.append(3)
    
    ubers = {
        0 : "straight",
        1 : "left", 
        2 : "right",
        3: "left and right",
        4 : "no orientation line",
        11 : "only left", 
        22: "only right", 
        33 : "only left and right",
    }


    #smooth the output, so that it want change if only one value change 
    if len(output) > 3:
        if output[-1] == output[-2]:
            res_text = ubers[output[-1]]
        elif output[-1] != output[-2]: 
            res_text = ubers[output[-3]]
        elif output[-1] == output[-3]:
            res_text = ubers[output[-1]]

    res_text = res_text + str(dist[-1])
    print(res_text)

    frame_show = put_text_image(frame_show, res_text)

    c = c+1
    #print(c)
    frame_show = draw_seg_orientationline(frame_show, img_cont)
    cv2.imshow('Modified Frame', frame_show)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video objects and close the windows
video.release()
cv2.destroyAllWindows()
