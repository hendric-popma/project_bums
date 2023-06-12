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


# Read and process the video frames
c = 0
save_cnts = 0
wait = 0
output = []
dist = []
koords = []
while video.isOpened() and c < 300:
    # Read a single frame
    ret, frame = video.read()
    frame_show = frame.copy()
    # Check if the frame was successfully read
    if not ret:
        break

    # chaneg to old thresh
    #img_thresh = thresh_gauss(frame)
    #img_cont = find_max_contour(img_thresh)

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    img_cont = seg_orientation_lines(frame, "W")
    
    
    cut = int(3/5 * frame_height)
    black_up = img_cont[:cut,:]
    black_down = img_cont[cut:,:]
    #find center of both halfs
    c_black_up, koords_up = find_center_plot(black_up)
    c_black_down, koords_down = find_center_plot(black_down)
    # if koords are center koords take the koords from the "run" bevore
    if len(koords) > 0: #otherwise error in first run
        if koords_up == [int(val/2) for val in list(frame.shape[:2])]:
            koords_up = koords[0]
        if koords_down == [int(val/2) for val in list(frame.shape[:2])]:
            koords_down = koords[1]
    
    
    #concatenate both halfs
    c_new = np.concatenate((c_black_up, c_black_down))
    # make line through middle 
    koords = calc_line_koords(koords_up[0], koords_up[1], koords_down[0], koords_down[1]+cut, [0,frame_height])
    img_line = cv2.line(c_new, koords[0], koords[1], [0,0,0], 20)
    frame_show = cv2.line(frame_show, koords[0], koords[1], [0,0,0], 20)
    canny = cv2.Canny(img_line, 0, 0)
    
    img_cut = img_line.copy()
    one = canny[frame_height-10,:] > 0

    #xposition where img is white at bottom
    on = np.where(one==True)

    # TODO make smarter    
    dist_min = frame_height*0.1
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
    
    new = cv2.merge((img_line, img_line, img_line))
    if wait <2:
        res_text = "straight"
        output.append(0)
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
                output.append(1)
                dist.append(frame_height-koords_hor[0][1])
            if koords_hor[0][0] > (koords[0][0]+koords[1][0])/2:
                res_text = "right"
                #rigth is two
                output.append(2)
                dist.append(frame_height-koords_hor[0][1])
        elif len(koords_hor) == 2:
#TODO check position of the cnts 
            if koords_hor[0][0] < (koords[0][0]+koords[1][0])/2 and koords_hor[1][0] > (koords[0][0]+koords[1][0])/2:
                res_text = "left and right"
                #left and right is 3
                output.append(3)
                dist.append(frame_height-koords_hor[0][1])
            elif koords_hor[1][0] < (koords[0][0]+koords[1][0])/2 and koords_hor[0][0] > (koords[0][0]+koords[1][0])/2:
                res_text = "left and right"
                #left and right is 3
                output.append(3)
                dist.append(frame_height-koords_hor[0][1])
    
    ubers = {
        0 : "straight",
        1 : "left", 
        2 : "right",
        3: "left and right"   
    }


    #smooth the output, so that it want change if only one value change 
    if len(output) > 3:
        if output[-1] == output[-2]:
            res_text = ubers[output[-1]]
        elif output[-1] != output[-2]: 
            res_text = ubers[output[-3  ]]
        elif output[-1] == output[-3]:
            res_text = ubers[output[-1]]

    res_text = res_text + str(dist[-1])
    print(res_text)

    frame_show = put_text_image(frame_show, res_text)

    c = c+1
    #print(c)
    cv2.imshow('Modified Frame', frame_show)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video objects and close the windows
video.release()
cv2.destroyAllWindows()

print(output)