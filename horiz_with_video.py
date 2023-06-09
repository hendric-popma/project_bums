import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from tools import *
plt.gray()

# Open the video file
video = cv2.VideoCapture("/Users/hendricpopma/Documents/Uni/Uni_6_Sem/Bums/test_videos/kempten1.MOV")

# Check if the video file was successfully opened
if not video.isOpened():
    print("Error opening video file.")


# Create VideoWriter object to save the modified frames
#output = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Read and process the video frames
c = 0
save_cnts = 0
wait = 0
while video.isOpened():# and c < 300:
    # Read a single frame
    ret, frame = video.read()

    # Check if the frame was successfully read
    if not ret:
        break

    # Perform frame modification (example: convert to grayscale)
    
    #img_thresh = thresh_gauss(frame)
    #img_cont = find_max_contour(img_thresh)

    img_cont = seg_orientation_lines(frame, "W")
    # cut image in half 
    #cut = 1000

    cut = int(4/5 * frame.shape[0])
    black_up = img_cont[:cut,:]
    black_down = img_cont[cut:,:]
    #find center of both halfs
    c_black_up, koords_up = find_center_plot(black_up)
    c_black_down, koords_down = find_center_plot(black_down)
    # concatenate both halfs
    c_new = np.concatenate((c_black_up, c_black_down))
    # make line through middle 
    koords = calc_line_koords(koords_up[0], koords_up[1], koords_down[0], koords_down[1]+cut, [0,frame.shape[0]])
    img_line = cv2.line(c_new, koords[0], koords[1], [0,0,0], 20)
    canny = cv2.Canny(img_line, 0, 0)
    
    img_cut = img_line.copy()
    one = canny[frame.shape[0]-10,:] > 0
    #one = canny[1910,:] > 0
    ### video2
    #one = canny[1070,:] > 0

    #xposition where img is white at bottom
    on = np.where(one==True)

    #dist = abs(on[0][0]-on[0][3])

    #check that line has diff to edge in percent at the moment 10%
    #TODO make functions and TEST IT!!!
    #TODO
    dist_min = frame.shape[0]*0.1
    #print(on[0])
    if len(on) >= 2 and on[0][0] > dist_min and on[0][-1] < (frame.shape[0]-dist_min):
        #make black in line 
        img_cut[:,on[0][0]-10:on[0][-1]+10] = 0
        #in half make black
        img_cut[:700,:] = 0
        #plt.imshow(img_cut)
    else: 
        offset_line_canny = int(frame.shape[0] * 0.2) #for horizontal video offset 0.8
        print(c)
        ycheck = frame.shape[0]-offset_line_canny
        # put y position up as long there is no white space found
        while len(on) == 0:
            one = canny[ycheck,:] > 0
            on = np.where(one==True)
            img_cut[:,on[0][0]-10:on[0][-1]+10] = 0
            #in half make black
            img_cut[:700,:] = 0
            #plt.imshow(img_cut)
            ycheck = ycheck - offset_line_canny

    cnts,_ = cv2.findContours(img_cut, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # when cnts come in picture wait bevor findcontours so that contour is in complete picture solved
    
    if len(cnts) == 0 and save_cnts == 0: 
        wait = 0
    if len(cnts) > 0 and save_cnts == 0: 
        wait = wait + 1
    if wait >= 1: # count up if bigger then one
        wait = wait +1 
    save_cnts = len(cnts) #save for next while step
    
    new = cv2.merge((img_line, img_line, img_line))
    if wait > 2:
        
        cnts_idx = []
        for i in range(0,len(cnts)):
            # Calculate the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(cnts[i])
            if w/h > 1:
                new = cv2.drawContours(new, cnts, i, [255,0,0], 2)
                cnts_idx.append(i)
        

        # TODO hold contours to y value and then search again need pixel change per frame or all 5 frames 
        # check with other videos if nessesary
        koords_hor = []        
        for i in cnts_idx:
            _,cent = find_center_plot(cnts[i])
            new = cv2.circle(new,cent,10, [255,0,0], cv2.FILLED)
            koords_hor.append(cent)
        #plt.imshow(new)
        if len(koords_hor) == 1:
            if koords_hor[0][0] < (koords[0][0]+koords[1][0])/2:
                print("left")
            if koords_hor[0][0] > (koords[0][0]+koords[1][0])/2:
                print("right")
        elif len(koords_hor) == 2:
            #TODO check position of the cnts 
            if koords_hor[0][0] < (koords[0][0]+koords[1][0])/2 and koords_hor[1][0] > (koords[0][0]+koords[1][0])/2:
                print("left and right")
            elif koords_hor[1][0] < (koords[0][0]+koords[1][0])/2 and koords_hor[0][0] > (koords[0][0]+koords[1][0])/2:
                print("left and right")

    c = c+1
    #print(c)
    cv2.imshow('Modified Frame', new)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video objects and close the windows
video.release()
cv2.destroyAllWindows()
