import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from tools_class import *
import sys
#import pyttsx3 
plt.gray()

# Open the video file

video, line_color, total_frames = user_input()
print(total_frames)
#video = cv2.VideoCapture("/Users/hendricpopma/Documents/Uni/Uni_6_Sem/Bums/test_videos/muenchen4.mp4")

# declare some list and start values which are needed
save_cnts = 0
wait = 0

# init class/values for while loop
direct = FindDirection()

while video.isOpened():
    # Read a single frame
    ret, frame = video.read()
    if not ret:
        print("Error load frame")
        break
    
    #Create a Frame Object
    frame = FrameObject(frame)
    img = frame.img

    # Create Segmemtation Object  
    seg = Segmentation(frame, line_color)
    img_cont = seg.seg_orientation_lines()


    frame.build_center_line(img_cont)
    img_cut = frame.make_block_over_center_line()
    
    #def waiting():
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
    
    # Get Values from FrameObject to Direction Object
    direct.get_values_from_frame_object(frame)
    direct.check_where_to_go(wait,cnts)
    res_text = direct.smooth_output()
    frame.put_text_frame(res_text)
    frame_show = frame.draw_seg_orientationline(img_cont)
    
    direct.add_one_counter()
    cv2.imshow('Modified Frame',frame_show)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if video.get(cv2.CAP_PROP_POS_FRAMES) == total_frames:
        break
# Release the video objects and close the windows
video.release()
cv2.destroyAllWindows()