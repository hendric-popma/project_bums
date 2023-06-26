import threading
import time
import cv2
import matplotlib.pyplot as plt 
from tools import user_input, audio_output, FrameObject, FindDirection, Segmentation

global_res_text = 0
global_previous_text = 0


# Video-Thread
def video_thread( video, line_color, total_frames):

    global global_res_text
    plt.gray()
    
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
        #get image from FrameObject
        img = frame.img
        #Create Segmemtation Object  
        seg = Segmentation(frame, line_color)
        #get the orientation line
        img_seg = seg.get_orientation_lines()
        frame.build_center_line(img_seg)
        #used later to find the directions
        img = frame.make_block_over_center_line()
        
        # Get Values from FrameObject to Direction Object
        direct.get_values_from_frame_object(frame)
        direct.wait_for_complete_contour(img)
        direct.find_nearest_line(img_seg)
        direct.check_where_to_go()
        res_text = direct.smooth_output()
        global_res_text = res_text
        frame.text_in_frame(res_text)
        frame_show = frame.overlay_segmentation(img_seg)

        cv2.imshow('Processed Frame',frame_show)

        # Check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if video.get(cv2.CAP_PROP_POS_FRAMES) == total_frames:
            break
    # Release the video objects and close the windows
    video.release()
    cv2.destroyAllWindows()


# Audio-Thread
def audio_thread():

    global global_res_text
    global global_previous_text

    while video.isOpened():
        # Read a single frame
        ret, frame = video.read()
        if not ret:
            print("Error load frame")
            break
        
        # Überprüfen, ob sich das letzte Element geändert hat
        if global_res_text != " 0" and global_res_text != " ":
            if global_res_text != global_previous_text:
                global_previous_text = global_res_text
                audio_output(global_res_text)   

        time.sleep(0.0005)         
        
     # Check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if video.get(cv2.CAP_PROP_POS_FRAMES) == total_frames:
            break
    # Release the video objects and close the windows
    video.release()
    cv2.destroyAllWindows()
 
video, line_color, total_frames = user_input()

# Threads starten
video_thread = threading.Thread(target=video_thread, args=(video, line_color, total_frames))
audio_thread = threading.Thread(target=audio_thread)

video_thread.start()
audio_thread.start()

video_thread.join()
audio_thread.join()