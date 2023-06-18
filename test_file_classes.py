#%%
from tools_class import * 
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from tools import *
plt.gray()

frames = read_video_frames("/Users/hendricpopma/Documents/Uni/Uni_6_Sem/Bums/test_videos/kempten1.MOV")

frames = frames[0:300]
frames = frames[10][1]
#plt.imshow(frames)
frame = FrameObject(frames)
canny = frame.canny()
plt.imshow(canny)
# %%
