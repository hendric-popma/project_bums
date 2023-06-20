# project_bums:    Visual Impairment Support

<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/de/thumb/d/dc/Hs-kempten-logo.svg/1200px-Hs-kempten-logo.svg.png" width="200" height="150" />
</p>


[project_bums]() is a Python library for visual impairment support for better navigation in public places with existing guiding lines
## Getting started

In order to run the code you need at least Python Version 3.10.0.
It's recommended to install all dependencies in a virtual enviroment.

To install all requirements run:

`pip install -r requirements.txt`


## Example

Bevor you can start with the example, download a test videos from the cloud:

[download_here](https://drive.google.com/drive/folders/1BDMWk-mU7YQDCDMTNvrdcFH8DbeT5F-C?usp=sharing)

The following example can be found in `main.py`


First the code ask the user for the path of the video and the color of the guiding line. After that, an FindDirectionObject ist going to be initialized. 


```python
import cv2
from tools import user_input, FrameObject, FindDirection, Segmentation

# Open the video file
video, line_color, total_frames = user_input()
# init class/values for while loop
direct = FindDirection()
```

Then the code loops through the video. With every frame a FrameObject is going to be initialized. 
```python
#loop through all frames of video
while video.isOpened():
    #Read a single frame
    ret, frame = video.read()
    if not ret:
        print("Error load frame")
        break
    
    #Create a Frame Object
    frame = FrameObject(frame)
    #get image from FrameObject
    img = frame.img
```

After that it creates a SegmentationObject. With the function 'get_orientation_lines', the guide line will be segmentated. The segmented image is going to be passed to the FrameObject. We prepare the image tio find the direction.

```python
    #Create Segmemtation Object  
    seg = Segmentation(frame, line_color)
    #get the orientation line
    img_seg = seg.get_orientation_lines()
    frame.build_center_line(img_seg)
    #used later to find the directions
    img = frame.make_block_over_center_line()
```
The code pass the FrameObject to the DirectionObject. After that it checks where to go. Then the Text will be printed in the image.

```python
    # Get Values from FrameObject to Direction Object
    direct.get_values_from_frame_object(frame)
    direct.wait_for_complete_contour(img)
    direct.find_nearest_line(img_seg)
    direct.check_where_to_go()
    res_text = direct.smooth_output()
    frame.text_in_frame(res_text)
    frame_show = frame.overlay_segmentation(img_seg)
```

At the end, the image will be shown. When the last frame of the video was processed and shown, the window closed and the video will be released.

```python
    cv2.imshow('Processed Frame',frame_show)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if video.get(cv2.CAP_PROP_POS_FRAMES) == total_frames:
        break
    
# Release the video objects and close the windows
video.release()
cv2.destroyAllWindows()
```

## Documentation
We did not have time to create a documentation yet, but there is a presentation about the project. You can find it [here]()

## Thanks

This project has received support from


