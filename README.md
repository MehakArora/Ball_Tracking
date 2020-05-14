# Tracking a ball's trajectory using OpenCV and a Python 3.8 implementation of Kalman Filters

This project is being done for multiple applications, primarily to study the translational motion of objects.

I hope to extend it to tracking multiple objects (maybe players in a basketball video) to gain more insightful data that could be applied to improve game understanding in various sports.

Videos used for this purpose are of a person shooting a basketball, recorded with a fixed camera. 

High-level Steps implemented:

1. Gray-level Conversion and Background Subtraction
2. Using OpenCV's findContours function to identify contours and processing them to identify circular objects.
3. Detecting object in each frame of the video. 
4. Using Kalman Filters we can eventually move on to detecting objects after skipping certain number of frames in between. This will also be useful in tracking objects that move out of the frame of the video.


References:

1. https://github.com/srianant/kalman_filter_multi_object_tracking
2. http://campar.in.tum.de/Chair/KalmanFilter
3. https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
