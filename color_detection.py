import cv2
import numpy as np

# Using "0" instead of a file name will use the webcam feed
# In the future replace this with the source to the ESP32 web server
video_capture = cv2.VideoCapture(0)

# These are random values for now
lower_bound = np.array([10, 60, 90])
upper_bound = np.array([100, 230, 250])

def BGR_TO_HSV(color):
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    return hsv

thing = BGR_TO_HSV(np.uint8([[[114, 158, 199]]]))

print(thing)

print(BGR_TO_HSV(np.uint8([[[97, 101, 14]]])))

# Throw an error and exit if the video file cannot be opened (not applicable in this case)
if not video_capture.isOpened():
    print("Error: video file could not be opened")
    video_capture.release()

    exit()

# Loop infinitely through the video file/webcam feed, retrieving frames and displaying them
while True:

    # The read() function returns a tuple that contains a boolean whether the frame was retrieved
    # and something that essentially represents the video frame 
    present, frame = video_capture.read()
    
    # If there is no frame retrieved, break out of the loop
    if not present:
        break

    # Creates an hsv color-space version of the frame
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create the mask which takes in the frame and uses the lower and upper bounds defined earlier
    # as the color range
    mask = cv2.inRange(hsvFrame, lower_bound, upper_bound)

    # These two lines of code will dilate the contours that are found in the picture, which can help
    # eliminate noise
    kernel = np.ones((4, 4), "uint8")

    mask = cv2.dilate(mask, kernel)
    
    # Using cv2.bitwise_and() we can modify the frame so that it only displays the parts of the frame
    # that match the desired color
    blackout = cv2.bitwise_and(hsvFrame, hsvFrame, mask = mask)

    # This creates the contours with the mask we defined earlier. The parameter cv2.RETR_TREE is the contour 
    # retrieval mode, and the cv2.CHAIN_APPROX_NONE parameter is the contour approximation method, and we 
    # select none since we want to display all contours
    # More in depth explanations of hierarchy and contour retrieval modes can be found here:
    # https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
    # https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)

    # Show the webcam feed
    cv2.imshow("Webcam Feed", frame)

    # Shows the same feed but filters out anything that doesn't match the color
    cv2.imshow("Mask", blackout)
    

    # If the key pressed by the user is q, break out of the loop
    if cv2.waitKey(1) == ord('q'):
        break

# These two lines of code will release the video capture object and close all of the
# windows that were previously open
video_capture.release()
cv2.destroyAllWindows()