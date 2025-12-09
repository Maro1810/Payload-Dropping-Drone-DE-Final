import cv2
import numpy as np

# Using "0" instead of a file name will use the webcam feed
# In the future replace this with the source to the ESP32 web server
video_capture = cv2.VideoCapture(0)

# These are random values for now
lower_bound = np.array([10, 100, 20])
upper_bound = np.array([20, 255, 200])

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
    # # as the color range
    mask = cv2.inRange(hsvFrame, lower_bound, upper_bound)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)

    # Show the webcam feed
    cv2.imshow("Webcam Feed", frame)

    # If the key pressed by the user is q, break out of the loop
    if cv2.waitKey(1) == ord('q'):
        break

# These two lines of code will release the video capture object and close all of the
# windows that were previously open
video_capture.release()
cv2.destroyAllWindows()