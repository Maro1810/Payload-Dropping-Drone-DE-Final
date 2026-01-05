import cv2
import numpy as np
from enum import Enum, auto
from simple_pid import PID
import matplotlib.pyplot as plt
import time

# Create enum class for states
class State(Enum):
    SEARCHING = auto()
    ALIGNING = auto()
    DESCENDING = auto()
    GRABBING = auto()

# Using "0" instead of a file name will use the webcam feed
# In the future replace this with the source to the ESP32 web server
video_capture = cv2.VideoCapture(0)

red_lower_bound = np.array([120, 90, 150])
red_upper_bound = np.array([180, 255, 255])

green_lower_bound = np.array([25, 40, 40])
green_upper_bound = np.array([105, 255, 255])

pidX = PID(0.005, 0, 0, 0) # Tune these constants
pidY = PID(0.005, 0, 0, 0) # Tune

deadband = 0 # idk what value to use yet

outputX = 0
outputY = 0

pidX.output_limits = (-1, 1)
pidY.output_limits = (-1, 1)

def BGR_TO_HSV(color):
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    return hsv

def FRAME_SIZE(frame):
    height, width, _ = frame.shape
    
    return height, width


allowed_error = 100

red_center_x = 0
red_center_y = 0

green_center_x = -10
green_center_y = -10

green_area = 0
red_area = 0

average_x = 0
average_y = 0

x_aligned = False
y_aligned = False

x_error = 100
y_error = 100

start_time = time.time()
current_time = start_time

output_x_array = []
output_y_array = []
time_array = []

fig = plt.figure()
fig2 = plt.figure()

ax = fig.add_subplot(1, 1, 1)

ax2 = fig2.add_subplot(1, 1, 1)

# resize the figures cuz they were too large
fig.set_size_inches((2, 2))
fig2.set_size_inches((2, 2))

plt.ion()

# Throw an error and exit if the video file cannot be opened (not applicable in this case)
if not video_capture.isOpened():
    print("Error: video file could not be opened")
    video_capture.release()

    exit()

currentState = State.SEARCHING

# Loop infinitely through the video file/webcam feed, retrieving frames and displaying them
while True:


    current_time = time.time() - start_time

    # Update arrays for MatPlotLib
    output_x_array.append(outputX)
    output_y_array.append(outputY)
    time_array.append(current_time)

    red_contour = None
    green_contour = None

    red_moment = None
    green_moment = None

    ax.plot(time_array, output_x_array)
    ax2.plot(time_array, output_y_array)

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
    red_mask = cv2.inRange(hsvFrame, red_lower_bound, red_upper_bound)

    green_mask = cv2.inRange(hsvFrame, green_lower_bound, green_upper_bound)

    # These two lines of code will dilate the contours that are found in the picture, which can help
    # eliminate noise
    kernel = np.ones((4, 4), "uint8")

    red_mask = cv2.dilate(red_mask, kernel)
    
    green_mask = cv2.dilate(green_mask, kernel)
    
    # Using cv2.bitwise_and() we can modify the frame so that it only displays the parts of the frame
    # that match the desired color
    blackout = cv2.bitwise_and(hsvFrame, hsvFrame, mask = red_mask)
    blackout2 = cv2.bitwise_and(hsvFrame, hsvFrame, mask = green_mask)

    # This creates the contours with the mask we defined earlier. The parameter cv2.RETR_TREE is the contour 
    # retrieval mode, and the cv2.CHAIN_APPROX_NONE parameter is the contour approximation method, and we 
    # select none since we want to display all contours
    # More in depth explanations of hierarchy and contour retrieval modes can be found here:
    # https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
    # https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    green_contours, __ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    height, width = FRAME_SIZE(frame)

    pidX.setpoint = width // 2
    pidY.setpoint = height // 2
    
    # Target square
    target = (int((width/2)-20), int((height/2)-20), int((width/2)+20), int((height/2)+20))

    
    best_red_contour = None
    best_red_contour_area = 0

    # Update contours red and green contours if they are in view on screen
    for i, contour in enumerate(red_contours):

        if cv2.contourArea(contour) > 6000:

            if cv2.contourArea(contour) > best_red_contour_area:        
                best_red_contour = contour
                best_red_contour_area = cv2.contourArea(contour)

            red_contour = best_red_contour

    best_green_contour = None
    best_green_contour_area = 0

    for i, contour in enumerate(green_contours):


        if cv2.contourArea(contour) > 6000:
            
            if cv2.contourArea(contour) > best_green_contour_area:        
                best_green_contour = contour
                best_green_contour_area = cv2.contourArea(contour)

        green_contour = best_green_contour

    if green_contour is not None and red_contour is not None:
        green_area = cv2.contourArea(green_contour)
        red_area = cv2.contourArea(red_contour)

        
        red_moment = cv2.moments(red_contour)
        green_moment = cv2.moments(green_contour)
    
        if red_moment['m00'] != 0 and green_moment['m00'] != 0:

    
            red_center_x = int(red_moment['m10']/red_moment['m00'])
            red_center_y = int(red_moment['m01']/red_moment['m00'])

            green_center_x = int(green_moment['m10']/green_moment['m00'])
            green_center_y = int(green_moment['m01']/green_moment['m00'])

            average_x = int((red_center_x+green_center_x)/2)
            average_y = int((red_center_y+green_center_y)/2)

            x_error = abs((red_center_x - green_center_x)/red_center_x*100)
            y_error = abs((red_center_y - green_center_y)/red_center_y*100)

        if (x_error < allowed_error) and (y_error < allowed_error):

            currentState = State.ALIGNING
        
            if not x_aligned and not y_aligned:
                # direction = "right" if average_x > target[2] else "left"
                # print("Move " + direction)    
                outputX = pidX(average_x)
                outputY = 0

                print(outputX)

            elif not x_aligned:
                # direction = "right" if average_x > target[2] else "left"
                # print("Move " + direction)
                outputX = pidX(average_x)


            elif not y_aligned:
                # direction = "down" if average_y < target[1] else "up"
                # print("Move " + direction)
                outputY = pidY(average_y)
                outputX = 0

            cv2.drawContours(frame, red_contour, -1, (0, 255, 0), 3)
            cv2.drawContours(frame, green_contour, -1, (0, 255, 0), 3)

            cv2.circle(frame, (average_x, average_y), 3, (0, 255, 255), 3)
        else:
            average_x = 0
            average_y = 0

    else:
        currentState = State.SEARCHING


    x_aligned = True if (average_x > target[0] and average_x < target[2]) else False
    y_aligned = True if (average_y > target[1] and average_y < target[3]) else False

    if x_aligned and y_aligned:
        cv2.putText(frame, "Aligned", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        currentState = State.DESCENDING

    else:
        cv2.putText(frame, "Not Aligned", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    cv2.putText(frame, currentState.name, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    cv2.rectangle(frame, (target[0], target[1]), 
                  (target[2], target[3]), (0, 255, 0), 3)

    # update matplotlib graphs
    plt.pause(0.001)

    # Show the camera feed
    cv2.imshow("Camera Feed", frame)

    # Shows the same feed but filters out anything that doesn't match the color
    # cv2.imshow("Red Mask", blackout)
    # cv2.imshow("Green Mask", blackout2)
    
    # If the key pressed by the user is q, break out of the loop
    if cv2.waitKey(1) == ord('q'):
        break

print(f"Exited program at timestamp: {current_time} seconds")
# These two lines of code will release the video capture object and close all of the
# windows that were previously open
video_capture.release()
cv2.destroyAllWindows()