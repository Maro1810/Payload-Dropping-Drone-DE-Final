import cv2
import numpy as np

# Using "0" instead of a file name will use the webcam feed
# In the future replace this with the source to the ESP32 web server
video_capture = cv2.VideoCapture(0)

# These are random values for now 
red_lower_bound = np.array([160, 120, 160])
red_upper_bound = np.array([180, 255, 255])

# Tune these upper and lower bounds for the green color
green_lower_bound = np.array([35, 40, 40])
green_upper_bound = np.array([85, 255, 255])

def BGR_TO_HSV(color):
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    return hsv

def FRAME_SIZE(frame):
    height, width, _ = frame.shape
    
    return height, width

red_center_x = 0
red_center_y = 0

green_center_x = 0
green_center_y = 0

x_aligned = False
y_aligned = False

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

    red_contour = None

    green_contour = None

    for i, contour in enumerate(red_contours):
        if cv2.contourArea(contour) > 6000:
            # cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
            M = cv2.moments(contour)

            red_contour = contour
            
            # Calculate the center of the contour, where m00 represents the area, and m10 and m01
            # are the weighted sums of the x and y values
            red_center_x = int(M['m10']/M['m00'])
            red_center_y = int(M['m01']/M['m00'])

            cv2.circle(frame, (red_center_x, red_center_y), 3, (0, 255, 255), 3)

    for i, contour in enumerate(green_contours):
        if cv2.contourArea(contour) > 6000:
            cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)

            green_contour = contour

            # M = cv2.moments(contour)

            # red_center_x2 = int(M['m10']/M['m00'])
            # red_center_y2 = int(M['m01']/M['m00'])

            # cv2.circle(frame, (red_center_x, red_center_y), 3, (255, 0, 255), 3)


    height, width = FRAME_SIZE(frame)


    # Target square
    target = (int((width/2)-20), int((height/2)-20), int((width/2)+20), int((height/2)+20))

    x_aligned = True if (red_center_x > target[0] and red_center_x < target[2]) else False
    y_aligned = True if (red_center_y > target[1] and red_center_y < target[3]) else False


    if x_aligned and y_aligned:
        cv2.putText(frame, "Aligned", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Not Aligned", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.rectangle(frame, (target[0], target[1]), 
                  (target[2], target[3]), (0, 255, 0), 3)

    # Show the webcam feed
    cv2.imshow("Webcam Feed", frame)

    # Shows the same feed but filters out anything that doesn't match the color
    cv2.imshow("Red Mask", blackout)
    cv2.imshow("Green Mask", blackout2)
    

    # If the key pressed by the user is q, break out of the loop
    if cv2.waitKey(1) == ord('q'):
        break

# These two lines of code will release the video capture object and close all of the
# windows that were previously open
video_capture.release()
cv2.destroyAllWindows()