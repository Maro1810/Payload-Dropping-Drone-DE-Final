import cv2

video = cv2.VideoCapture("http://192.168.68.55")

if not video.isOpened():
    print("An error has occurred with opening the video")

    video.release()
    exit()

def inRange(a, b, thresh):
    return abs(a-b) <= thresh

while True:
    present, frame = video.read()

    if not present:
        break

    # TODO add comments to explain thresholding
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, thresh_frame = cv2.threshold(gray_frame, 220, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):

        # We skip the first index since it is the the entire image
        if i == 0:
            continue
        
        # This essentially indicates how accurate we want the polygon approximation to be
        # It specifies the max distance the approximated shape contour can be from the original
        # contour. More information here: https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
        # The boolean parameter specifies whether the shape is closed or not. Since we are looking for a 
        # rectangle, we specify the parameter as true.
        epsilon = 0.12*cv2.arcLength(contour, True)   

        # This is the polygon approximation, which takes in three parameters, the contours used for the
        # approximation, the accuracy of the approximation (epsilon value), and whether the shape is closed
        # or not
        approx = cv2.approxPolyDP(contour, epsilon, True)

        
        # Create a bounding rectangle for each of the shapes
        x, y, w, h = cv2.boundingRect(approx)

        xApprox = approx[0][0][0]
        yApprox = approx[0][0][1]

        # xContour = contour[0][0][0]
        # yContour = contour[0][0][1]

        if (len(approx) == 4 and inRange(x, xApprox, 10) and inRange(y, yApprox, 10) and (h > 10 or w > 10)):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # cv2.drawContours(frame, approx, -1, (0, 255, 0), 5)

        # if (inRange(x, xApprox, 10) and inRange(y, yApprox, 10)):
        #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw the bounding rectangle on screen
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("original", frame)
    cv2.imshow("thresholded", thresh_frame)

    if cv2.waitKey(1) == ord('q'):
        break


video.release()
cv2.destroyAllWindows()