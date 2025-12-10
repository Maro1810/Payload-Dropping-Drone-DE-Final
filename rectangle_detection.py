import cv2

video = cv2.VideoCapture("imgs/shapes.jpeg")

if not video.isOpened:
    print("An error has occurred with opening the video")

    video.release()
    exit()

while True:
    present, frame = video.read()

    if not present:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, thresh_frame = cv2.threshold(gray_frame, 245, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):

        # We skip the first index since it is the the entire image
        if i == 0:
            continue
        


    cv2.imshow("thresholded", thresh_frame)

    if cv2.waitKey() == ord('q'):
        break


video.release()
cv2.destroyAllWindows()