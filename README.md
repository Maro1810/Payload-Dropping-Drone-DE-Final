
# Payload Dropping Drone Project for Digital Electronics

Here you will find the codebase for Sanchay and I's project (although Sanchay has his repository [here](https://github.com/Chrohma/Payload-Dropping-Drone-DE-Final-Sanchay-Koppa))

## **Required Libaries**
- OpenCV
- NumPy
- Matplotlib
- simple-pid

## Overview

The goal of this project was to create a drone that has the capability of autonomously detecting and picking up packages. However, due to time constraints, my partner, Sanchay, and I decided to focus on just the claw mechanism along with the alignment code, which is the part I worked on.

## Structure

Before I started coding, I made a flowchart and a state machine diagram to effectively map out how the code will function. Below, the flowchart shows how the code will go from a video stream -> detecting a package -> giving pseudocommands to the drone on how to move

![Flowchart](imgs/flowchart.png)

I decided to use a state machine approach when designing the alignment code as it provides an easy way to know when the drone is in each of the four states, searching, aligning, descending, and grabbing. Below is the diagram of the possible transitions between the four states.

![StateMachine](imgs/state_machine.png)

## Package Detection

The first part of the program is the package detection which can be split up into two main parts, color masking, and noise reduction. 

#### Color Masking

The way we planned on designing the package was to have two bright colors, red and green, with green on the outer edge of the package, and red on the inside. The reason for this is to make detecting the package easier, and having two colors will reduce the likelihood of detecting an object that is not the package. 

![test_sheet](imgs/testing_sheet.png)

In this project, I mainly used OpenCV for the image processing, and essentially, I first applied two color masks that will filter out everything except for the colors red and green respectively. From these color masks, we can then figure out exactly where the colors red and green show up on screen. The continuous green/red "areas" on screen are known as contours.

#### Noise Reduction

Once we have these two color masks, the next thing I implemented was determining whether the area of the green/red contours were significant enough to be considered. This was done to filter out small "specks" of red and green that might be present in the view of the camera. Next, the program determines the center of the red and green contours using the moments() function from OpenCV. The moments() function can find the center of a contour by taking the weighted sum of the x/y pixel positions multiplied by the pixel intensity, and then dividing these values by the area to get the center-x and center-y. 

After calculating the center of the red and green contours, the program determines whether these centers are close enough to each other to be considered the package. Since the package is rectangular, the center of the green contours and red contours should theoretically be in the same spot.

Finally, if these contour centers are close enough to each other, then they are averaged out, and the "average" center of the package is displayed on screen. 

## Alignment

The alignment part of the program is somewhat simple. Once we know where the center of the package is relative to the center of the screen (assuming that the camera is mounted such that the center of the camera matches the center of the claw), the program can determine whether the drone needs to move left, right, forwards, or backwards to align to the target.

Initially, I believed that this approach of using relative alignment would be able to function fine. However, after discussing with an industry-level engineer, Mr. Marc-Aurele, I realized that it would be better to determine the horizontal distance to the package, and use absolute alignment. This brings us to the next section, which involves using optics.