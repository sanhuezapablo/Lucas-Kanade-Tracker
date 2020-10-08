# Lucas-Kanade-Tracker

## **PROJECT DESCRIPTION**

The aim of this project is to implement the Lucas-Kanade (LK) template tracker. Then to evaluate the code on three video sequences from the Visual Tracker benchmark database: featuring a car on the road, a human walking, and a box on a table.

To initialize the tracker, I define a template by drawing a bounding box around the object to be tracked in the first frame of the video. For each of the subsequent frames the tracker will update an affine transform that warps the current frame so that the template in the first frame is aligned with the warped current frame.

Please refer to [Project Report](https://github.com/sanhuezapablo/Lucas-Kanade-Tracker/blob/master/Report/FINAL%20REPORT.pdf) for further description

### Implementation of the Tracker

<p align="center">
  <img src="/Images/bounding_box.png" alt="Input">
</p>

Initialize manually the coordinates of a bounding box surrounding the object to be tracked.

At the core of the algorithm, I get as input a grayscale image of the current frame (img), the template image (tmp), the bounding box (rect) that marks the template region in tmp, and the parameters (p_prev) of the previous warping.

Then I iteratively compute the new warping parameters (p_new) and return these parameters.

The algorithm computes the affine transformations from the template to every frame in the sequence and draws the bounding boxes of the rectangles warped from the first frame.

#### LK Algorithm

<p align="center">
  <img src="/Images/lk_algo.png" alt="LK Algo">
</p>

Refer to [Project Report](https://github.com/sanhuezapablo/Lucas-Kanade-Tracker/blob/master/Report/FINAL%20REPORT.pdf) or [References](https://github.com/adheeshc/Lucas-Kanade-Tracker/tree/master/References) for further explanation

### Evaluation of the Tracker

<p align="center">
  <img src="/Images/car_tracker_new.gif" alt="car">
</p>

<p align="center">
  <img src="/Images/human_tracker.gif" alt="human">
</p>

<p align="center">
  <img src="/Images/vase_tracker.gif" alt="vase">
</p>

The tracker is then evaluated on the three sequences: the car sequence, the human walking, and the table top scene.

### Robustness to Illumination

The LK tracker as it is formulated, breaks down when there is a change in illumination because the sum of squared distances error that it tries to minimize is sensitive to illumination changes. We fix this using two techniques - 1) Adaptive brightness fix and 2) Gamma Correction

#### Adaptive Brightness Fix

<p align="center">
  <img src="/Images/car_adaptive.gif" alt="adaptive">
</p>

Here, I scale the brightness of pixels in each frame so that the average brightness of pixels in the tracked region stays the same as the average brightness of pixels in the template. Basically, I shift the pixel values to adapt to the brightness by changing them by a certain z-score relative to the mean of the template. 

#### Gamma Correction Fix

<p align="center">
  <img src="/Images/car_gamma.gif" alt="gamma">
</p>

if we use a high gamma correction, we can brighten images. However, this may cause potential errors as it is not adaptive in nature making it less robust than the previous method. Hence, only if the norm value of the current frame was lesser than that of the template, a gamma correction was applied.

## **DEPENDANCIES**

- Python 3
- OpenCV
- Numpy
- Glob (built-in)


## **FILE DESCRIPTION**

- Code Folder/[AffineLKTracker_Project4_Final.py](https://github.com/sanhuezapablo/Lucas-Kanade-Tracker/blob/master/Code/AffineLKTracker_Project4_Final.py) - This is the file that works for all datasets
- Code Folder/[EXTRA CREDIT - Car_LKAffine_ShiftBrightness.py](https://github.com/sanhuezapablo/Lucas-Kanade-Tracker/blob/master/Code/EXTRA%20CREDIT%20-%20Car_LKAffine_ShiftBrightness.py) - This file is used for fixing the brightness issue of the LK Tracker. Implemented to work on the Car video.

- Datasets folder - Contains 3 folders with frames of the 3 videos - car, human and vase 

- Images folder - Contains images for github use (can be ignored)

- Output folder - Contains output videos

- References folder - Contains supplementary documents that aid in understanding

- Report folder - Contains [Project Report](https://github.com/sanhuezapablo/Lucas-Kanade-Tracker/blob/master/Report/FINAL%20REPORT.pdf)

## **RUN INSTRUCTIONS**

- Make sure all dependancies are met
- Ensure the location of the input video files are correct in the code you're running. **Check paths on lines 272,273,274 and 284,285,286**
- **Comment brightness correction methods to switch between gamma correction and adaptive brightness techniques on 190/194**

- RUN AffineLKTracker_Project4_Final.py to run for all 3 datasets. Follow Instructions on screen
- RUN EXTRA CREDIT - Car_LKAffine_ShiftBrightness.py to check the brightness corrected tracker




