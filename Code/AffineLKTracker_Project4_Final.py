# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 18:22:36 2019

@author: psanh
"""

import numpy as np
import cv2
import glob


"""@param This uses the mean of the template and the mean of the average cropped frames, and
calculates a z-score to adjust pixel values to take into account the change in lighting. This returns 
a new image with adjusted pixel values to get closer to the average of the template."""

def shift_data(tmp_mean, img):
    
    tmp_mean_matrix = np.full((img.shape), tmp_mean)
    img_mean_matrix  = np.full((img.shape), np.mean(img))    
    std_ = np.std(img)    
    z_score = np.true_divide((img.astype(int) - tmp_mean_matrix.astype(int)), std_)
    
    dmean = np.mean(img)-tmp_mean
    
    if dmean < 10:
        shifted_img = -(z_score*std_).astype(int) + img_mean_matrix.astype(int)
        
    else:
        shifted_img = (z_score*std_).astype(int) + img_mean_matrix.astype(int)
    
    return shifted_img.astype(dtype = np.uint8)


def run_code(frame_path, frame, rect, thresh, scale):
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    template = frame[rect[0][1]:rect[1][1] , rect[0][0]:rect[1][0]]
    p = np.zeros(6) 
    
    for path in frame_path:
        img = cv2.imread(path)
        clone = img.copy()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        p, new_rect = affineLKtracker(img, template, rect, p, thresh, scale)
        
        cv2.rectangle(clone, new_rect[0], new_rect[1], (0, 0, 255), 2)
        
        cv2.imshow('Frame', clone)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        
def run_code_vase(frame_path, frame, rect, rect1):
    
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    template = frame[rect[1][1]:rect[0][1] , rect[0][0]:rect[1][0]]
    template1 = frame[rect1[0][1]:rect1[1][1] , rect1[0][0]:rect1[1][0]]
    
    p = np.zeros(6)
    p1 = np.zeros(6) 
    
    for path in frame_path:
        img = cv2.imread(path)
        clone = img.copy()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=cv2.GaussianBlur(img,(5,5),5)
        p, new_rect = affineLKtracker_vase(img, template, rect, p)
        p1, new_rect1 = affineLKtracker_vase1(img, template1, rect1, p1)
        
        rect_pts=np.array((new_rect[0],new_rect1[0],new_rect[1],new_rect1[1],new_rect[0]))
        cv2.polylines(clone,[rect_pts],False,(0, 255, 0),5)
        
        cv2.imshow('Frame', clone)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

            
def affineLKtracker(img, tmp, rect, p, thresh, scale):
    
    W = np.array([[1+p[0], p[2], p[4]], [p[1], 1+p[3], p[5]]])
    check = 10

    while check > thresh:
 
        Ix = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, ksize = 5)
        Iy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, ksize = 5)
        
        warped = cv2.warpAffine(img, W, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        
        warped = warped[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]

        Ix = cv2.warpAffine(Ix, W, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        Iy = cv2.warpAffine(Iy, W, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        
        Ix = Ix[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
        Iy = Iy[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
        
        error = (tmp.astype(int)-warped.astype(int)).reshape(-1,1)
        
        B = np.zeros((tmp.shape[0]*tmp.shape[1],6))

        H = np.zeros((6,6))
        
        xv, yv = np.meshgrid(range(tmp.shape[1]), range(tmp.shape[0]))
        xv = xv.reshape(-1,1)
        yv = yv.reshape(-1,1)
        
        for i in range(0, len(xv)):

            jacobian = np.array([[xv[i][0], 0, yv[i][0], 0, 1, 0],[0, xv[i][0], 0, yv[i][0], 0 , 1]])

            I = np.array([Ix[yv[i][0]][xv[i][0]], Iy[yv[i][0]][xv[i][0]]])
            
            B[i] = np.dot(I,jacobian).reshape(1,-1)
               
        H = B.T @ B

        dp = np.linalg.inv(H) @ B.T @ error
        
        check = np.linalg.norm(dp)
        
        dp = scale*dp
   
        for i in range(0, len(p)):
            p[i] = p[i] + dp[i]
            
        W = np.array([[1+p[0], p[2], p[4]], [p[1], 1+p[3], p[5]]])
    
    newrow = [0,0,1]
    matrix = np.vstack([W,newrow])
    
    pt1 = matrix @ np.array([rect[0][0], rect[0][1], 1]).reshape(1,-1).T
    pt2 = matrix @ np.array([rect[1][0], rect[1][1], 1]).reshape(1,-1).T
    
    new_rect = [(int(pt1[0][0]),int(pt1[1][0])),(int(pt2[0][0]),int(pt2[1][0]))]
       
    return p, new_rect


def affineLKtracker_vase(img, tmp, rect, p):
    
    W = np.array([[1+p[0], p[2], p[4]], [p[1], 1+p[3], p[5]]])


    for _ in range(20):
        
        Ix = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, ksize = 5)
        Iy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, ksize = 5)
        
        warped = cv2.warpAffine(img, W, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        
        warped = warped[rect[1][1]:rect[0][1], rect[0][0]:rect[1][0]]
        
        Ix = cv2.warpAffine(Ix, W, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        Iy = cv2.warpAffine(Iy, W, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        
        Ix = Ix[rect[1][1]:rect[0][1], rect[0][0]:rect[1][0]]
        Iy = Iy[rect[1][1]:rect[0][1], rect[0][0]:rect[1][0]]

        error = (tmp.astype(int)-warped.astype(int)).reshape(-1,1)
        
        B = np.zeros((tmp.shape[0]*tmp.shape[1],6))

        H = np.zeros((6,6))
        
        xv, yv = np.meshgrid(range(tmp.shape[1]), range(tmp.shape[0]))
        xv = xv.reshape(-1,1)
        yv = yv.reshape(-1,1)
        
        for i in range(0, len(xv)):

            jacobian = np.array([[xv[i][0], 0, yv[i][0], 0, 1, 0],[0, xv[i][0], 0, yv[i][0], 0 , 1]])

            I = np.array([Ix[yv[i][0]][xv[i][0]], Iy[yv[i][0]][xv[i][0]]])
            
            B[i] = np.dot(I,jacobian).reshape(1,-1)
               
        H = B.T @ B

        dp = np.linalg.inv(H) @ B.T @ error
        
        dp = 80*dp
   
        for i in range(0, len(p)):
            p[i] = p[i] + dp[i]
            
        W = np.array([[1+p[0], p[2], p[4]], [p[1], 1+p[3], p[5]]])
    
    newrow = [0,0,1]
    matrix = np.vstack([W,newrow])
    
    pt1 = matrix @ np.array([rect[0][0], rect[0][1], 1]).reshape(1,-1).T
    pt2 = matrix @ np.array([rect[1][0], rect[1][1], 1]).reshape(1,-1).T
    
    new_rect = [(int(pt1[0][0]),int(pt1[1][0])),(int(pt2[0][0]),int(pt2[1][0]))]

        
    return p, new_rect


def affineLKtracker_vase1(img, tmp, rect, p):
    
    W = np.array([[1+p[0], p[2], p[4]], [p[1], 1+p[3], p[5]]])

    for _ in range(20):
        
        Ix = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, ksize = 5)
        Iy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, ksize = 5)
        
        warped = cv2.warpAffine(img, W, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        
        warped = warped[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
        
        Ix = cv2.warpAffine(Ix, W, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        Iy = cv2.warpAffine(Iy, W, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        
        Ix = Ix[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
        Iy = Iy[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
        

        error = (tmp.astype(int)-warped.astype(int)).reshape(-1,1)

        
        B = np.zeros((tmp.shape[0]*tmp.shape[1],6))

        H = np.zeros((6,6))
        
        xv, yv = np.meshgrid(range(tmp.shape[1]), range(tmp.shape[0]))
        xv = xv.reshape(-1,1)
        yv = yv.reshape(-1,1)
        
        for i in range(0, len(xv)):

            jacobian = np.array([[xv[i][0], 0, yv[i][0], 0, 1, 0],[0, xv[i][0], 0, yv[i][0], 0 , 1]])

            I = np.array([Ix[yv[i][0]][xv[i][0]], Iy[yv[i][0]][xv[i][0]]])
            
            B[i] = np.dot(I,jacobian).reshape(1,-1)
               
        H = B.T @ B

        dp = np.linalg.inv(H) @ B.T @ error
        
        dp = 80*dp
   
        for i in range(0, len(p)):
            p[i] = p[i] + dp[i]
            
        W = np.array([[1+p[0], p[2], p[4]], [p[1], 1+p[3], p[5]]])
    
    newrow = [0,0,1]
    matrix = np.vstack([W,newrow])
    
    pt1 = matrix @ np.array([rect[0][0], rect[0][1], 1]).reshape(1,-1).T
    pt2 = matrix @ np.array([rect[1][0], rect[1][1], 1]).reshape(1,-1).T
    
    new_rect = [(int(pt1[0][0]),int(pt1[1][0])),(int(pt2[0][0]),int(pt2[1][0]))]

    return p, new_rect


"""Variables to Run Code"""
#Threshholds for tracker and scale to simulate to speed up convergence
thresh_car = 0.006
scale_car = 100

thresh_human = 0.01
scale_human = 19.00009

#Paths to read images - these need to be changed for whoever runs the code in their computer
car_path = glob.glob("car/*.jpg")
human_path = glob.glob("human/*.jpg")
vase_path = glob.glob("vase/*.jpg")

#Rectangles for affine tracker and to crop template
rect_car = [(115,100),(340,276)]
rect_human = [(260,290),(285,358)]

rect_vase = [(115,150),(180,90)]
rect_vase1 = [(115,90),(180,150)]

#First frame of each video, used for template
frame_car = cv2.imread('car/frame0020.JPG')
frame_human = cv2.imread('human/0140.JPG')
frame_vase = cv2.imread('vase/0019.JPG')


"""Run Code"""
print("\n Video Options:\n -Car\n -Human\n -Vase\n")
Video_input = input("Please Enter the video option you want to see: ")

if Video_input == 'Car' or Video_input == 'car':
    run_code(car_path ,frame_car, rect_car, thresh_car, scale_car)
    
elif Video_input == 'Human' or Video_input == 'human':
    run_code(human_path ,frame_human, rect_human, thresh_human, scale_human)
    
elif Video_input == 'Vase' or Video_input == 'vase':
    run_code_vase(vase_path, frame_vase, rect_vase, rect_vase1)
    
else:
    print("Invalid or no option selected - Please run code again")
    
cv2.destroyAllWindows()
        