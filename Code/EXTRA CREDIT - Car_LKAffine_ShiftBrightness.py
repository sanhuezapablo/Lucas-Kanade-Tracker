import numpy as np
import cv2
import glob


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

def adjust_gamma(image, gamma=1.0):
    
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    return cv2.LUT(image, table)


def affineLKtracker_shiftdata(img, tmp, rect, p):
    
    W = np.array([[1+p[0], p[2], p[4]], [p[1], 1+p[3], p[5]]])
    
    check = 10
    thresh = .006

    while check > thresh:

        
        Ix = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, ksize = 5)
        Iy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, ksize = 5)
        
        warped = cv2.warpAffine(img, W, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        
        warped = warped[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
        warped = shift_data(np.mean(tmp), warped)
        
        
        cv2.imshow('warped', warped)
        cv2.imshow('tmp', tmp)
        
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
        
        dp = 100*dp
   
        for i in range(0, len(p)):
            p[i] = p[i] + dp[i]
            
        W = np.array([[1+p[0], p[2], p[4]], [p[1], 1+p[3], p[5]]])
    
    newrow = [0,0,1]
    matrix = np.vstack([W,newrow])
    
    pt1 = matrix @ np.array([rect[0][0], rect[0][1], 1]).reshape(1,-1).T
    pt2 = matrix @ np.array([rect[1][0], rect[1][1], 1]).reshape(1,-1).T
    
    new_rect = [(int(pt1[0][0]),int(pt1[1][0])),(int(pt2[0][0]),int(pt2[1][0]))]
       
    return p, new_rect
    
def affineLKtracker_gammacorrection(img, tmp, rect, p):
    
    W = np.array([[1+p[0], p[2], p[4]], [p[1], 1+p[3], p[5]]])
    #img = cv2.GaussianBlur(img,(5,5),5)
    #tmp = cv2.GaussianBlur(tmp,(5,5),5)
    check = 10
    thresh = .006

    while check > thresh:

        warped = cv2.warpAffine(img, W, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        warped = warped[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
        if np.linalg.norm(warped) < np.linalg.norm(tmp):
            img  = adjust_gamma(img, gamma=1.5)
        
        Ix = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, ksize = 5)
        Iy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, ksize = 5)
        
        cv2.imshow('warped', warped)
        #cv2.imshow('tmp', tmp)
        
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
        
        dp = 100*dp
   
        for i in range(0, len(p)):
            p[i] = p[i] + dp[i]
            
        W = np.array([[1+p[0], p[2], p[4]], [p[1], 1+p[3], p[5]]])
    
    newrow = [0,0,1]
    matrix = np.vstack([W,newrow])
    
    pt1 = matrix @ np.array([rect[0][0], rect[0][1], 1]).reshape(1,-1).T
    pt2 = matrix @ np.array([rect[1][0], rect[1][1], 1]).reshape(1,-1).T
    
    new_rect = [(int(pt1[0][0]),int(pt1[1][0])),(int(pt2[0][0]),int(pt2[1][0]))]
       
    return p, new_rect

 
car_path = glob.glob("car/*.jpg")
 
frame = cv2.imread('car/frame0020.JPG')
frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('Car_LKTracker.avi', fourcc, 20.0, (frame.shape[1],frame.shape[0]))   

rect_car = [(115,100),(340,276)]
rect = rect_car

template = frame[rect_car[0][1]:rect_car[1][1] , rect_car[0][0]:rect_car[1][0]]
p = np.zeros(6) 

for path in car_path:
    img = cv2.imread(path)
    clone = img.copy()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #p, new_rect = affineLKtracker_shiftdata(img, template, rect, p)
    
    p,new_rect=affineLKtracker_gammacorrection(img,template,rect,p)
    
    cv2.rectangle(clone, new_rect[0], new_rect[1], (0, 0, 255), 2)
    
    cv2.imshow('frame', clone)
    #out.write(clone)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()