import os
from this import d
from cv2 import CALIB_CB_FAST_CHECK, waitKey
import numpy as np
import cv2
import os

IMG_DIR = 'Calibration_Imgs'
N_ROWS = 9
N_COLS = 6
BLOCK_SIZE = 21.5 #in mm
image_paths = [os.path.join(IMG_DIR, image) for image in os.listdir(IMG_DIR)]
images = [cv2.imread(img_path) for img_path in image_paths]

#init V matrix
V = []
#init Homography list containing H of all images
H_all_images = []
#create M
M_X, M_Y = np.meshgrid(np.arange(BLOCK_SIZE,BLOCK_SIZE*(N_ROWS+1),BLOCK_SIZE),np.arange(BLOCK_SIZE,BLOCK_SIZE*(N_COLS+1),BLOCK_SIZE))
M = np.concatenate([M_X.reshape(-1,1),M_Y.reshape(-1,1)],axis=1)

for img in images:
    #get m from images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    is_found, corners = cv2.findChessboardCorners(gray,(N_ROWS,N_COLS),flags=cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK)
    cv2.drawChessboardCorners(img,(9,6),corners,is_found)
    #adhering to convention in the paper
    # m = np.concatenate([corners, np.ones((corners.shape[0],1))],axis=1)
    m = corners

    #find homography
    H, mask = cv2.findHomography(m,M) 
    #verify homography
    # dst = cv2.warpPerspective(img,H, dsize=img.shape[:2])

    #get V
    # vij = [hi1hj1, hi1hj2 + hi2hj1, hi2hj2, hi3hj1 + hi1hj3, hi3hj2 + hi2hj3, hi3hj3]T
    # v = [v12.T, (v11-v22).T]
    v12 = np.array([H[0,0]*H[1,0], H[0,0]*H[1,1]+H[0,1]*H[1,0],H[0,1]*H[1,1], H[0,2]*H[1,0]+H[0,0]*H[1,2], H[0,2]*H[1,1]+H[0,1]*H[1,2], H[0,2]*H[1,2]])
    v11 = np.array([H[0,0]*H[0,0], H[0,0]*H[0,1]+H[0,1]*H[0,0],H[0,1]*H[0,1], H[0,2]*H[0,0]+H[0,0]*H[0,2], H[0,2]*H[0,1]+H[0,1]*H[0,2], H[0,2]*H[0,2]])
    v22 = np.array([H[1,0]*H[1,0], H[1,0]*H[1,1]+H[1,1]*H[1,0],H[1,1]*H[1,1], H[1,2]*H[1,0]+H[1,0]*H[1,2], H[1,2]*H[1,1]+H[1,1]*H[1,2], H[1,2]*H[1,2]])     
    v = [v12, v11-v22]
    
    V.append(v)
    H_all_images.append(H)
    # cv2.imshow('img', cv2.resize(dst,(700,700)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

#get b
V = np.array(V).reshape(-1,6)
u, eigenVal, eigenVec = np.linalg.svd(V)
b = eigenVec[-1]
print(b)
#get B
B = np.array([
     [b[0], b[1], b[3]],
     [b[1], b[2], b[4]],
     [b[3], b[4], b[5]] 
    ])
print(B)

#get A
py = (B[0,1]*B[0,2] - B[0,0]*B[1,2])/(B[0,0]*B[1,1] - B[0,1]**2)
lmbda = B[2,2] - B[0,2]**2 + py*(B[0,1]*B[0,2] - B[0,0]*B[1,2])/B[0,0]
alpha = np.sqrt(lmbda/B[0,0])
beta = np.sqrt(lmbda * B[0,0]/ (B[0,0]*B[1,1]-B[0,1]**2))
gamma = - B[0,1] * alpha**2 * beta / lmbda
px = gamma * py / beta - B[0,2] * alpha**2 / lmbda

A = np.array([
    [alpha, gamma, px],
    [0, beta, py],
    [0,0,1]
])
print(A)

#get extrinsic val
A_inverse = np.linalg.inv(A)
R_t_all_images = []

for H in H_all_images:
    r1 = lmbda * A_inverse * H[0]
    r2 = lmbda * A_inverse * H[1]
    t = lmbda * A_inverse * H[3]
    r3 = np.cross(r1, r2)

# init distortion parameters 
k = [0,0]

# maximum likelihood estimation
