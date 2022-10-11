import os
from cv2 import CALIB_CB_FAST_CHECK, waitKey
import numpy as np
import cv2
import os
import scipy.optimize

IMG_DIR = 'Calibration_Imgs'
N_ROWS = 9
N_COLS = 6
BLOCK_SIZE = 21.5 #in mm
image_paths = [os.path.join(IMG_DIR, image) for image in os.listdir(IMG_DIR)]
# print(image_paths)
images = [cv2.imread(img_path) for img_path in image_paths]

V = []
H_all_images = []
m_all_images = []
M_all_images = []
for i,img in enumerate(images):
    #get m from images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    is_found, corners = cv2.findChessboardCorners(gray,(N_ROWS,N_COLS),flags=cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK)
    cv2.drawChessboardCorners(img,(9,6),corners,is_found)
    cv2.imwrite('draw_corners_'+str(i)+'.png',img)
    corners = corners.reshape(-1,2)
    #adhering to convention in the paper
    m = corners
    m_all_images.append(np.concatenate([m,np.ones((N_COLS*N_ROWS,1))],axis=1))
    #create M
    M_X, M_Y = np.meshgrid(np.arange(BLOCK_SIZE,BLOCK_SIZE*(N_ROWS+1),BLOCK_SIZE),np.arange(0,BLOCK_SIZE*(N_COLS),BLOCK_SIZE))
    M = np.concatenate([M_X.reshape(-1,1),M_Y.reshape(-1,1)],axis=1)
    # print(M)

    #find homography
    # H, _ = cv2.findHomography(M,m)

    M_hat = np.concatenate([M,np.ones((N_COLS*N_ROWS,1))],axis=1,dtype=np.float32)
    M_all_images.append(M_hat)
    # M_hat = M_hat/M_hat.mean(axis=0)

    L_i1 = np.concatenate([M_hat, np.zeros(M_hat.shape), -1*m[:,0].reshape(-1,1)*M_hat],axis=1)
    L_i2 = np.concatenate([np.zeros(M_hat.shape), M_hat, -1*m[:,1].reshape(-1,1)*M_hat],axis=1)

    L=[]
    for i in range(M_hat.shape[0]):
        L.append(L_i1[i]) 
        L.append(L_i2[i])
    L = np.array(L)
    
    u, s, vh = np.linalg.svd(L.T @ L)
    H = vh[-1].reshape(3,3)
    H /= H[2,2]
    # H = np.concatenate((H,[[0,0,1]]),axis=0)
    # print(H)
    #verify homography
    # dst = cv2.warpPerspective(img,H,dsize=img.shape[:2])
    H_all_images.append(H)
    #get V
    # v = [v01.T, (v00-v11).T]

    def v(H,i,j):
        #hi = col vector of H
        #vij = [hi0hj0,     hi0hj1 + hi1hj0,        hi1hj1,         hi2hj0 + hi0hj2,        hi2hj1 + hi1hj2,         hi2hj2]T
        H = H.T
        v1 = H[i,0]*H[j,0]
        v2 = H[i,0]*H[j,1] + H[i,1]*H[j,0]
        v3 = H[i,1]*H[j,1]
        v4 = H[i,2]*H[j,0] + H[i,0]*H[j,2]
        v5 = H[i,2]*H[j,1] + H[i,1]*H[j,2]
        v6 = H[i,2]*H[j,2]
        return np.array([v1,v2,v3,v4,v5,v6])
        
    v01 = v(H,0,1)
    v00 = v(H,0,0)
    v11 = v(H,1,1)
    
    V.append(v01)
    V.append(v00-v11)

    # cv2.imshow('img', cv2.resize(img,(700,700)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

#get b
V = np.array(V).reshape(2*len(image_paths),6)
u, eigenVal, eigenVec = np.linalg.svd(V.T @ V)
b = eigenVec[-1]
print(b)
#get B
# B = np.array([
#      [b[0], b[1], b[3]],
#      [b[1], b[2], b[4]],
#      [b[3], b[4], b[5]] 
#     ])

#get A
#B = [B11, B12, B13]
B11 = b[0]
B12 = b[1]
B13 = b[3]
B22 = b[2]
B23 = b[4]
B33 = b[5]

py = (B12*B13 - B11*B23)/(B11*B22 - B12**2)
lmbda = B33 - ((B13**2 + py*(B12*B13 - B11*B23))/B11)
alpha = np.sqrt(lmbda/B11)
beta = np.sqrt((lmbda * B11)/ (B11*B22-B12**2))
gamma = -1 * B12 * alpha**2 * beta / lmbda
px = (gamma * py) / beta - (B13 * alpha**2) / lmbda

A = np.array([
    [alpha, gamma, px],
    [0, beta, py],
    [0,0,1]
])
print(A)

#get extrinsic val
A_inverse = np.linalg.inv(A)
R_t_all_images = []
t_all_images = []

for H in H_all_images:
    r1 = lmbda * A_inverse @ H[:,0]
    r2 = lmbda * A_inverse @ H[:,1]
    t =  lmbda * A_inverse @ H[:,2]
    r3 = lambda r1,r2: np.cross(r1, r2)

    R_t_all_images.append(np.array([r1.T,r2.T,t.T]))
    # t_all_images.append(t)

# init distortion parameters 
k1,k2=0,0

# def L(m):
#     #m = (x,y,1)
#     r = np.square(m - np.repeat(np.array([[px,py,1]]),m.shape[0],axis=0)).sum(axis=1)
#     return np.reshape((1+ k1*r**2 +k2*r**4),(-1,1))

# maximum likelihood estimation 
def L2(m, R_t_i, M):
    return np.square(m - (A @ (R_t_i @ M.T)).T).sum()

def minimization_fn(params):
    error = 0
    n_images = len(image_paths)
    for i in range(n_images):
        error += L2(m_all_images[i],R_t_all_images[i],M_hat)
    return error

params = [alpha,beta,px,py,k1,k2]
optimized = scipy.optimize.minimize(minimization_fn, params, )
print(optimized)

alpha_opt, beta_opt, px_opt, py_opt, k1_opt, k2_opt = optimized.x

opt_A = np.array([
    [alpha_opt, 0, px_opt],
    [0, beta_opt, py_opt],
    [0,0,1]
])
print(opt_A)