import numpy as np
import cv2
import copy
from scipy.interpolate import RectBivariateSpline
import glob as gl



video = input("type 'car', 'bolt' or 'dragon_baby' without the '' quotes ")
filename = r'../data/{}'.format(video)
imgs = gl.glob(filename+r'/img/*.jpg')
imgs.sort()
def load_bbox(filename):
    with open(filename+ '/groundtruth_rect.txt','r') as f:
        bbox = f.read()
    bbox = bbox.replace('\n', ' ')
    bbox = bbox.replace(',', ' ')
    bbox = bbox.split()
    bbox = np.array(list(int(i) for i in bbox)).reshape(-1,4)
    return bbox

bbox = load_bbox(filename)
jacobian = np.array([[1, 0], [0, 1]])
def get_bbox(frame_no):
    return bbox[frame_no]

def run_gt_bbox():
    for i,path in enumerate(imgs):
        img = cv2.imread(path,0)
        a,b,c,d = get_bbox(i)
        cv2.rectangle(img,(a,b),(a+c,b+d), 0, 1)
        cv2.imshow('sequence', img)
        cv2.waitKey(20)
    cv2.destroyAllWindows()



def LKTracker(prev_img, cur_img, rectpoints, p=np.zeros(2)):
    #threshold: 0.1 for bolt,car
    threshold = 0.2
    x1, y1, x2, y2 = rectpoints[0], rectpoints[1], rectpoints[2], rectpoints[3]
    
    grad_y, grad_x = np.gradient(cur_img)
    dp = 1
    count = 0
    
    x_range = np.arange(0, prev_img.shape[0], 1)
    y_range = np.arange(0, prev_img.shape[1], 1)

    tx_range = np.linspace(x1, x2, 53)
    ty_range = np.linspace(y1, y2, 24)
    tx_mesh, ty_mesh = np.meshgrid(tx_range, ty_range)

   
    spline = RectBivariateSpline(x_range, y_range, prev_img)

    T = spline.ev(ty_mesh, tx_mesh)
    while np.square(dp).sum() > threshold:
    
        x1_warp, y1_warp, x2_warp, y2_warp = x1 + p[0], y1 + p[1], x2 + p[0], y2 + p[1]

        wx_range = np.linspace(x1_warp, x2_warp, 53)
        wy_range = np.linspace(y1_warp, y2_warp, 24)
        wx_mesh, wy_mesh = np.meshgrid(wx_range, wy_range)

        spline1 = RectBivariateSpline(x_range, y_range, cur_img)
        spline_gx = RectBivariateSpline(x_range, y_range, grad_x) 
        spline_gy = RectBivariateSpline(x_range, y_range, grad_y)
        
        warpImg = spline1.ev(wy_mesh, wx_mesh)
        dx = spline_gx.ev(wy_mesh, wx_mesh) 
        dy = spline_gy.ev(wy_mesh, wx_mesh)
        
        error = (T - warpImg).reshape(-1, 1)    
        delI = np.vstack((dx.ravel(), dy.ravel())).T

        hessian = delI @ jacobian

        H = hessian.T @ hessian

        dp = np.linalg.inv(H) @ (hessian.T) @ error

        p[0] += dp[0, 0]
        p[1] += dp[1, 0]

    return p


x,y,length,width = get_bbox(0)
rectpoints = [x,y,x+length, y+width]
rectpoints0 = copy.deepcopy(rectpoints)
prev_img = cv2.imread(imgs[0])
# prev_img2 = cv2.GaussianBlur(prev_img,(3,3),1)
template = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
template = cv2.equalizeHist(template)/255
outfile = r'./car.avi'
# out = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'MJPG') , 10, (360,240))
cur_img = cv2.imread(imgs[1])
for i in range(2, len(imgs)):
    #40 frames for bolt, 40 or 60 for car
    if i%60 ==0:
        x,y,length,width = get_bbox(i-1)
        rectpoints = [x,y,x+length, y+width]
        rectpoints0 = copy.deepcopy(rectpoints)
        prev_img = cv2.imread(imgs[i-1])
#         prev_img2 = cv2.GaussianBlur(prev_img,(3,3),1)
        template = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        template = cv2.equalizeHist(template)/255

    
    cv2.rectangle(cur_img,\
    (int(rectpoints[0]),int(rectpoints[1])),(int(rectpoints[0])+length,int(rectpoints[1])+width),(0,255,0),1)
    
    cv2.imshow('Tracker', cur_img)
    # out.write(cur_img)

    cur_img = cv2.imread(imgs[i])
#     cur_img2 = cv2.GaussianBlur(cur_img,(3,3),1)
    warped = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
    warped = cv2.equalizeHist(warped)/255
    
    p = LKTracker(template, warped, rectpoints0)
    
    rectpoints[0] = rectpoints0[0] + p[0]
    rectpoints[1] = rectpoints0[1] + p[1]
    rectpoints[2] = rectpoints0[2] + p[0]
    rectpoints[3] = rectpoints0[3] + p[1]

    if cv2.waitKey(10) == 27:
        cv2.destroyAllWindows()
        break
        
# out.release()
cv2.destroyAllWindows()