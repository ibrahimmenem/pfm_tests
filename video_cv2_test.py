import cv2 
import binascii
import numpy as np
from common import anorm

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing

flann_params = dict(algorithm = FLANN_INDEX_KDTREE,
                    trees = 4)

def match_flann(desc1, desc2, r_threshold = 0.6):
    flann = cv2.flann_Index(desc2, flann_params)
    idx2, dist = flann.knnSearch(desc1, 2, params = {}) # bug: need to provide empty dict
    mask = dist[:,0] / dist[:,1] < r_threshold
    idx1 = np.arange(len(desc1))
    pairs = np.int32( zip(idx1, idx2[:,0]) )
    return pairs[mask]
    

def match_bruteforce(desc1, desc2, r_threshold = 0.75):
    res = []
    for i in xrange(len(desc1)):
        dist = anorm( desc2 - desc1[i] )
        n1, n2 = dist.argsort()[:2]
        r = dist[n1] / dist[n2]
        if r < r_threshold:
            res.append((i, n1))
    return np.array(res)

def ReadVideoProps(video_capture):
        pro={"POS_MSEC":cv2.cv.CV_CAP_PROP_POS_MSEC,"POS_FRAMES":cv2.cv.CV_CAP_PROP_POS_FRAMES,"AVI_RATIO":cv2.cv.CV_CAP_PROP_POS_AVI_RATIO,
             "FRAME_WIDTH": cv2.cv.CV_CAP_PROP_FRAME_WIDTH,"FRAME_HEIGHT":cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,"FPS":cv2.cv.CV_CAP_PROP_FPS,
             "FOURCC":cv2.cv.CV_CAP_PROP_FOURCC,"FRAME_COUNT":cv2.cv.CV_CAP_PROP_FRAME_COUNT}
        props=pro.copy()
        for p in pro.keys():
              if p!="FOURCC":
                   props[p]= video_capture.get( pro[p]) 
              else:
                  try:
                       FCC=hex(int(video_capture.get( pro[p]) ))
                       props[p]= binascii.unhexlify(FCC[2:4])+binascii.unhexlify(FCC[4:6])+binascii.unhexlify(FCC[6:8])+binascii.unhexlify(FCC[8:10])   
                  except:
                       print "Bad Fourcc"
        return props

video_capture=cv2.VideoCapture()
video_capture.open("/home/ibra/Mourinho1.mp4") # Mourinho1.mp4
gray_Master_logo= cv2.imread("/home/ibra/bwin.jpg", 0) 


retval,frame=video_capture.read()
props= ReadVideoProps(video_capture)
print props
FRAME_WIDTH=int(props['FRAME_WIDTH'])
FRAME_HEIGHT=int(props['FRAME_HEIGHT'])
FPS=int(props['FPS'])
FRAME_COUNT=int(props['FRAME_COUNT'])
Read_Frames=0
DS_rate=2# (1 to FPS) frame per secound
#Estimated_Video_Length=float(FRAME_COUNT)/FPS

#gray_frame=cv2.cv.CreateMatND((FRAME_WIDTH ,FRAME_HEIGHT) , cv2.CV_8UC1)# CreateImage(, cv2.cv.IPL_DEPTH_8U, 1)
# ds_gray_frame=CreateImage((FRAME_WIDTH/2 ,FRAME_HEIGHT/2), cv2.cv.IPL_DEPTH_8U, 1)

surf = cv2.SURF(400,4,4,0)
kp_master_logo, desc_master_logo = surf.detect(gray_Master_logo, None, False)
desc_master_logo.shape = (-1, surf.descriptorSize()) 
surf = cv2.SURF(1000)
#cv2.startWindowThread()
cv2.namedWindow("test")

#for j in range(FRAME_COUNT*DS_rate/FPS):
while True:
        for i in range(FPS/DS_rate):
             video_capture.grab()
             Read_Frames=Read_Frames+1
        if Read_Frames>=FRAME_COUNT:
              print "All frames have been read!"   
              break
        retval,frame=video_capture.retrieve()
        #if retval==False:
        #     print "All frames have been read!"   
        #     break
        gray_frame=cv2.cvtColor(frame, cv2.cv.CV_RGB2GRAY) 
        ds_gray_frame=cv2.pyrDown(gray_frame)
        #ds_gray_frame=cv2.pyrDown(ds_gray_frame)
        kp_frame, desc_frame = surf.detect(ds_gray_frame, None, False)
        desc_frame.shape = (-1, surf.descriptorSize())
        #m = match_bruteforce(desc_frame, desc_master_logo)
        m = match_flann(desc_frame, desc_master_logo)
        print len(m) , "@" , float(Read_Frames)/FPS
        #for p  in kp_frame:
        #      cv2.circle(ds_gray_frame, (int(p.pt[0]),int(p.pt[1])) , p.size,  cv2.cv.Scalar(0, 0, 255, 0), thickness=1, lineType=4)#lineType=cv2.CV_AA
        #cv2.imshow("test",ds_gray_frame)
        ch = cv2.waitKey(10)
        if ch == 1048603: #27: puede dependerse del sistema operativo
              print "escaped" 
              break

cv2.waitKey()     
video_capture.release()


