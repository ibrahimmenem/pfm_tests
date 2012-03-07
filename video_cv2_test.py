import sys ,os
import cv2 
import binascii
import numpy as np
from common import anorm
from time import time
#sudo apt-get install python-matplotlib
help_message = '''
USAGE: video_cv2_pythn.py [ <video file full path> <logo file full path> <64 | 128> <BF|FLANN> <FPS>]
'''
FLANN_INDEX_KDTREE = 1  
flann_params = dict(algorithm = FLANN_INDEX_KDTREE,trees = 4)

#goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, useHarrisDetector[, k]]]]]) -> corners
def sort_keypoints_and_descriptors(kps,dsc,by):
     kp_ds=zip(kps,dsc)
     if by=='size':
         sortedzip = sorted(kp_ds, key=lambda x: x[0].size, reverse=True)
     if by=='stringth': 
         sortedzip = sorted(kp_ds, key=lambda x: x[0].response, reverse=True)
     if by=='octave': 
         sortedzip = sorted(kp_ds, key=lambda x: x[0].octave, reverse=True)
     if by=='angle': 
         sortedzip = sorted(kp_ds, key=lambda x: x[0].angle, reverse=True)
     sortedkp,sorteddesc=zip(*sortedzip)
     return sortedkp,sorteddesc
     
def extract_features_from_logo(gray_Master_logo,extended):   
    surf = cv2.SURF(0,4,4,extended) # gives large numbers of features
    kp_master_logo, desc_master_logo = surf.detect(gray_Master_logo, None, False)
    desc_master_logo.shape = (-1, surf.descriptorSize()) 
    for p  in kp_master_logo:
          print p.pt, p.size, p.angle, p.response, p.octave, p.class_id, '\n'
          cv2.circle(gray_Master_logo, (int(p.pt[0]),int(p.pt[1])) ,3,  cv2.cv.Scalar(0, 0, 255, 0), thickness=1, lineType=4)#lineType=cv2.CV_AA
    print "/////////////////////////////////////////////////"
    print "/////////////////////////////////////////////////"
    print "/////////////////////////////////////////////////"
    kp_master_logo, desc_master_logo =sort_keypoints_and_descriptors(kp_master_logo,desc_master_logo,'size')
    for p  in kp_master_logo:
          print p.pt, p.size, p.angle, p.response, p.octave, p.class_id, '\n'
    #gftt_corners= cv2.goodFeaturesToTrack(gray_Master_logo,40, 0.04, 1.0)
    #print gftt_corners
    #for p in gftt_corners:
    #      cv2.circle(gray_Master_logo, (int(p[0][0]),int(p[0][1])) ,7,  cv2.cv.Scalar(0, 0, 255, 0), thickness=2, lineType=4)#lineType=cv2.CV_AA          
    cv2.imshow("features from logo",gray_Master_logo)
    
    return kp_master_logo , desc_master_logo
    
def match(match_func,desc1, desc2,):
	m=match_func(desc1,desc2)
	return m
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


def export_features(x,y,y1):
        import matplotlib.pyplot as plt
        f=plt.figure()
        plt.plot(x,y,color='green', linestyle='dashed')
        plt.title('Number of matched features per frame')
        plt.savefig("./"+dir+"/Matched_features.png" )
        plt.figure()
        plt.plot(x,y1,color='red', linestyle='dashed')
        plt.title('Number of detected features per frame')
        plt.savefig("./"+dir+"/Detected_features.png") 
        #plt.show()

if __name__ == '__main__':
    try: 
        video_fn, logo_fn, desclength,matchfunction, DS_rate= sys.argv[1:6]
        video_capture=cv2.VideoCapture(video_fn)
        video_capture.open(video_fn) 
        gray_Master_logo= cv2.imread(logo_fn, 0) 
        extended= (0, 1)[desclength=='128']
        match_function=(match_flann,match_bruteforce)[matchfunction=='BF']
        DS_rate=int(DS_rate)
    except ValueError:
        print  "*** No or bad input args!\n", help_message
        sys.exit(1)     
    dir=os.path.split(video_fn)[1]+"_"+os.path.split(logo_fn)[1]+"_len:"+desclength+"_"+matchfunction+"_fps:"+str(DS_rate)
    try: 
          os.system("rm -r "+dir)
          os.system("mkdir "+dir)
    except: os.system("mkdir "+dir)
    features_txt_file=open("./"+dir+"/features_txt_file.txt",'w')
    #os.system("rm ./"+dir+"/*.png")
    retval,frame=video_capture.read()
    props= ReadVideoProps(video_capture)
    print props
    #FRAME_WIDTH=int(props['FRAME_WIDTH'])
    #FRAME_HEIGHT=int(props['FRAME_HEIGHT'])
    FPS=int(props['FPS'])
    FRAME_COUNT=int(props['FRAME_COUNT'])
    Read_Frames=0
    #Estimated_Video_Length=float(FRAME_COUNT)/FPS
    #gray_frame=cv2.cv.CreateMatND((FRAME_WIDTH ,FRAME_HEIGHT) , cv2.CV_8UC1)# CreateImage(, cv2.cv.IPL_DEPTH_8U, 1)
    # ds_gray_frame=CreateImage((FRAME_WIDTH/2 ,FRAME_HEIGHT/2), cv2.cv.IPL_DEPTH_8U, 1)
    kp_master_logo, desc_master_logo = extract_features_from_logo(gray_Master_logo,extended) 
    print "number of features in the master logo: ", len(kp_master_logo)
    surf = cv2.SURF(1000,4,4,extended)
    #cv2.startWindowThread()
    #cv2.namedWindow("test")
    t_start=time()
    matched_features=np.array([])
    sampling_times=np.array([])
    number_of_features=np.array([])
    #for j in range(FRAME_COUNT*DS_rate/FPS):
    while True:
            for i in range(FPS/DS_rate):
                 video_capture.grab()
                 Read_Frames=Read_Frames+1
            if Read_Frames>=FRAME_COUNT:
                  print "All frames have been read!"   
                  break
            retval,frame=video_capture.retrieve()
            gray_frame=cv2.cvtColor(frame, cv2.cv.CV_RGB2GRAY) 
            ds_gray_frame=cv2.pyrDown(gray_frame)
            #ds_gray_frame=cv2.pyrDown(ds_gray_frame)
            kp_frame, desc_frame = surf.detect(ds_gray_frame, None,False)
            try:
                 #cv2.waitKey()     
                 desc_frame.shape = (-1, surf.descriptorSize())
            except:
                 print "frame with no or insignificant features, skipped"    
                 continue  
            #m = match_flann(desc_frame, desc_master_logo)
            m = match(match_flann,desc_frame, desc_master_logo)
            number_of_features=np.append(number_of_features ,[len(desc_frame)])
            matched_features=np.append(matched_features,[len(m)])
            sampling_times=np.append(sampling_times,[float(Read_Frames)/FPS ]) 
            #for p  in kp_frame:
            #      cv2.circle(ds_gray_frame, (int(p.pt[0]),int(p.pt[1])) , int(p.size),  cv2.cv.Scalar(0, 0, 255, 0), thickness=1, lineType=4)#lineType=cv2.CV_AA
            #cv2.imshow("test",ds_gray_frame)
            ch = cv2.waitKey(1)
            #if ch == 1048603: #27: puede dependerse del sistema operativo
            #      print "escaped" 
            #      break

    total_time=time()-t_start
    for t,num_features in zip(sampling_times,matched_features):
          features_txt_file.write(str(num_features)+"\t features, matched\t @ " +str(t)+"\n")
    features_txt_file.close()
    print "Total execution time=", total_time, "sec"
    export_features(sampling_times,matched_features,number_of_features)
    cv2.waitKey()     
    video_capture.release()
    
    
    