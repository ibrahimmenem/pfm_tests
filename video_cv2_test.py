#!/usr/bin/env python
import sys ,os
import cv2 
import binascii
import numpy as np
from common import anorm
from time import time
#sudo apt-get install python-matplotlib
help_message = '''
USAGE: video_cv2_pythn.py [ <video file full path> <logo file full path> <64 | 128> <BF|FLANN> <FPS> <size|octave|angle|stringth > <(0->100)> <NOGFTT | GFTT> <0 | 1 |2>]

<64 | 128>                   : Descriptor length 64 or 128 
<BF|FLANN>                   : Match algorithm Brute force  
<FPS>                        : Processed frame rate
<size|octave|angle|stringth> : Select features sort key
<(0->100)>                   : Percent of used sorted features 
<NOGFTT | GFTT>              : Use or not good features to track 
< 0 | 1 | 2 >                : DownScale 0, 1 or 2 times
'''
FLANN_INDEX_KDTREE = 1  
flann_params = dict(algorithm = FLANN_INDEX_KDTREE,trees = 4)

def spacial_downscale_function(spacial_DS):
	if spacial_DS==0:
	   def SDS(gray_frame):
	       return gray_frame

	elif spacial_DS==1:
	   def SDS(gray_frame):
	       return cv2.pyrDown(gray_frame)
	       
	else:
	   def SDS(gray_frame):
	       return cv2.pyrDown(cv2.pyrDown(gray_frame))
	       
	return SDS

def find_GFTT(gray_Master_logo,kp_master_logo,kp2dsc_dict):
    #goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, useHarrisDetector[, k]]]]]) -> corners
     gftt_corners= cv2.goodFeaturesToTrack(gray_Master_logo,100, 0.04, 1.0) # (gray_Master_logo,40, 0.04, 1.0)
     selectedkps=[]
     if gftt_corners==None:
        return selectedkps
     print "all GFTT ",len(gftt_corners)
     ###for p in gftt_corners:
     ###     cv2.circle(gray_Master_logo, (int(p[0][0]),int(p[0][1])) ,7,  cv2.cv.Scalar(0, 0, 255, 0), thickness=2, lineType=4)#lineType=cv2.CV_AA          
     #selecting from kp_master_logo the keypoint near to gftt_corners
     for gp in gftt_corners:
           for kp in kp_master_logo:
                 if  (abs(kp.pt[0] - gp[0,0]) < 2) and (abs(kp.pt[1] - gp[0,1]) < 2):
                      selectedkps.append(kp)
     return selectedkps
        
#def sort_keypoints_and_descriptors(kps,kp2dsc_dict,by,percent):
#     #print kp2dsc_dict
#     # sorting the keypoints list
#     if by=='size':
#         sortedkps = sorted(kps, key=lambda item: item.size, reverse=True) # True= larger is at kps[0]
#     if by=='stringth': 
#         sortedkps = sorted(kps, key=lambda item: item.response, reverse=True)
#     if by=='octave': 
#         sortedkps = sorted(kps, key=lambda item: item.octave, reverse=True)
#     if by=='angle': 
#         sortedkps = sorted(kps, key=lambda item: item.angle, reverse=True)
#     #select the first 20%  of sorted keypoints
#     sortedkps=sortedkps[:int((percent/100)*len(sortedkps))]
#     # construct sorted desc from the kp2dsc_dict and the sorted keypoints
#     sorteddsc=np.vstack((kp2dsc_dict[sortedkps[0]],kp2dsc_dict[sortedkps[1]])) 
#     for i in range(2,len(sortedkps)):
#          sorteddsc=np.vstack((sorteddsc,kp2dsc_dict[sortedkps[i]])) 
#     #print sorteddsc
#     return list(sortedkps), sorteddsc 
     
def extract_and_select_features_from_logo(gray_Master_logo,extended,by,percent,dir,gftt):   
    surf = cv2.SURF(100,4,4,extended) # gives very large number of features
    kps, desc_master_logo = surf.detect(gray_Master_logo, None, False)
    desc_master_logo.shape = (-1, surf.descriptorSize()) 
    MSG= "Number of detected keypoints in the master logo: {0}\n".format(len(kps))
    #creating dict to map points to descs  (not using zip because it returns tuple and I want numpy.ndarray)
    kp2dsc_dict={}
    for i in range(len(kps)):
           kp2dsc_dict[kps[i]]=desc_master_logo[i,:]
    # sorting the keypoints list
    if by=='size':
        sortedkps = sorted(kps, key=lambda item: item.size, reverse=True) # True= larger is at kps[0]
    elif by=='stringth': 
        sortedkps = sorted(kps, key=lambda item: item.response, reverse=True)
    elif by=='octave': 
        sortedkps = sorted(kps, key=lambda item: item.octave, reverse=True)
    elif by=='angle': 
        sortedkps = sorted(kps, key=lambda item: item.angle, reverse=True)
    else:
         print "Unknown sort key", by 
    #select the first 20%  of sorted keypoints
    sel_kps=sortedkps[:int((float(percent)/100)*len(sortedkps))] # the selected sorted keypoints
    MSG+="Number of selected keypoints by sort: {0}\n".format(len (sel_kps))
    rem_kps=sortedkps[int((float(percent)/100)*len(sortedkps)):] #the remaining sorted keypoints
    MSG+="Number of remaining keypoints by sort: {0}\n".format(len(rem_kps)) 
    #print "after sort\n"
    for p  in kps:
          #print p.pt, p.size, p.angle, p.response, p.octave, p.class_id, '\n'
          cv2.circle(gray_Master_logo, (int(p.pt[0]),int(p.pt[1])) ,3,  cv2.cv.Scalar(0, 0, 255, 0), thickness=1, lineType=4)#lineType=cv2.CV_AA

    for p  in sel_kps:
          #print p.pt, p.size, p.angle, p.response, p.octave, p.class_id, '\n'
          cv2.circle(gray_Master_logo, (int(p.pt[0]),int(p.pt[1])) ,3,  cv2.cv.Scalar(0, 255, 0, 0), thickness=3, lineType=4)#lineType=cv2.CV_AA

    if gftt=="gftt": # find gftt from the remaining sorted keypoints 
          gftt_kps=find_GFTT(gray_Master_logo,rem_kps,kp2dsc_dict)
          MSG+="Number of added keypoints by GFTT: {0}\n".format(len(gftt_kps))
          for p  in gftt_kps:
                #print p.pt, p.size, p.angle, p.response, p.octave, p.class_id, '\n'
                cv2.circle(gray_Master_logo, (int(p.pt[0]),int(p.pt[1])) ,7,  cv2.cv.Scalar(255, 0, 0, 0), thickness=1, lineType=4)

	  sel_kps.extend(gftt_kps)

    MSG+= "Final number of selected keypoints from master logo: {0}\n".format(len(sel_kps)) 
    # construct sorted desc from the kp2dsc_dict and the sorted keypoints
    sorteddsc=np.vstack((kp2dsc_dict[sel_kps[0]],kp2dsc_dict[sel_kps[1]])) 
    for i in range(2,len(sel_kps)):
         sorteddsc=np.vstack((sorteddsc,kp2dsc_dict[sel_kps[i]])) 
  
    kp_master_logo, desc_master_logo =list(sel_kps),sorteddsc
    print MSG
    cv2.imshow("features from logo",gray_Master_logo)
    cv2.imwrite("./"+dir+"/featuresfromlogo.png",gray_Master_logo)
    return kps , desc_master_logo, MSG 
    
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


def export_features(x,y,y1,dir):
        import matplotlib.pyplot as plt
        f=plt.figure()
        plt.plot(x,y,color='green', linestyle='dashed')
        plt.title('Number of matched features per frame\n'+dir)
        plt.savefig("./"+dir+"/Matched_features.png" )
        plt.figure()
        plt.plot(x,y1,color='red', linestyle='dashed')
        plt.title('Number of detected features per frame\n'+dir)
        plt.savefig("./"+dir+"/Detected_features.png") 
        plt.figure()
        plt.plot(x,y/y1,color='red', linestyle='dashed')
        plt.title('Ratio of matched to detected features\n'+dir)
        plt.savefig("./"+dir+"/ratio.png") 
        #plt.show()

if __name__ == '__main__':
    try: 
        video_fn, logo_fn = sys.argv[1:3]  
        desclength,matchfunction, DSF_rate,logoKPsort,logoKPsortpercent,gftt,spacial_DS=[x.lower() for x in sys.argv[3:10] ]
        video_capture=cv2.VideoCapture(video_fn)
        video_capture.open(video_fn) 
        gray_Master_logo= cv2.imread(logo_fn, 0) 
        extended= (0, 1)[desclength=='128']
        match_function=(match_flann,match_bruteforce)[matchfunction=='bf']
        DSF_rate=int(DSF_rate) # frequency downscale rate
        SDS=spacial_downscale_function(int(spacial_DS)) # spacial downscale function
    except ValueError:
        print  "*** No or bad input args!\n", help_message
        sys.exit(1)     
    dir=os.path.split(video_fn)[1]+"_"+os.path.split(logo_fn)[1]+"_len:"+desclength+"_"+matchfunction+"_fps:"+str(DSF_rate)+"_sort:"+logoKPsort+"_"+logoKPsortpercent+"_"+str(gftt)+"_DS_"+str(spacial_DS)
    try: 
          os.system("rm -r "+dir)
          os.system("mkdir "+dir)
    except: os.system("mkdir "+dir)
    features_txt_file=open("./"+dir+"/features_txt_file.txt",'w')
    #os.system("rm ./"+dir+"/*.png")
    _,frame=video_capture.read()
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
    kp_master_logo, desc_master_logo, MSG = extract_and_select_features_from_logo(gray_Master_logo,extended,logoKPsort,logoKPsortpercent,dir,gftt) 
    
    surf = cv2.SURF(1000,4,4,extended)
    #cv2.startWindowThread()
    #cv2.namedWindow("test")
    t_start=time()
    matched_features=np.array([])
    sampling_times=np.array([])
    number_of_features=np.array([])
    #for j in range(FRAME_COUNT*DSF_rate/FPS):
    while True:
            for i in range(FPS/DSF_rate):
                 video_capture.grab()
                 Read_Frames=Read_Frames+1
            if Read_Frames>=FRAME_COUNT:
                  print "All frames have been read!"   
                  break
            print "Processing frame number {0} ".format(Read_Frames)
            _,frame=video_capture.retrieve()
            gray_frame=cv2.cvtColor(frame, cv2.cv.CV_RGB2GRAY) 
            ds_gray_frame=SDS(gray_frame)
            #ds_gray_frame=cv2.pyrDown(gray_frame) # first spacial downscale
            #ds_gray_frame=cv2.pyrDown(ds_gray_frame)  # secound spacial downscale 
            kp_frame, desc_frame = surf.detect(ds_gray_frame, None,False)
            try:
                 #cv2.waitKey()     
                 desc_frame.shape = (-1, surf.descriptorSize())
            except:
                 print "frame with no or insignificant features, skipped"    
                 continue  
            #m = match_flann(desc_frame, desc_master_logo)
            m = match(match_function,desc_frame, desc_master_logo)
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
    features_txt_file.write(
'\n==============================LOGO FEATURES===============================\n'
+MSG+
'\n==============================VIDEO FEATURES==============================\n'
+''.join([' %s: %s \n' % (key, value) for (key, value) in props.items()])+
'\n=============================MATCHED FEATURES=============================\n')
    for t,num_features in zip(sampling_times,matched_features):
          features_txt_file.write(str(num_features)+"\t features, matched\t @ " +str(t)+"\n")
    features_txt_file.close()
    print "Total execution time=", total_time, "sec"
    export_features(sampling_times,matched_features,number_of_features,dir)
    cv2.waitKey()     
    video_capture.release()
    
    
    