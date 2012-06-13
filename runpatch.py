#!/usr/bin/env python
import sys, os
from time import time
t_start=time()

i=1
for logoname in os.listdir("./Logos"):
	if logoname.split(".")[1].lower()=="jpg":
              for videoname in os.listdir("./Videos"):
                   if videoname.split(".")[1].lower()=="mp4":
                        try:
                            #os.system("sudo schedtool -R -p 20 -e python video_cv2_test_patch.py ./Videos/"+videoname+"  ./Logos/"+logoname+" 128 flann 10  size  100 gftt 0")
			    os.system("python video_cv2_test_patch.py ./Videos/"+videoname+"  ./Logos/"+logoname+" 128 flann 10  size  100 gftt 0")
		            print i
                            i=i+1 
			    #if i==3:
			    #    break
                        except:
                            pass

total_time=time()-t_start
print "runs:"+str(i)+" times in "+str(total_time)

