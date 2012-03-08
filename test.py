import numpy as np


class kp:

    def __init__(self,x,y):
         self.x=x
         self.y=y


if __name__ == '__main__':
    hh=np.ones((4,10))
    dsc=np.ndarray(hh.shape,dtype=float,buffer=hh)
    dsc[1,:]=2
    dsc[2,:]=3
    dsc[3,:]=4
    print dsc
    kps= [kp(4,2), kp(3,4),kp(2,6) ,kp(1,3)]

    kp2dsc_dict={}
    for i in range(len(kps)):
           kp2dsc_dict[kps[i]]=dsc[i,:]
    #print kp2dsc_dict
  

    sorteddsc=np.vstack((kp2dsc_dict[kps[0]],kp2dsc_dict[kps[1]])) 
    for i in range(2,len(kps)):
          sorteddsc=np.vstack((sorteddsc,kp2dsc_dict[kps[i]])) 
    print sorteddsc

"""	

    l2=[4,3,2,1]
    l1= [kp(4,2), kp(3,4),kp(2,6) ,kp(1,3)]
    print type(l1)
    print type(l2)
    ziped=zip(l1,l2)
    for i in range(4): 
           print l1[i].x, l1[i].y
    print "______________"
    #print ziped
    l2,l1=zip(*sorted(ziped, key=lambda item: item[0].x )) #, reverse=True
    l2=list(l2)
    l1=list(l1)
    for i in range(4): 
           print l2[i].x, l2[i].y
    print type(l1)
    print type(l2)
"""