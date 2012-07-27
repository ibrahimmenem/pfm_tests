#!/usr/bin/env python

# espcial for lark, to match matlab experiments. 
# frame number index, and another frame rate mod (50)

import pygtk ,sys , cv2 ,os,numpy , operator
pygtk.require('2.0')
import gtk

class VideoTagger:
    # Our callback.
    # The data passed to this method is printed to stdout
    def callback(self, widget,data=None):
              #print widget.get_name() 
              if widget.get_active():
                    for item in widget.get_parent():
                         item.set_active(True)
              else:
                    for item in widget.get_parent():
                         item.set_active(False)

    # This callback quits the program
    def delete_event(self,*args):
        video_capture.release()
        video_tags_file.close()
        gtk.main_quit()
        return False
    
    def nextframe(self, *args):
         if not self.finished:
             while  operator.mod(self.Read_Frames ,DSF_rate)!=0 :
                       video_capture.grab()
                       self.Read_Frames=self.Read_Frames+1
             if   self.Read_Frames>=FRAME_COUNT:
                  print "All frames have been read!"  
                  self.finished=1	
  
             
             frame=cv2.pyrDown(video_capture.retrieve()[1])  # spacial downscale 
             image.set_from_pixbuf(gtk.gdk.pixbuf_new_from_array(numpy.array(frame), gtk.gdk.COLORSPACE_RGB, 8))

             t= float(self.Read_Frames)/FPS 
     
     
             self.record_holder.append(t)
             self.record_holder.append(self.Read_Frames) # adding frame number
             for chkbox in self.chkboxs:
                   if chkbox.get_active()==True:
                       self.record_holder.append(chkbox.get_name())

             if not self.finished:
                 line=''
                 for item in self.record_holder:
                       line=line+str(item)+','
                 video_tags_file.write(line.strip(',')+'\n')
                 print "Saving: "+line.strip(',')+'\n'
                  
                 self.record_holder=[] # empty record holder        
                 self.Read_Frames=self.Read_Frames+1

    def __init__(self):
        self.finished=0
        self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        self.window.set_title("tag video")
        self.window.connect("delete_event", self.delete_event)
        self.window.set_border_width(20)
        hbox = gtk.HBox(False, 2)
        self.window.add(hbox)
        logolist={ 'GB':['Logo_FCB','Logo_pepsi','Logo_LibraryAdvocacy','Logo_BMW','Logo_ferrari','Logo_facebook','Logo_linux','Logo_chrome','Logo_apple','Logo_DOMINOSPIZZA','Logo_fifa','Logo_Firefox'],
                        'GM':['Logo_BurgerKing','Logo_GENERALI','Logo_Lexus','Logo_Mcdonalds','Logo_Mercedes','Logo_intel','Logo_mazda','Logo_msn','logo-iberia','logo_indra','Logo_carrefour','Logo_mastercard'],
                        'GN':['Logo_Abb','Logo_Dell','Logo_Nokia','Logo_RealMadrid','Logo_Renault','Logo_aljazeeradoc','Logo_cocacola','Logo_nike','Logo_nvidia','Logo_pringles','Logo_waltdisney','logo_adidas11','logo_renfe_2425','Logo_iberdrola'],
                        'GS':['Logo_SEAT','Logo_Skype','Logo_Sony_Ericsson','Logo_Starbucks','Logo_TELEFONICA','Logo_UPM','Logo_Wikipedia','Logo_Windows','Logo_samsung','Logo_toyota','Logo_twitter','Logo_visa','Logo_xeon','logo_santander']}
        self.chkboxs=[]
        for videogroup, logogroup in logolist.items():
             vbox= gtk.VBox(False, 2)
             button=gtk.CheckButton(videogroup)
             button.connect("toggled", self.callback, videogroup) 
             button.set_name(videogroup)
             vbox.pack_start(button, True, True, 2)
             for bname in logogroup:
                  button = gtk.CheckButton(label=bname)
                  self.chkboxs.append(button)
                  #button.connect("toggled", self.callback, bname)
                  button.set_name(bname)
                  vbox.pack_start(button, True, True, 2)
             hbox.pack_start(vbox, True, True, 2)

        vbox= gtk.VBox(True, 2) 
        # Create "Quit" button
        button = gtk.Button("Quit")
        button.connect("clicked", lambda wid: gtk.main_quit())
        vbox.pack_start(button, True, True, 2)

        # Create next frame button
        button = gtk.Button("NextFrame")
        button.connect("clicked",self.nextframe)
        vbox.pack_start(button, True, True, 2)

        hbox.pack_start(vbox, True, True, 2)
        hbox.pack_start(image, True, True, 2)
        self.Read_Frames=1
        self.record_holder=[] 
        self.window.show_all()

def main():
    gtk.main()
    return 0       
    
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

if __name__ == "__main__": 
        if len(sys.argv)!=3:
             video_fn="./Videos/GB_1.MP4" 
             DSF_rate=50
        elif len(sys.argv)==3:
             video_fn = sys.argv[1]
             DSF_rate=int( sys.argv[2] )  # frequency downscale rate
        else:
             print "bad input arguments!"
             print """Usage:  
                                     tagvideo.py [videopath] [framerate]  """
             sys.exit()

        dir,file= os.path.split(video_fn)
        video_tags_file=open(dir+"/"+file.split('.')[0]+"_lark.csv",'w')
        video_capture=cv2.VideoCapture(video_fn)
        video_capture.open(video_fn) 

        frame=cv2.pyrDown(video_capture.read()[1])  # spacial downscale 

        #cv2.imshow("frame",frame)
        image = gtk.Image()

        props= ReadVideoProps(video_capture)
        FPS=int(props['FPS'])
        FRAME_COUNT=int(props['FRAME_COUNT'])
        image.set_from_pixbuf(gtk.gdk.pixbuf_new_from_array(numpy.array(frame), gtk.gdk.COLORSPACE_RGB, 8))

        VideoTagger()
        main()
