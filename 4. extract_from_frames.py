import cv2
import numpy as np
import os

print(cv2.__version__)
vid_dir = 'video/'
img_dir = 'data/'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
for vid_path in os.listdir(vid_dir):
    vid_name=os.path.splitext(vid_path)[0]
    vid_path = os.path.join(vid_dir, vid_path)
    print (vid_path)
    cap = cv2.VideoCapture(vid_path)
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")
    i=0;
    while(cap.isOpened()):  
      # Capture frame-by-frame
      ret, frame = cap.read()
      if ret == True: 
        #print(vid_name)
        
        img_name=vid_name+"/"+str(i)+".png"
        img_path=img_dir+img_name
        #print(img_path)
        if not os.path.exists(img_dir+vid_name):
          os.makedirs(img_dir+vid_name)
        cv2.imwrite(img_path,frame)
        i=i+1
      else :
        break
    cap.release()