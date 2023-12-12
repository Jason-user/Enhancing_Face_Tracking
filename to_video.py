import cv2
import numpy as np
import glob
import os

image_path = '/scratch2/users/jason890425/face_tracking/output/cqqozdaojd/images'
frameSize = (1920, 1080)

out = cv2.VideoWriter('/scratch2/users/jason890425/face_tracking/output/cqqozdaojd/video/' + 'cqqozdaojd' + '.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20, frameSize)
filename_list = os.listdir(image_path)
filename_list.sort()

for filename in filename_list:
    # breakpoint()
    img = cv2.imread(os.path.join(image_path, filename))
    img = cv2.resize(img, frameSize, interpolation=cv2.INTER_AREA)
    out.write(img)

out.release()