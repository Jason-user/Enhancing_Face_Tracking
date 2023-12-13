import cv2
import numpy as np
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--images_path', help='Path to images', type=str)
parser.add_argument('--output_path', help='Path to output', type=str)
parser.add_argument('--filename', help='The video you are going to process', type=str)
args = parser.parse_args()

image_path = args.images_path
output_path = args.output_path
frameSize = (1920, 1080)

out = cv2.VideoWriter(output_path + '/' + args.filename + '.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20, frameSize)
filename_list = os.listdir(image_path)
filename_list.sort()

for filename in filename_list:
    # breakpoint()
    img = cv2.imread(os.path.join(image_path, filename))
    img = cv2.resize(img, frameSize, interpolation=cv2.INTER_AREA)
    out.write(img)

out.release()