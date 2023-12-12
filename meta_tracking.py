'''
Put the video and csv_file (generated from ego4d_face_tracking.py) together (Within same folder).
Change filename (the video you are going to process), then run: python meta_tracking.py
'''

import cv2
import csv
import argparse
import glob
import torch
import math
import os
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torchvision.transforms as T
from accelerate import Accelerator
from facexlib.detection import init_detection_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from facexlib.recognition.recognition import resnet_face18
from sklearn.cluster import AgglomerativeClustering

filename = 'crybjsrcff'
old_csv_video_path = './nora_data'

max_frame = None # number of processing frames, set to `None` to process the entire video
video_path = old_csv_video_path + '/' + filename + '.mp4'
csv_path = old_csv_video_path + '/' + filename
csv_output_path = old_csv_video_path + '/' + filename  ### Directly replace the old one

face_helper = FaceRestoreHelper(upscale_factor=1, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50', save_ext='png')
face_helper.clean_all()
det_net = init_detection_model('retinaface_resnet50', half=False)

def cosin_metric(x1, x2):
    x1 = x1
    x2 = x2.transpose()
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

### Store all different box_id
frame_attrs = {}
total_id = []
with open(f'{csv_path}.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for attr in reader:
        frame_id = float(attr[0])
        face = {
            'id': float(attr[5]),
            'bbox': [float(a) for a in attr[1:5]],
            'lm': [np.array([[float(attr[7])-float(attr[1]), float(attr[8])-float(attr[2])],
                            [float(attr[9])-float(attr[1]), float(attr[10])-float(attr[2])],
                            [float(attr[11])-float(attr[1]), float(attr[12])-float(attr[2])],
                            [float(attr[13])-float(attr[1]), float(attr[14])-float(attr[2])],
                            [float(attr[15])-float(attr[1]), float(attr[16])-float(attr[2])]])]
        }
        if frame_id in frame_attrs:
            frame_attrs[frame_id].append(face)
        else:
            frame_attrs[frame_id] = [face]

        if face['id'] not in total_id:
            total_id.append(face['id'])

### Create space and store faces for each box_id
for i in total_id:
    globals()['room' + str(int(i))] = []

rec_net = nn.DataParallel(resnet_face18(False))
rec_net.load_state_dict(torch.load('./resnet18_110.pth'))
rec_net = rec_net.module
rec_net.cuda()
rec_net.eval()

### Process each frames
pbar = tqdm(total=int(max(frame_attrs.keys())))
cap = cv2.VideoCapture(video_path)
frame_idd = 0
count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if max_frame and frame_idd >= max_frame:
        break
    if not ret:
        break
    if frame_idd not in frame_attrs:
        pbar.update()
        frame_idd += 1
        continue

    img = frame
    img_id = int(f'{frame_idd:04}.png'.split('.')[0])
    different_faces_id = []
    for face in frame_attrs[frame_idd]:
        bboxes = face['bbox'] #x1, y1, x2, y2
        box_id = face['id']
        landmarks = face['lm']
        if box_id not in different_faces_id:
            # breakpoint()
            different_faces_id.append(box_id)
        bboxes[0] -= 50
        bboxes[1] -= 50
        bboxes[2] += 50
        bboxes[3] += 50
        for i in range(4):
            if bboxes[i] < 0:
                bboxes[i] = 0
        
        img = frame
        img = img[int(bboxes[1]):int(bboxes[3]), int(bboxes[0]):int(bboxes[2]), :]
        face_helper.read_image(img)

        # face_helper.get_face_landmarks_5(only_center_face=True)
        image = face_helper.align_warp_face(face_landmark=landmarks)
        # image = face_helper.align_warp_face()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128,128), interpolation=cv2.INTER_CUBIC)

        img = np.dstack((img, np.fliplr(img)))
        img = img.transpose((2, 0, 1))
        img = img[:, np.newaxis, :, :]
        img = img.astype(np.float32, copy=False)
        img -= 127.5
        img /= 127.5

        img = torch.from_numpy(img)
        with torch.no_grad():
            feature = rec_net(img.cuda())
            feature = feature.mean(0).cpu()

        locals()['room'+str(int(box_id))].append(feature)

    if len(different_faces_id) > 1:
        locals()['Store_different_faces_id'+str(int(count))] = different_faces_id
        count += 1
    
    pbar.update()
    frame_idd = frame_idd + 1
cap.release()
pbar.close()

### Calculate average of faces for each box_id
face_representation = []

for box_id in total_id:
    face_representation.append(torch.stack(locals()['room'+str(int(box_id))]).mean(0))

face_representation_array = torch.stack(face_representation).numpy()
### Calculate average of each representation again
agg = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0).fit(face_representation_array)

new_id = agg.labels_
total_id_new = []
total_idd = []
for x in range(len(total_id)):
    total_idd.append(float(new_id[x]))
    if total_idd[x] not in total_id_new:
        total_id_new.append(total_idd[x])

for y in total_id_new:
    globals()['new_room' + str(int(y))] = []

for z in range(len(total_idd)):
    locals()['new_room'+str(int(total_idd[z]))].append(face_representation_array[z])

face_representation_new = []
for z in range(len(total_id_new)):
    feature_new = np.add.reduce(locals()['new_room'+str(int(total_id_new[z]))]) / len(locals()['new_room'+str(int(total_id_new[z]))])
    face_representation_new.append(feature_new)

### Calculate similarity
similarity_map_new = np.zeros((len(total_id_new), len(total_id_new)))
for x in range(len(total_id_new)):
    for y in range(len(total_id_new)):
        sim = cosin_metric(face_representation_new[x], face_representation_new[y])
        similarity_map_new[x][y] = sim

### Modify csv file(by clustering)
rrr = csv.reader(open(csv_path + '.csv'))
lines = list(rrr)

agg = AgglomerativeClustering(n_clusters=None, distance_threshold=0.1).fit(similarity_map_new)
# agg = AgglomerativeClustering(n_clusters=None, distance_threshold=0.01).fit(similarity_map_new)
new_id = agg.labels_
new_id += 10000

for x in range(len(total_id)):
    for y in range(len(lines)-1):
        if lines[y+1][5] == str(total_id[x]):
            lines[y+1][5] = str(total_idd[x])

for x in range(len(new_id)):
    for y in range(len(lines)-1):
        if lines[y+1][5] == str(total_id_new[x]):
            if (lines[y+1][0] == lines[y][0]) and (lines[y][5] == str(new_id[x])):
                lines[y+1][5] = str(new_id[x]+50)
            else:
                lines[y+1][5] = str(new_id[x])
        
www = csv.writer(open(csv_output_path + '.csv', 'w'))
www.writerows(lines)
