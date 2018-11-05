"""
Example (runs YOLO on a frame every 20 seconds for 5 min vids):
    python plugins/PyTorch-YOLOv3/video_extract.py 
        --videos_folder data/inputs/videos_512x288/videos_5min 
        --frames_folder data/inputs/videos_512x288/frames_1fps 
        --output_folder data/inputs/videos_512x288/yolo80_3fpm
        --max_num_frames 15
        --frame_skip 20 
"""

from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

parser = argparse.ArgumentParser()
parser.add_argument('--videos_folder', type=str, help='path to videos to get names for batches')
parser.add_argument('--frames_folder', type=str, help='path to dataset of frames')
parser.add_argument('--output_folder', type=str, help='where to store raw, supp0.8 and supp0.5')

parser.add_argument('--max_num_frames', type=int, default=sys.maxsize, help='max num frames to use in output')
parser.add_argument('--frame_skip', type=int, default=1, help='max num frames to use for output')
parser.add_argument('--frame_offset', type=int, default=0, help='starting frame to use for output')

parser.add_argument('--config_path', type=str, default='plugins/PyTorch-YOLOv3/config/yolov3.cfg', help='path to model config')
parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=512, help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs('output', exist_ok=True)

if opt.batch_size != 1:
    raise Exception("Not supported.")

# Set up model
model = Darknet(opt.config_path, img_size=opt.img_size)
model.load_weights(opt.weights_path)

if cuda:
    model.cuda()

model.eval() # Set in evaluation mode

video_filenames = os.listdir(opt.videos_folder)
video_ids = [os.path.splitext(name)[0] for name in video_filenames]

dataloader = DataLoader(VideoFrameImageFolder(
                            opt.frames_folder, 
                            img_size=opt.img_size, 
                            max_num_frames=opt.max_num_frames,
                            frame_skip=opt.frame_skip,
                            frame_offset=opt.frame_offset,
                        ), 
                        batch_size=1, shuffle=False, num_workers=opt.n_cpu)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

print (f'\nPerforming object detection and saving to files in {opt.output_folder}:')
prev_time = time.time()
for batch_i, (video_id, _, batch_of_input_imgs) in enumerate(dataloader):
    # Already batched by data loader (so torch batch size is 1, but there is a batch here).
    input_imgs = torch.squeeze(batch_of_input_imgs, dim=0)
    video_id = video_id[0]

    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))    

    # Get detections
    with torch.no_grad():
        detections_raw = model(input_imgs)
        detections_point5 = non_max_suppression(detections_raw, 80, 0.5, opt.nms_thres, fixed_num_preds=16)
        detections_point8 = non_max_suppression(detections_raw, 80, 0.8, opt.nms_thres, fixed_num_preds=8)
        detections = detections_point8

        print (f"Saving the video id {video_id}")
        # torch.save(detections_raw, os.path.join(f"{opt.output_folder}/raw", f'{video_id}.pt'))
        torch.save(torch.stack(detections_point5), os.path.join(f"{opt.output_folder}/supp0.5", f'{video_id}.pt'))
        torch.save(torch.stack(detections_point8), os.path.join(f"{opt.output_folder}/supp0.8", f'{video_id}.pt'))

    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print (f'\t+ Batch {batch_i}, Video ID {video_id} Inference Time: {inference_time}')


