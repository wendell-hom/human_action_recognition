#!/usr/bin/env python

import argparse
import os
from multiprocessing import Pool
import h5py
import numpy as np
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--video_dir', help='path to where the videos are located', required=True)
parser.add_argument('--output_dir', help='path where the hdf5 files will be saved', required=True)
parser.add_argument('--jobs', help='Number of different jobs to split the task into', required=True)
args = parser.parse_args()


##
## Each training example should be 16x112x112x3
##
## Then there is also the batch dimension
## (None, 16, 112, 112, 3)
##

from PIL import Image, ImageSequence
import cv2

# Use same pixel height and width from Du Tran et-el 2015 paper
pixel_height = 171
pixel_width = 128

#import vidaug.augmentors as va

def get_frames(filename):
    frames = []
    
    vs = cv2.VideoCapture(filename)

    (W, H) = (None, None)

    count = 0

    all_predictions = []

    while True:
        
        (grabbed, frame) = vs.read()
        
        if not grabbed:
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (pixel_height, pixel_width)).astype("uint8")
      
        frames.append(frame)        

    vs.release()
    cv2.destroyAllWindows()
    
    return frames



def create_h5py_file(filename, video_path, category_name):
    
    if (os.path.exists(filename)):
        print(f"{filename} already exists, skipping...")
        return
    else:
        print(f"Creating hdf5 file for {category_name}")
    
    filenames = os.listdir(video_path)
    
    video_to_idx = dict()
    idx_to_video = dict()
        
    count = 0    
    
    with h5py.File(filename, 'w',) as f:
        
        
        X = f.create_group("X")
        X.attrs['category_name'] = category_name
        
        m = f.create_group("fn_to_idx")
        n = f.create_group("idx_to_fn")
        
        for filename in filenames:

            fullname = os.path.join(video_path, filename)
            frames = get_frames(fullname)
            frames = np.array(frames)
            
            dset = X.create_dataset(str(count), data=frames, compression="gzip", dtype='uint8', compression_opts=4 )
            m.create_dataset(filename, data=str(count))
            n.create_dataset(str(count), data=filename)
            
            count += 1


def run_process(job_id):
    num_categories = len(categories)

    jobs_per_id = num_categories // jobs
    start = job_id * jobs_per_id
    end = start + jobs_per_id

    for idx in range(start, end):
        if idx < num_categories:
            category = categories[idx]
            filename = os.path.join(output_dir, category + ".hdf5")
            video_path = os.path.join(video_dir, category)
            print("job ", job_id, f"create_h5py_file {filename} {video_path} {category}")
            create_h5py_file(filename, video_path, category)


jobs = int(args.jobs)
video_dir = args.video_dir
output_dir = args.output_dir

all_categories = os.listdir(video_dir)
existing_files = os.listdir(output_dir)

categories = []    

for category in all_categories:
    filename = os.path.join(output_dir, category + ".hdf5")
    if (os.path.exists(filename)):
        print("Skipping ", filename)
    else:
        categories.append(category) 


print("Total number of categories: ", len(categories))

processes = list(range(jobs+1))

pool = Pool(processes=jobs+1)
pool.map(run_process, processes)


