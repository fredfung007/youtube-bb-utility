#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from subprocess import check_call
from concurrent import futures
import subprocess
import youtube_dl
import socket
import os
import io
import sys
import cv2
import pandas as pd
import numpy as np

# The data sets to be downloaded
d_sets = ['yt_bb_detection_validation', 'yt_bb_detection_train']

# Column names for detection CSV files
col_names = ['youtube_id', 'timestamp_ms','class_id','class_name',
             'object_id','object_presence','xmin','xmax','ymin','ymax']

# Host location of segment lists
web_host = 'https://research.google.com/youtube-bb/'

# Help function to get the index of the element in an array the nearest to a value
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

# Print iterations progress (thanks StackOverflow)
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '█' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()

# Download and cut a clip to size
def dl_and_cut(vid, data, d_set_dir):

    # Use youtube_dl to download the video
    FNULL = open(os.devnull, 'w')
    check_call(['youtube-dl', \
      '-f','best[ext=mp4]', \
      '-o',d_set_dir+'/'+vid+'_temp.mp4', \
      'youtu.be/'+vid ], \
      stdout=FNULL,stderr=subprocess.STDOUT )

    # Verify that the video has been downloaded. Skip otherwise
    video_path = d_set_dir+'/'+vid+'_temp.mp4'
    if os.path.exists(video_path):

        # Use opencv to open the video
        capture = cv2.VideoCapture(video_path)
        fps, total_f = capture.get(5), capture.get(7)

        # Get time stamps (in seconds) for every frame in the video
        # This is necessary because some video from YouTube come at 29.99 fps,
        # other at 30fps, other at 24fps
        timestamps = [i/float(fps) for i in xrange(int(total_f))]
        labeled_timestamps = data['timestamp_ms'].values / 1000

        # Get nearest frame for every labeled timestamp in CSV file
        indexes = []
        for label in labeled_timestamps:
            frame_index = find_nearest(timestamps, label)
            indexes.append(frame_index)

        i = 0
        for index, row in data.iterrows():
            # Get the actual image corresponding to the frame
            capture.set(1,indexes[i])
            ret, image = capture.read()

            # Uncomment lines below to print bounding boxes on downloaded images
            # w, h = capture.get(3),capture.get(4)
            # x1, x2, y1, y2 = row.values[6:10]
            # x1 = int(x1*w)
            # x2 = int(x2*w)
            # y1 = int(y1*h)
            # y2 = int(y2*h)
            # cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 2)
            i += 1

            # Make the class directory if it doesn't exist yet
            class_dir = d_set_dir+str(row.values[2])
            check_call(['mkdir', '-p', class_dir])

            # Save the extracted image
            frame_path = class_dir+'/'+row.values[0]+'_'+str(row.values[1])+\
                '_'+str(row.values[2])+'_'+str(row.values[4])+'.jpg'
            cv2.imwrite(frame_path, image)
        capture.release()

    # Remove the temporary video
    os.remove(d_set_dir+'/'+vid+'_temp.mp4')
    return vid

# Parse the annotation csv file and schedule downloads and cuts
def parse_and_sched(dl_dir='videos', num_threads=4):
    """Download the entire youtube-bb data set into `dl_dir`.
    """

    # Make the download directory if it doesn't already exist
    check_call(['mkdir', '-p', dl_dir])

    # For each of the two datasets
    for d_set in d_sets:

        # Make the directory for this dataset
        d_set_dir = dl_dir+'/'+d_set+'/'
        check_call(['mkdir', '-p', d_set_dir])

        # Download & extract the annotation list
        print (d_set+': Downloading annotations...')
        check_call(['wget', web_host+d_set+'.csv.gz'])
        print (d_set+': Unzipping annotations...')
        check_call(['gzip', '-d', '-f', d_set+'.csv.gz'])

        # Parse csv data using pandas
        print (d_set+': Parsing annotations into clip data...')
        df = pd.DataFrame.from_csv(d_set+'.csv', header=None, index_col=False)
        df.columns = col_names

        # Get list of unique video files
        vids = df['youtube_id'].unique()

        # Download and cut in parallel threads giving
        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(dl_and_cut,vid,df[df['youtube_id']==vid],d_set_dir) for vid in vids]
            for i, f in enumerate(futures.as_completed(fs)):
                # Write progress to error so that it can be seen
                printProgress(i, len(vids),
                            prefix = d_set,
                            suffix = 'Done',
                            barLength = 40)

        print( d_set+': All videos downloaded' )

if __name__ == '__main__':
    # Use the directory `videos` in the current working directory by
    # default, or a directory specified on the command line.
    parse_and_sched(sys.argv[1], int(sys.argv[2]))
