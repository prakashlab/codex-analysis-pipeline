import glob
import numpy as np
import cv2
import re
from natsort import natsorted
import pandas as pd

# Cycle indices are 0-11, we can choose a subset of the cycles to analyze
start_idx = 0 #2
end_idx   = 4 #11
# 4 channels
n_ch      = 4
# How many pixels around the mask to expand
expansion = 7
# filename
out = "meanbright_" + str(expansion) + ".csv"

# root_dir needs a trailing slash (i.e. /root/dir/)
root_dir = 'gs://octopi-codex-data-processing/TEST_1HDcVekx4mrtl0JztCXLn9xN6GOak4AU/'#"/home/prakashlab/Documents/kmarx/pipeline/test/"
exp_id   = "20220601_20x_75mm/"
channel =  "Fluorescence_405_nm_Ex"
key = '/home/prakashlab/Documents/fstack/codex-20220324-keys.json'
gcs_project = 'soe-octopi'

print("Reading .npy paths")
allpaths = [path for path in glob.iglob(root_dir + exp_id  + "**/**.npy", recursive=True)]
# remove duplicates
allpaths = list(dict.fromkeys(allpaths))
allpaths.sort()
allpaths = np.array(allpaths)

# get cycle from image paths
c = np.array([int(i.split('/')[-1].split('_')[0]) for i in allpaths])

# only consider the earliest cycle
npypaths = allpaths[c == start_idx]

# repeat to get png paths
print("Reading .png paths")
allpaths = [path for path in glob.iglob(root_dir + exp_id  + "**/**.png", recursive=True)]
# remove duplicates
allpaths = list(dict.fromkeys(allpaths))
allpaths.sort()
allpaths = np.array(allpaths)
c = np.array([int(i.split('/')[-1].split('_')[0]) for i in allpaths])
pngpaths = allpaths[(c >= start_idx) * (c <= end_idx)]

for i in npypaths:
    print(i.split('/')[-1])
for i in pngpaths:
    print(i.split('/')[-1])

# make a dataframe
header = ['i', 'j', 'x', 'y', 'sz_msk', 'sz_nuc']
for cy in range(start_idx, end_idx + 1,1):
    for ch in range(0, n_ch):
        header.append(str(cy) + "_" + str(ch))

# # process one at a time
offset = 0
npypaths = npypaths[offset:]
for idx, path in enumerate(npypaths):
    idx += offset
    # make a fresh df
    df = pd.DataFrame(columns=header)
    #   1 row for each cell
    #   1 column for each cycle x channel
    print("Loading mask " + path)
    digits =  [int(i) for i in re.split('/|_', path) if i.isdigit()]
    ch = digits[-1]
    cy = digits[-2]
    j  = digits[-3]
    i  = digits[-4]
    pattern = '\/' + str(i) + '\/' + str(j) + '\/\d+_\d+\.png'
    imgpath = natsorted([path for path in pngpaths if re.search(pattern, path)])
    # preload each image
    imgs = np.array([cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in imgpath])

    # get the mask
    reading = np.load(path, allow_pickle=True).item()
    masks = reading['masks']
    # expand the mask (capture more than just the nucleus)
    kernel = np.zeros((expansion,expansion),np.uint8)
    kernel = cv2.circle(kernel, (int(expansion/2), int(expansion/2)), int(expansion/2), (255,255,255), -1)
    dilation = cv2.dilate(masks,kernel,iterations = 1)
    
    # for each cell, calculate the mean pixel value
    for l in tqdm(range(np.max(masks))):
        # Each cell in the mask gets its own row in the CSV
        # mask an individual cell
        cell_mask = dilation == (l+1)
        # create a row for holding cell data
        row = [str(i), str(j)]
        # find the coordinates of the cell
        contours, hierarchy = cv2.findContours(np.array(cell_mask, dtype="uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        M = cv2.moments(contours[0])
        # avoid divide-by-zero errors
        try:
            x = round(M['m10'] / M['m00'])
        except:
            x = -1
        try:
            y =  round(M['m01'] / M['m00'])
        except:
            y = -1
        coord = [x, y]
        row = row + coord
        # get mean brightness
        brightness = np.zeros(len(imgpath))
        # for each image find the avg brightness around that cell
        for m, im in enumerate(imgs):
            avg = np.mean(im[cell_mask])
            brightness[m]   = avg
        # write the row
        sz = [str(np.sum(cell_mask)), str(np.sum(masks==(l+1)))]
        brightness = [str(b) for b in brightness]
        row = row + sz + brightness
        df.loc[len(df.index)] = row
    # append to csv
    df.to_csv(savepath, mode='a')