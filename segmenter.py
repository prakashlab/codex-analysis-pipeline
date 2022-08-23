# note - cellpose only reads images on local storage
import numpy as np
from cellpose import models
from cellpose import models, io
import glob


# root_dir needs a trailing slash (i.e. /root/dir/)
root_dir = "/home/prakashlab/Documents/kmarx/pipeline/test/"
exp_id   = "20220601_20x_75mm/"
channel =  "Fluorescence_405_nm_Ex"
use_gpu = True
model = '/home/prakashlab/Documents/images/cloud_training/models/cellpose_trained2.380017'
channels = [0,0] # grayscale only

print("Reading image paths")
# filter - only look for specified channel
print(root_dir + exp_id + channel + "/**" + channel + '.png')
allpaths = [path for path in glob.iglob(root_dir + exp_id + channel + "/**" + channel + '.png', recursive=True)]
# remove duplicates
imgpaths = list(dict.fromkeys(allpaths))
imgpaths.sort()
imgpaths = np.array(imgpaths)
print("Reading images")
# get cycle, i, j from image paths
c = np.array([int(i.split('/')[-1].split('_')[0]) for i in imgpaths])
i = np.array([int(i.split('/')[-1].split('_')[1]) for i in imgpaths])
j = np.array([int(i.split('/')[-1].split('_')[2]) for i in imgpaths])
# only consider the earliest cycle
targetpaths = imgpaths[c == np.min(c)]
# load images
imgs = [io.imread(path) for path in targetpaths]

print("Starting cellpose")
# start cellpose
model = models.CellposeModel(gpu=use_gpu, pretrained_model=model)
print("Starting segmentation")

# segment one at a time - gpu bottleneck
for idx, im in enumerate(imgs):
    print(idx)
    imlist  = [im]
    masks, flows, styles = model.eval(imlist, diameter=None, channels=channels)
    diams = 0
    io.masks_flows_to_seg(imlist, masks, flows, diams, [imgpaths[idx]], channels)