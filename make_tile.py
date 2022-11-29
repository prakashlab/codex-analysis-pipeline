
import glob
import cv2
import numpy as np
import os
from tqdm import trange

# settings
ax0 = 3000 # read from the image
ay0 = 50
ax = ax0
ay = ay0
padding_x = ax - ax0
padding_y = ay - ay0
nx = 1



# go through the images
subfolders = glob.glob("imgs_crop_idx/*")
subfolders = [f for f in subfolders if '.' not in f]
fname = "tiles2/"
os.makedirs(fname, exist_ok=True)
for folder in tqdm(subfolders):
    # initialize
    tiled_image = np.array([])
    counter = 0
    i = 0
    print(folder)
    celltype = folder.split('/')[1]
    files = glob.glob(folder + "/*.bmp")
    files = [f for f in files if ".bmp" in f]

    with open(fname + celltype + ".csv" , 'w') as f:
        f.write("index,x,y\n")
    
    for file in files:
        print(file)

        indices = file.split('/')[-1]
        indices = indices.split('.')[0]
        indices = indices.split('_')
        line =indices[0] + "," + indices[1] + "," + indices[2] + "\n" 
        print(line)
        with open(fname + celltype + ".csv" , 'a') as f:
            f.write(line)

        I = cv2.imread(file)
        tiled_image = np.vstack([tiled_image,I]) if tiled_image.size else I
    
    #save image
    cv2.imwrite(fname + celltype + '.bmp',tiled_image)