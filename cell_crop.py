import gcsfs
import imageio
import argparse
import numpy as np
import pandas as pd
import cv2
import os
import time

def main():
    # root_dir needs a trailing slash (i.e. /root/dir/)
    root_dir = "gs://octopi-codex-data/"
    exp_id   = "20220823_20x_PBMC_2/"
    dest_dir = "/home/prakashlab/Documents/kmarx/" + exp_id + "image_crop/"
    channel =  "Fluorescence_405_nm_Ex" # only crop images from this channel
    celltype_file = "./08_23_22_PBMC_Octopi_celltypes.csv"
    zstack  = 0 # select which z to crop. set to 'f' to select the focus-stacked
    key = "/home/prakashlab/Documents/fstack/codex-20220324-keys.json"
    gcs_project = 'soe-octopi'
    cell_radius = 30
    t0 = time.time()
    make_crops(root_dir, exp_id, channel, zstack, celltype_file, key, gcs_project, dest_dir, cell_radius)
    t1 = time.time()
    print(t1-t0)

def make_crops(root_dir, exp_id, channel, zstack, celltype_file, key, gcs_project, dest_dir, cell_radius):
    img_remote = False
    if root_dir[0:5] == 'gs://':
        img_remote = True
    data_remote = False
    if celltype_file[0:5] == 'gs://':
        data_remote = True
    dest_remote = False
    dest_local = dest_remote
    if dest_dir[0:5] == 'gs://':
        dest_local = "./imgs_"+exp_id
        dest_remote = True
    
    fs = None
    if img_remote or data_remote or dest_remote:
        fs = gcsfs.GCSFileSystem(project=gcs_project,token=key)
    
    # get remote data
    if data_remote:
        with fs.open(celltype_file, 'r' ) as f:
            celltype_df = pd.read_csv(celltype_file)
    else:
        with open(celltype_file, 'r' ) as f:
            celltype_df = pd.read_csv(celltype_file)
    
    for i in range(celltype_df.shape[0]):
        i_idx = int(celltype_df["i"][i])
        j_idx = int(celltype_df["j"][i])
        xpos  = int(celltype_df["x"][i])
        ypos  = int(celltype_df["y"][i])
        id = 'cycle0_2022-08-23_20-15-33.401781'
        # load image 
        filename = id + '/0/' + str(i_idx) + '_' + str(j_idx) + '_' + str(zstack) + '_' + channel + '.bmp'
        image_path = root_dir + exp_id + filename
        print(image_path)
        if img_remote:
            im = imread_gcsfs(fs,image_path)
        else:
            im = cv2.imread(image_path)
        im = np.array(im)
        xmax, ymax = im.shape

        xcrop = [int(max(0, xpos - cell_radius)), int(min(xmax, xpos + cell_radius))]
        ycrop = [int(max(0, ypos - cell_radius)), int(min(ymax, ypos + cell_radius))]

        cropped_image = im[xcrop[0]:xcrop[1], ycrop[0]:ycrop[1]]

        cell_type = celltype_df["cell_type"][i]
        filename = cell_type + str(i_idx) + '_' + str(j_idx) + '_' + str(zstack) + '_' + channel + '.bmp'
        savepath = dest_dir + exp_id + id + '/0/'
        os.makedirs(savepath, exist_ok=True)
        savepath = savepath + filename
        print(savepath)
        cv2.imwrite(savepath, cropped_image)




def imread_gcsfs(fs,file_path):
    '''
    imread_gcsfs gets the image bytes from the remote filesystem and convets it into an image
    
    Arguments:
        fs:         a GCSFS filesystem object
        file_path:  a string containing the GCSFS path to the image (e.g. 'gs://data/folder/image.bmp')
    Returns:
        I:          an image object
    
    This code has no side effects
    '''
    img_bytes = fs.cat(file_path)
    im_type = file_path.split('.')[-1]
    I = imageio.core.asarray(imageio.v2.imread(img_bytes, im_type))
    return I


if __name__ == '__main__':
    main()