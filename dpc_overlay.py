import numpy as np
import glob
import os
import gcsfs
import cv2
import math
import random
import imageio
import os

def main():
    # root_dir needs a trailing slash (i.e. /root/dir/)
    root_dir = "/media/prakashlab/T7/malaria-tanzina-2021/Negative-Donor-Samples/"#'gs://octopi-codex-data-processing/' #"/home/prakashlab/Documents/kmarx/pipeline/tstflat/"# 'gs://octopi-codex-data-processing/TEST_1HDcVekx4mrtl0JztCXLn9xN6GOak4AU/'#
    dest_dir = "/media/prakashlab/T7/malaria-tanzina-2021/Negative-Donor-Samples/"
    exp_id   = "" # experiment ID - needs a trailing '/'
    left_light = "BF_LED_matrix_left_half" # name of the left illumination
    right_light = "BF_LED_matrix_right_half" # set emppty to not do DPC
    fluorescence = "Fluorescence_405_nm_Ex"  # set empty to not do overlay
    zstack  = 'f' # select which z to run segmentation on. set to 'f' to select the focus-stacked
    key = '/home/prakashlab/Documents/kmarx/malaria_deepzoom/deepzoom uganda 2022/uganda-2022-viewing-keys.json'
    gcs_project = 'soe-octopi'
    ftype = 'png'
    do_dpc_overlay(root_dir, dest_dir, exp_id, left_light, right_light, fluorescence, zstack, key, gcs_project, ftype)

def do_dpc_overlay(root_dir, dest_dir, exp_id, left_light, right_light, fluorescence, zstack, key, gcs_project, ftype):
    # Load remote files if necessary
    root_remote = False
    if root_dir[0:5] == 'gs://':
        root_remote = True
    fs = None
    if root_remote:
        fs = gcsfs.GCSFileSystem(project=gcs_project,token=key)
    
    path = os.path.join(root_dir , exp_id , "**/0/**_" + zstack + "_" + left_light + '.' + ftype)
    print(path)
    if root_remote:
        allpaths = [p for p in fs.glob(path, recursive=True) if p.split('/')[-2] == '0']
    else:
        allpaths = [p for p in glob.iglob(path, recursive=True) if p.split('/')[-2] == '0']
    # remove duplicates
    lftpaths = list(dict.fromkeys(allpaths))
    lftpaths = [p.replace('//', '/') for p in lftpaths]
    print(str(len(lftpaths)) + " views")

    # get right, fluorescent image paths if able
    rhtpaths = None
    flrpaths = None
    if right_light != "":
        rhtpaths = [p.replace(left_light, right_light) for p in lftpaths]
    if fluorescence != "":
        flrpaths = [p.replace(left_light, fluorescence) for p in lftpaths]
    
    for idx in range(len(lftpaths)):
        print(lftpaths[idx])
        print(rhtpaths[idx])
        print(flrpaths[idx])
        # read the images
        rht = None
        flr = None
        if root_remote:
            lft = np.array(imread_gcsfs(fs, lftpaths[idx]), dtype=np.uint8)
            if type(rhtpaths) != type(None):
                rht = np.array(imread_gcsfs(fs, rhtpaths[idx]), dtype=np.uint8)
            if type(flrpaths) != type(None):
                flr = np.array(imread_gcsfs(fs, flrpaths[idx]), dtype=np.uint8)
        else:
            lft = np.array(cv2.imread(lftpaths[idx]), dtype=np.uint8)
            if type(rhtpaths) != type(None):
                rht = np.array(cv2.imread(rhtpaths[idx]), dtype=np.uint8)
            if type(flrpaths) != type(None):
                flr = np.array(cv2.imread(flrpaths[idx]), dtype=np.uint8)

        # generate DPC
        dpc = None
        if type(rht) != type(None):
            dpc = generate_dpc(lft, rht)
        # generate overlay
        over = None
        if type(flr) != type(None):
            if type(dpc) == type(None):
                over = overlay(lft, flr)
            else:
                over = overlay(dpc, flr)
        
        # save results:
        path = lftpaths[idx].rsplit('_', 2)[0]
        if type(over) != type(None):
            savepath = path + "_overlay." + ftype
            
            if dest_dir[0:5] == 'gs://':
                cv2.imwrite("./img." + ftype, over)
                fs.put("./img." + ftype, savepath)
                os.remove("./img." + ftype)
            else:
                os.makedirs(savepath, exist_ok=True)
                cv2.imwrite(savepath, over)
        if type(dpc) != type(None):
            savepath = path + "_dpc." + ftype
            
            if dest_dir[0:5] == 'gs://':
                cv2.imwrite("./img." + ftype, over)
                fs.put("./img." + ftype, savepath)
                os.remove("./img." + ftype)
            else:
                os.makedirs(savepath, exist_ok=True)
                cv2.imwrite(savepath, over)

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

def generate_dpc(I_1,I_2):
    I1 = I_1.astype('float')/(255.0)
    I2 = I_2.astype('float')/(255.0)
    I_dpc = np.divide(I1-I2,I1+I2)
    I_dpc = I_dpc + 0.5
    I_dpc[I_dpc<0] = 0
    I_dpc[I_dpc>1] = 1

    return I_dpc * 255

def overlay(I_dpc, I_flr):
    I_overlay = 0.64*I_flr.astype('float') + 0.36*I_dpc.astype('float')
    return I_overlay.astype('uint8')

if __name__ == "__main__":
    main()