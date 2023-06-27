# CLI/script to focus-stack a series of local or remote images 
import gcsfs
import imageio
import argparse
import numpy as np
import pandas as pd
import cv2
from itertools import product
from skimage.morphology import white_tophat
from skimage.morphology import disk
from skimage.registration import phase_cross_correlation
from skimage.transform import warp, AffineTransform
import os
debugging = False
import time

def main():
    prefix = ""     # if index.csv DNE, use prefix, else keep empty
    key = 'path/to/key/'
    gcs_project = 'project-name'
    src = "gs://source-bucket-or-local-path/"
    dst = "gs://dest-bucket-or-local-path/"
    exp = ['experiment_id_1/', 'experiment_id_2/']
    cha = ['Fluorescence_405_nm_Ex', 'BF_LED_matrix_full', 'BF_LED_matrix_left_half', 'BF_LED_matrix_low_NA', 'BF_LED_matrix_right_half']
    typ = "bmp"
    colors = {'0':[255,255,255],'1':[255,200,0],'2':[30,200,30],'3':[0,0,255]} # BRG
    remove_background = False
    invert_contrast = False
    shift_registration = True
    subtract_background = False
    use_color = False
    imin = 0    # view positions
    imax = 9
    jmin = 0
    jmax = 9
    kmin = 0 
    kmax = 0
    cmin = 1
    cmax = 10
    crop_start = 0 # crop settings
    crop_end = 3000
    verbose = True
    
    perform_stack(colors, prefix, key, gcs_project, src, exp, cha, dst, typ, imin, imax, jmin, jmax, kmin, kmax, cmin, cmax, crop_start, crop_end, remove_background, subtract_background, invert_contrast, shift_registration, use_color, verbose)
    return
    
def perform_stack(colors, prefix, key, gcs_project, src, exp, cha, dst, typ, imin, imax, jmin, jmax, kmin, kmax, cmin, cmax, crop_start, crop_end, remove_background, subtract_background, invert_contrast, shift_registration, use_color, verbose):
    os.makedirs(dst, exist_ok = True)
    t0 = time.time()
    a = crop_end - crop_start
    # Initialize arguments
    error = 0
    # verify source is given
    if len(src) == 0:
        print("Error: no source provided")
        error += 1
    # verify experiment ID is given
    if len(exp) == 0:
        print("Error: no experiment ID provided")
        error += 1
    # verify channel is given
    if len(cha) == 0:
        print("Error: no channel name provided")
        error += 1
    # verify file type is given
    if len(typ) == 0:
        print("Error: no file type provided")
        error += 1
    # check for destination
    if len(dst) == 0:
        dst = src
        if verbose:
            print("dst not given, set to src by default")
    # check if remote
    is_remote = False
    if src[0:5] == 'gs://':
        is_remote = True

    if is_remote and len(gcs_project) == 0:
        print("Remote source but no project given")
        error += 1

    if is_remote and len(key) == 0:
        print("Remote source but no key ")
        error += 1
        
    # if there are any errors, stop
    if error > 0:
        print(str(error) + " errors detected")
        return

    # Initialize the remote filesystem
    fs = None
    if is_remote:
        fs = gcsfs.GCSFileSystem(project=gcs_project,token=key)

    # store total # of imgs
    with open(dst + "log.txt", 'w') as f:
        f.write(str(len(exp) * (imax - imin + 1) * (jmax - jmin + 1) * (kmax - kmin + 1) * (cmax - cmin + 1)))
        f.write(' images\n')
    # perform fstack for each experiment and for each channel and for each i,j
    for exp_i in exp:
        # load index.csv for each top-level experiment index
        df = None
        path = src + exp_i + 'index.csv'
        try:
            if is_remote:
                with fs.open(path, 'r' ) as f:
                    df = pd.read_csv(f)
            else:
                with open( path, 'r' ) as f:
                    df = pd.read_csv(f)
        except:
            print(path + " cannot be opened")
             # exit this loop
        if verbose and len(prefix)==0:
            print(path + " opened")
            n = df.shape[0] # n is the number of cycles
            print("n cycles = " + str(n))
            for i in range(n):
                print(df.loc[i, 'Acquisition_ID'])
        if len(prefix) > 0:
            if is_remote:
                if prefix == '*':
                    loc = [a.split('/')[-1] for a in fs.ls(src + exp_i)]
                else:
                    loc = [a.split('/')[-1] for a in fs.ls(src + exp_i) if a.split('/')[-1][0:len(prefix)] == prefix ]
            else:  
                if prefix == '*':
                    loc = [a.split('/')[-1] for a in os.listdir(src  + exp_i)]
                else:
                    loc = [a.split('/')[-1] for a in os.listdir(src  + exp_i) if a.split('/')[-1][0:len(prefix)] == prefix ]
            print(loc)

        for i, j in product(range(imin, imax+1), range(jmin, jmax+1)):
            if debugging and (i > imin + 2 or j > jmin + 2):
                break
            for c in range(cmin, cmax+1):
                print('c = ' + str(c))
                if debugging and c >= cmin+4:
                    break
                try:
                    if len(prefix) > 0:
                        id = loc[c - cmin]
                    else:
                        id = df.loc[c, 'Acquisition_ID']
                except:
                    break
                    
                if verbose:
                    print(id)  
                for l in range(len(cha)):
                    if use_color:
                        color = colors[str(l)]
                        if l == 0 and invert_contrast:
                            color = [0,0,0]
                    
                    channel = cha[l]
                    if verbose:
                        print(channel)

                    k = int((kmax+1-kmin)/2)
                    
                    filename = id + '/0/' + str(i) + '_' + str(j) + '_' + str(k+kmin) + '_' + channel + '.' + typ
                    target = src + exp_i + filename
                    print(target)
                    try:
                        if is_remote:
                            I = imread_gcsfs(fs, target)
                        else:
                            I = cv2.imread(target)
                    except:
                        # Log missing data
                        with open(dst + "log.txt", 'a') as f:
                            f.write(filename)
                            f.write(str(time.time() - t0))
                            f.write("\n")

                        print("Data missing")
                        I = np.zeros((a,a))
                    # crop the image
                    I = I[crop_start:crop_end,crop_start:crop_end]
                    if not use_color:
                        if len(I.shape)==3:
                            I = np.squeeze(I[:,:,0])

                    if remove_background:
                        selem = disk(30) 
                        I = white_tophat(I,selem)
                        
                    if subtract_background:
                        I = I - np.amin(I)

                    # registration across channels
                    if shift_registration:
                        # take the first channel of the first cycle as reference
                        if c == cmin:
                            if l == 0:
                                I0 = I
                        else:
                            if l == 0:
                                # compare the first channel of later cycles to the first channel of the first cycle
                                try:
                                    shift, __, __ = phase_cross_correlation(I, I0, upsample_factor = 5)
                                except:
                                    shift = [0,0]
                                    print("Phase cross correlation failed")
                                if verbose:
                                    print(shift)
                                # create the transform
                                transform = AffineTransform(translation=(shift[1],shift[0]))
                                I = warp(I, transform)
                            else:
                                # apply shift to all channels
                                I = warp(I, transform)
                    if use_color:
                        for m in range(3):
                            if invert_contrast:
                                I[:,:,m] = 255 - I*(1-color[m]/255.0)
                            else:
                                I[:,:,m] = I*color[m]/255.0
                    else:
                        if invert_contrast:
                            I = 255 - I
                    
                    # save images 
                    fname =  str(i) + '_' + str(j) + '_f_' + channel + '.png'
                    savepath = dst + exp_i + id + '/0/'
                    print(savepath+fname)
                    if dst[0:5] == 'gs://':
                        cv2.imwrite(fname, I)
                        fs.put(fname, savepath+fname)
                        os.remove(fname)
                    else:
                        try:
                            os.makedirs(savepath)
                        except:
                            pass
                        cv2.imwrite(savepath+fname, I)
        with open(dst + "log.txt", 'a') as f:
            f.write(str(time.time() - t0))
            f.write("\n")

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
