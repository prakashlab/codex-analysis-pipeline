import numpy as np
import glob
import os	
import gcsfs
from natsort import natsorted
import time

'''
For each .npy output from cellpose, make a new .npy with a boolean mask and number of cells
'''

def main():
    # root_dir needs a trailing slash (i.e. /root/dir/)
    root_dir = "/media/prakashlab/T7/malaria-tanzina-2021/dpc/"#'gs://octopi-codex-data-processing/' #"/home/prakashlab/Documents/kmarx/pipeline/tstflat/"# 'gs://octopi-codex-data-processing/TEST_1HDcVekx4mrtl0JztCXLn9xN6GOak4AU/'#
    exp_id   = "Negative-Donor-Samples/segmentation/"
    channel =  "BF_LED_matrix_dpc" # only run segmentation on this channel
    key = "/home/prakashlab/Documents/fstack/codex-20220324-keys.json"#'/home/prakashlab/Documents/kmarx/malaria_deepzoom/deepzoom uganda 2022/uganda-2022-viewing-keys.json'
    gcs_project = 'soe-octopi'
    t0 = time.time()
    run_flatmask(root_dir, exp_id, channel, key, gcs_project)
    t1 = time.time()
    print(t1-t0)

def run_flatmask(root_dir, exp_id, channel, key, gcs_project):
    # Load remote files if necessary
    root_remote = False
    if root_dir[0:5] == 'gs://':
        root_remote = True
    fs = None
    if root_remote:
        fs = gcsfs.GCSFileSystem(project=gcs_project,token=key)

    print("Reading npy paths")
    # get all .npy
    path = root_dir + exp_id + "**/0/**.npy"
    print(path)
    if root_remote:
        allpaths = [p for p in fs.glob(path, recursive=True)]
    else:
        allpaths = [p for p in glob.iglob(path, recursive=True)]
    # remove duplicates
    nppath = np.array(natsorted(list(dict.fromkeys(allpaths))))
    print(str(len(nppath)) + " .npy available")

    placeholder = "./placeholder.npy"
    dest = root_dir + exp_id + "masks/"
    os.makedirs(dest, exist_ok=True)

    # segment one at a time - gpu bottleneck
    for idx, path in enumerate(nppath):
        print(str(idx) + ": " + path)
        if root_remote:
            # get from remote
            fs.get(path, placeholder)
            nppath = placeholder
        else:
            nppath = path
        # load the data
        reading = np.load(nppath, allow_pickle=True).item()
        masks = np.array(reading['masks'])
        count = np.max(masks)
        boolmask = (masks > 0)
        # save the data
        out_dict = {
            "count": count,
            "boolmask": boolmask
        }
        out_dict = np.array(out_dict)
        outpath = nppath.rsplit('.', -1)[0] + "_boolmask_count.npy"

        np.save(outpath, out_dict, allow_pickle=True)
        # move the .npy to remote if necessary
        if root_remote:
            # generate the local and remote path names
            rpath ="gs://" + path.rsplit(".", 1)[0] + "_boolmask_count.npy"
            fs.put(outpath, rpath)
            print(rpath)
            os.remove(outpath)
            os.remove(placeholder)

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

if __name__ == "__main__":
    main()
