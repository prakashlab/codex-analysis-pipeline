# CLI to focus-stack a series of local or remote images 
import gcsfs
import imageio
import sys
import argparse
import numpy as np
from natsort import natsorted

def main():
    error = 0
    parser = argparse.ArgumentParser(description='focus stack parameters')
    # settings for CPU vs GPU
    hardware_args = parser.add_argument_group("hardware arguments")
    hardware_args.add_argument('--use_gpu', action='store_true', help='use gpu if cuda installed')
    # settings for locating and formatting images
    img_args = parser.add_argument_group("input image arguments")
    img_args.add_argument('--key', default=[], type=str, help='path to JSON key to GCSFS server')
    img_args.add_argument('--src', default=[], type=str, help='source directory or GCSFS path containing images')
    img_args.add_argument('--dst', default=[], type=str, help='destination directory or GCSFS path to save images')
    # settings for stacking the images
    stack_args = parser.add_argument_group("stacking behavior arguments")
    img_args.add_argument('--WSize', default=9,   type=int,   help='Filter size')
    img_args.add_argument('--alpha', default=0.2, type=float, help='blending parameter')
    img_args.add_argument('--sth',   default=13,  type=int,   help='blending parameter')
    # misc
    parser.add_argument('--verbose', action='store_true', help='show information about running and settings')
    args = parser.parse_args()

    # Initialize arguments
    if len(args.src) == 0:
        print("Error: no source provided")
        error += 1
    
    if len(args.dst) == 0:
        args.dst = args.src
        if args.verbose:
            print("dst not given, set to src by default")
    
    if args.use_gpu:
        from fstack_cu import fstack_images
        if args.verbose:
            print("Using GPU")
    else:
        from fstack import fstack_images
        if args.verbose:
            print("Using CPU")

    # if there are any errors, stop
    if error > 0:
        print(str(error) + " errors detected")
        return

    # load a nparray of images
    # images shaped (imagesets, stack idx, x-brightness/blue, y-brightness/blue, x-brightness/green, y-brightness/green, x-brightness/red, y-brightness/red)
    if args.src[0:4]== 'gs://':
        type, imgs = load_imgs_remote(args.src, args.key)
    else:
        type, imgs = load_imgs_local(args.src)
    
    for set in imgs:
        im = fstack_images(set, WSize=args.WSize, alpha=args.alpha, sth=args.sth)
        # Save image
        if args.src[0:4]== 'gs://':
            imgs = load_imgs_remote(args.src, args.key)
        else:
            imgs = load_imgs_local(args.src)

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
    I = imageio.core.asarray(imageio.imread(img_bytes, im_type))
    return im_type, I