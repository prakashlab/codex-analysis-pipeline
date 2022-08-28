# CODEX Pipeline

This repository contains everything necessary to process image data, from focus stacking to basic analysis.

## Repo Contents

### focus stack

- `requirements-fstack.txt`: List of requirements to run fstack and fstack_cu. Activate your conda environment and run `pip install -r requirements-fstack.txt`
- fstack
  - `__init__.py`: this is the fstack code.
  - Python code in the pipeline/ directory can import fstack with `from fstack import fstack`
- fstack_cu
  - `__init__.py`: this is the CUDA-accelerated fstack code.
  - Python code in the pipeline/ directory can import fstack with `from fstack_cu import fstack_cu` or `from fstack_cu import fstack_cu_images`
- `fstacker.py`: perform a focus stack using local (provide path) or remote (using GCSFS) images. This code assumes the existence of an `index.csv` file in the source directory containing cycle names.

### segmentation

- `requirements-segment.txt`: see `requirements-fstack.txt` description
- `segmenter.py`: segment nuclei using cellpose using a pretrained model. Use the cellpose gui to create the model. `segmenter.py` assumes a folder and file structure created by `fstacker.py`.

### analysis

- `requirements-analyze.txt`: see `requirements-fstack.txt` description
- `analyzer.py`: measure the size and average brightness of each cell in each channel and save as a csv. `analyzer.py` assumes a folder and file structure created by `segmenter.py`.

## Guides

### fstacker usage

#### theory of operation

Focus-stack every imageset from a codex experiment. Suppose you have the following file structure:

- `src` source directory (can be any valid string)
  - `exp_id_1` experiment ID (can be any string)
    - `index.csv` CSV with the valid cycle names (must be named "index.csv")
    - `cycle1` folder with name of fisrt cycle (can be any string as long as it matches index.csv)
      - `0` folder named "0" (must be named "0")
        - `0_0_0_Fluorescence_405_nm_Ex.bmp` bmp image with this name format. The first digit represents the i coordinate, the second digit represents the j coordinate, the third represents the z height, and the rest of the filename represnts the channel wavelength
        - `0_0_0_Fluorescence_488_nm_Ex.bmp` bmp with same coordinates as above but with a different channel
        - more BMPs
        - `6_7_5_Fluorescence_488_nm_Ex.bmp` for our example, suppose i ranges from 0 to 6, j ranges from 0 to 7, and z index ranges from 0 to 5
  - `exp_id_2` another experiment (can have any number)
    - `0`
      - identical structure to `exp_id_1`

For each experiment ID, for each channel `Fluorescence_NNN_nm_Ex`, and for each i index `i` and j index `j` in the range, `fstacker.py` generates an image called "`i`_`j`\_f_Fluorescence_`NNN`\_nm_Ex.png" image (with different values for `i`, `j`, and `NNN` for each image stacked) and saves it to either the `src` directory or a different directory of your choosing.

#### set parameters

there are many parameters to set which images get focus stacked and where to save them. here's the complete list of the parameters and what they mean:

- `CLI`: set to `True` or `False`. Keep it set to false, the command line interface development is lagging behind the script development.
- `use_gpu`: set to `True` or `False`. Focus stacking can be accelerated with the GPU. Depending on the dataset and computer, it might be faster to set it to `False` and use the CPU; try running with either option and compare focus stack times.
- `prefix`: string. If you have an index.csv, leave this string empty. If you don't have an index.csv with the names of the cycles to analyze, you can select which cycles to run by prefix. For example, if you have three cycles with names, `cycle_good1`, `cycle_also_good`, and `cycle_bad` and you only want to run focus stacking on the two good datasets, you can set `prefix="cy"` and rename `cycle_bad` to `_cycle_bad` so it will be excluded.
- `key`: string. If you are connecting to a Google Cloud File Storage, set this to the local path to the authentication token .json. If you are running segmentation locally, this doesn't matter.
- `gcs_project`: string. Set this to the Google Cloud Storage project name if you are connecting to GCS. Otherwise, it doesn't matter.
- `src`: string. path to the folder that contains the experiment folder. Can be a GCS locator (e.g.`gs://octopi-data`) or a local path (e.g. `/home/user/Documents/src`). Note - must NOT have a trailing slash (`/`)
- `dst`: string. path to folder to save data. If left blank, the images will be stored in the same directory as `src`. `fstacker.py` will recreate the source folder structure in this folder.
- `exp`: list of strings. List of experiment IDs. In the example above, this would be `["exp_id_1", "exp_id_2"]`
- `cha`: list of strings. List of channel names. In the example above, this would be `["Fluorescence_405_nm_Ex", "Fluorescence_488_nm_Ex"]` but it should be left as-is.
- `typ`: string. Filetype of the images to read. Leave as-is.
- `colors`: Dictionary of lists. Leave as-is.
- `remove_background`: boolean. Set `True` to run a white-tophat filter. This will increase runtime considerably.
- `invert_contrast`: boolean. Set `True` to invert the image.
- `shift_registration`: boolean. Set `True` to ensure images stay aligned across cycles.
- `subtract_background`: boolean. Set `True` to subtract the minimum brightess off from the entire image.
- `use_color`: boolean. Set `True` when processing RGB images (untested).
- `imin`, `imax`, `jmin`, and `jmax`: integers. The range of i and j values to process. Generally, they should be the actual values for the experiment
- `kmin` and `kmax`: integers. The range of z index values to image stack. If (kmax+1-kmin) is less than or equal to 4, focus stacking will not run and the midpoint k index will be used. Set `kmin`=`kmax` if there's only one valid z index to bypass focus stacking.
- `crop_start` and `crop_end`: integers. Crop the image; x and y coordinates less than or greater than `crop_start` or `crop_end` respectively will be cropped out.
- `WSize`: integer. Must be odd. Size of blurring kernel for focus stacking. 9 is a good value.
- `alpha`: float. Must be between 0 and 1. Parameter for blending z stacks. 0.2 is a good value.
- `sth`: integer. Parameter for blending z stacks. 13 is a good value.
- `verbose`: boolean. Set `True` to print extra details during operation (good for determining whether GPU stacking is faster than CPU)

#### run it

1. to use fstacker as a script, set `CLI = TRUE` in `main()` in `fstacker.py` and set the constants. Then, run the file
2. to use the command line interface, set `CLI = TRUE` in `main()` in `fstacker.py`. Navigate to the directory `fstacker.py` is in, activate the conda environment if necessary, and run `python -m fstacker --help` to see all the flags and how to set them

### segmenter usage

if you are having trouble installing cellpose, try uninstalling all other python packages that use QTpy (e.g. cv2), install cellpose, then reinstall everthing you uninstalled.

1. to use segmenter as a script, set the constants and run the file. Note that it takes a pretrained model as a parameter; you can change it to one of the built-in pretrained models or set a path to a custom pretrained model
2. cellpose already has a CLI; `segmenter.py` itself does not have a CLI

### analyzer usage

1. to use analyzer as a script, set the constants and run the file
2. there currently is no CLI version
