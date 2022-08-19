# CODEX Pipeline

This repository contains everything necessary to process image data, from focus stacking to basic analysis.

## Repo Contents

### fstack

- `requirements-fstack.txt`
  - List of requirements to run fstack and fstack_cu. Activate your conda environment and run `pip install -r requirements-fstack.txt`
- fstack
  - `__init__.py`: this is the fstack code.
  - Python code in the pipeline/ directory can import fstack with `from fstack import fstack`
- fstack_cu
  - `__init__.py`: this is the CUDA-accelerated fstack code.
  - Python code in the pipeline/ directory can import fstack with `from fstack_cu import fstack_cu` or `from fstack_cu import fstack_cu_images`
- `fstacker.py`: perform a focus stack using local (provide path) or remote (using GCSFS) images

## Guides

### fstack cli usage

1. Prepare your images. They should all be in a single directory (local or remote) and have the format "\[filename\]\_\[focus_idx\]\_.\[extension\]" where \[filename\] is the same across the images in the series, \[focus_idx\] is the index of the image ranging from 0 to n-1 where n is the total number of images in the series. The indices must correspond to the focus distance in the image. Focus stacking can only be done if n is greater than or equal to 5.
