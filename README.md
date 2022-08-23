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
- `fstacker.py`: perform a focus stack using local (provide path) or remote (using GCSFS) images. This code assumes the existence of an `index.csv` file in the source directory containing cycle names.
- `segmenter.py`: segment nuclei using cellpose using a pretrained model

## Guides

### fstacker usage

1. to use fstacker as a script, set `CLI = TRUE` in `main()` in `fstacker.py` and set the constants. Then, run the file.
2. to use the command line interface, set `CLI = TRUE` in `main()` in `fstacker.py`. Navigate to the directory `fstacker.py` is in, activate the conda environment if necessary, and run `python -m fstacker --help` to see all the flags and how to set them.
