# Universe Render

This is a mini package for rendering the hydro simulation into awesome visualizations.

Required Python Packages: `numpy`, `scipy`, `numpy-quaternion`, `numba`, `mpi4py`

*`numpy-quaternion`is used in 3D rotation of the camera for interpolation of frame. You can run the without it if you don't need to interpolate the frames.

*`numba` and `mpi4py` is for acceleration. In theory you can run with only `numpy` but this is not recommened.

Required System Installation: mpi environment (for multi-process support), FFmpeg with x264 (for video making)

## Roadmap

- [x] A working prototype for rendering
- [ ] Documentation (ongoing)
- [ ] Example run
- [ ] More kernels
- [ ] GPU accelerate support

## Installation 

First, clone this repo to your machine

```shell
git clone git@github.com:pointeee/universe_render.git
```

Then, install with `pip`

```shell
cd universe_render
pip install -e .
```

This will allow you to run the code and edit it conveniently when necessary (this is common at this stage).

## Usage

Please refer the documentation and [the exmaple notebooks](./exmaples).

