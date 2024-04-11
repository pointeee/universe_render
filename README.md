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

