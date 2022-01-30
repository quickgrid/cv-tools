> **WARNING:** IT IS NOT PROPERLY TESTED AND MAY DAMAGE YOUR FILES. USE AT YOUR OWN RISK.

# vision-tools

A bunch of cli tools for deep learning and computer vision.


## Modules

#### [Face Dataset Generator](https://github.com/quickgrid/vision-tools/tree/main/face-dataset-generator)

- Extracts all faces in image with user defined padding from images.
- Crops and saves all faces in image with correct orientation around `Z` axis.
- Saves `json` file for each face with rotated and non-rotated rectangle, rotation angle.

#### [Object Dataset Generator]()

- Uses OpenCV DNN to run yolo model with `*.weight`, `classes.names` and `*.cfg`.
- Takes list of class objects to detect and extracts the image, json file.

## TODO

- [ ] Replace python looping with numba njit parallel or use cython.
- [ ] Make a setup script for install and run from terminal.
- [ ] Upload object dataset generator code.
- [ ] Use mediapipe face detector with face mesh landmarks as alternate of PCN.

## References

Look into [references.md](https://github.com/quickgrid/vision-tools/blob/main/references.md).

## License

- Unless otherwise stated the license for this repository is [MIT License](https://github.com/quickgrid/vision-tools/blob/main/LICENSE).
- [PCN](https://github.com/quickgrid/vision-tools/tree/main/face-dataset-generator/pcn) was copied from [pytorch-PCN](https://github.com/siriusdemon/pytorch-PCN). Its license can be [found here](https://github.com/quickgrid/vision-tools/blob/main/face-dataset-generator/LICENSE).
