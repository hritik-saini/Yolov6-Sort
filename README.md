# YOLOv6-SORT 

Yolov6-Sort is a simple real-time tracker for Tennis-Ball.

## Introduction

**YOLOv6-SORT Uses** 
+  [tensorturtle/classy-sort-yolov5](https://github.com/tensorturtle/classy-sort-yolov5) with modifications include( Replacing Yolov5 - with Yolov6 ) 
+  [meituan/YOLOv6](https://github.com/meituan/YOLOv6)  with minor modifications (Used for training YOLOv6 on Tennis-Ball Dataset)
+  [abewley/SORT](https://github.com/abewley/sort) with minor modifications 

## Using YOLOv6-SORT

Clone this repository

```bash
git clone https://github.com/hritik-saini/Yolov6-Sort.git
cd Yolov6-Sort
```

### Install Requirements

Python 3.8 or later with all requirements.txt. To install run:

```bash
pip install -r requirements.txt
```

### Download YOLOv5 weights

```bash
./download_weights.sh
```
This script will save yolov5 weights to `yolov5/weights/` directory.

### Run Tracking

To run the tracker on your own video and view the tracked bounding boxes, run:

```bash
python classy_track.py --source /path/to/video.mp4 --view-img
```

To get a summary of arguments run:

```bash
python classy_track.py -h
```

The text results are saved to `/inference/output/` from the array above in the following format. That location in the script is also a good point to plug your own programs into.

The saved text file contains the following information:

```bash
[frame_index, x_left_top, y_left_top, x_right_bottom, y_right_bottom, object_category, u_dot, v_dot, s_dot, object_id]
```

where

+ u_dot: time derivative of x_center in pixels
+ v_dot: time derivative of y_center in pixels
+ s_dot: time derivative of scale (area of bbox) in pixels



## License

YOLOv6-SORT is released under the GPL License version 3 to promote the open use of the tracker and future improvements.
Among other things, this means that code from this repository cannot be used for closed source distributions,
and you must license any derived code as GPL3 as well.
