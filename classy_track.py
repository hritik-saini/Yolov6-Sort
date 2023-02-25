# packages
import argparse
import os
import cv2
import time
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

# yolov6
from yolov6.layers.common import DetectBackend
from yolov6.data.datasets import LoadData
from yolov6.utils.nms import non_max_suppression

# sort
from ClassySortYolov6.sort.sort import Sort
from ClassySortYolov6.utils import check_img_size, precess_image, CalcFPS, rescale, plot_box_and_label, generate_colors, \
    draw_text, model_switch


def detect(opt):
    out, source, weights, view_img, save_txt, imgsz, save_img, sort_max_age, sort_min_hits, sort_iou_thresh, conf_thres, iou_thres = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.save_img, opt.sort_max_age, opt.sort_min_hits, opt.sort_iou_thresh, opt.conf_thres, opt.iou_thres

    # Initialize SORT
    sort_tracker = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)

    # Directory and CUDA settings for yolov5
    device = torch.device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    class_names = 'tennis-ball'
    classes = None
    agnostic_nms = None
    max_det = 1000
    model = DetectBackend(weights, device=device)
    stride = model.stride

    ## use in inference for yolov6
    model_switch(model.model, imgsz)
    model.model.float()

    # Set DataLoader
    dataset = LoadData(source)

    # Run inference
    fps_calculator = CalcFPS()
    save_path, vid_writer, windows = None, None, []
    for img_src, img_path, vid_cap in tqdm(dataset):
        img, img_src = precess_image(img_src, imgsz, stride, half)
        img = img.to(device)
        if len(img.shape) == 3:
            img = img[None]
            # expand for batch dim
        t1 = time.time()
        result = model(img)
        det = non_max_suppression(result, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
        print(det)
        t2 = time.time()
        img_ori = img_src.copy()
        img_ori_sort = img_src.copy()
        dets_to_sort = np.empty((0, 6))  # initial input for sort

        ## Creating the required directory
        dir_yolov6 = Path(out + '/yolov6_output')
        if os.path.exists(dir_yolov6):
            pass
        else:
            os.mkdir(dir_yolov6)
        dir_sort = Path(out + '/sort_output')
        if os.path.exists(dir_sort):
            pass
        else:
            os.mkdir(dir_sort)

        if len(det):
            det[:, :4] = rescale(img.shape[2:], det[:, :4], img_src.shape).round()

            ## plotting original detection by yolov6 and creating array for passing to sort
            for *xyxy, conf, detclass in reversed(det):
                x1, y1, x2, y2 = xyxy
                if save_img:
                    class_num = 0  # integer class
                    dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))
                    plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, class_names,
                                       color=generate_colors(class_num, True))

            img_src = np.asarray(img_ori)

        # FPS counter
        fps_calculator.update(1.0 / (t2 - t1))
        avg_fps = fps_calculator.accumulate()

        ### If you want to save the YOLOv6 inference results only un-comment the below lines and comment-out the Sort algorithm part
        ## For yolov6 inference results only

        if dataset.type == 'video':
            draw_text(
                img_src,
                f"FPS: {avg_fps:0.1f}",
                pos=(20, 20),
                font_scale=1.0,
                text_color=(204, 85, 17),
                text_color_bg=(255, 255, 255),
                font_thickness=2,
            )

        if view_img:
            if img_path not in windows:
                windows.append(img_path)
                cv2.namedWindow(str(img_path),
                                cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(img_path), 1496, 800)
            cv2.imshow(str(img_path), img_src)
            cv2.waitKey(1)  # 1 millisecond

        # Save results (image with detections)

        if save_img:
            if dataset.type == 'image':
                cv2.imwrite(save_path, img_src)
            else:  # 'video' or 'stream'
                if save_path != dir_yolov6:  # new video
                    save_path = dir_yolov6
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, img_ori.shape[1], img_ori.shape[0]
                    vid_writer = cv2.VideoWriter(str(save_path)+"\output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(img_src)

        ### Sort algorithm start here

        # passing yolov6 detections to sort
        # tracked_dets = sort_tracker.update(dets_to_sort)
        #
        # if len(tracked_dets) > 0:
        #     bbox_xyxy = tracked_dets[:, :4]
        #     plot_box_and_label(img_ori_sort, max(round(sum(img_ori_sort.shape) / 2 * 0.003), 2), bbox_xyxy[0],
        #                        class_names,
        #                        color=generate_colors(1, True))
        #
        # img0 = np.asarray(img_ori_sort)
        #
        # if dataset.type == 'video':
        #     draw_text(
        #         img0,
        #         f"FPS: {avg_fps:0.1f}",
        #         pos=(20, 20),
        #         font_scale=1.0,
        #         text_color=(204, 85, 17),
        #         text_color_bg=(255, 255, 255),
        #         font_thickness=2,
        #     )
        #
        # if view_img:
        #     cv2.namedWindow(str(dir_sort),
        #                     cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
        #     cv2.resizeWindow(str(dir_sort), 1496, 800)
        #     cv2.imshow(str(dir_sort), img0)
        #     cv2.waitKey(1)  # 1 millisecond
        #
        # if save_img:
        #     if dataset.type == 'image':
        #         cv2.imwrite(save_path, img0)
        #     else:  # 'video' or 'stream'
        #         if save_path != dir_sort:  # new video
        #             save_path = dir_sort
        #             if vid_cap:  # video
        #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
        #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #             else:  # stream
        #                 fps, w, h = 30, img_ori.shape[1], img_ori.shape[0]
        #             vid_writer = cv2.VideoWriter(str(save_path) + "\output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps,
        #                                          (w, h))
        #         vid_writer.write(img0)

    vid_writer.release()
    cv2.destroyAllWindows()
    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov6/weights/Yolov6best500complete.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='yolov6/weights/VID_20221227_152211.mp4', help='source')
    parser.add_argument('--output', type=str, default='output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1600,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', default='True',
                        help='display results')
    parser.add_argument('--save-img', action='store_true', default='True',
                        help='save video file to output folder (disable for speed)')
    parser.add_argument('--save-txt', action='store_true', default='True',
                        help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[i for i in range(80)], help='filter by class')  # 80 classes in COCO dataset
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')

    # SORT params
    parser.add_argument('--sort-max-age', type=int, default=3,
                        help='keep track of object even if object is occluded or not detected in n frames')
    parser.add_argument('--sort-min-hits', type=int, default=0,
                        help='start tracking only after n number of objects detected')
    parser.add_argument('--sort-iou-thresh', type=float, default=0,
                        help='intersection-over-union threshold between two frames for association')

    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        detect(args)
