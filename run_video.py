import argparse
import logging
import time
import os
import shutil
import datetime
import json
import psutil

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger("TfPoseEstimator-Video")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tf-pose-estimation Video")
    parser.add_argument("--video", type=str, default="")
    parser.add_argument(
        "--resolution",
        type=str,
        default="432x368",
        help="network input resolution. default=432x368",
    )
    parser.add_argument(
        "--resize-out-ratio",
        type=float,
        default=2.0,
        help="if provided, resize heatmaps before they are post-processed. default=1.0 (without tracking lines)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet_thin",
        help="cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small",
    )
    parser.add_argument(
        "--show-process",
        type=bool,
        default=False,
        help="for debug purpose, if enabled, speed for inference is dropped.",
    )
    parser.add_argument(
        "--showBG", type=bool, default=True, help="False to show skeleton only."
    )
    args = parser.parse_args()

    logger.debug("initialization %s : %s" % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    cap = cv2.VideoCapture(args.video)

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    oriFilename = args.video.split("/")[-1]
    exportDirPath = f"detection/exports/video_{oriFilename}_{formatted_datetime}"
    # logger.info('ori filename: %s' % oriFilename)

    if not os.path.exists(f"{exportDirPath}"):
        os.makedirs(f"{exportDirPath}")
    else:
        shutil.rmtree(f"{exportDirPath}/")
        os.makedirs(f"{exportDirPath}/")

    if cap.isOpened() is False:
        print("Error opening video stream or file")

    fps_time = 0
    summary_report = []
    frame_report = []
    total_time = 0
    frame_id = 0

    start_time = time.time()
    while cap.isOpened():
        ret_val, image = cap.read()

        try:
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        except Exception as e:
            logger.error("Error: %s" % e)
            break
        
        if not args.showBG:
            image = np.zeros(image.shape)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        ## Calculate the fps
        current_time = time.time()
        fps = round(1.0 / (current_time - fps_time), 2)
        
        ## Access the CPU usage
        current_cpu_load = psutil.cpu_percent()

        cv2.putText(
            image,
            "FPS: %f" % fps,
            (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.imshow("tf-pose-estimation result", image)

        frame_report.append(
            {"frame_id": frame_id, "frame_time": current_time, "fps": fps, "cpu_load": current_cpu_load}
        )
        
        frame_id += 1
        fps_time = time.time()

        cv2.imwrite(f"{exportDirPath}/frame_{frame_id}.jpg", image)

        if cv2.waitKey(1) == 27:
            break
        

    total_time = time.time() - start_time
    if len(frame_report) > 0:
        avg_fps = sum([item["fps"] for item in frame_report]) / len(frame_report)
        avg_cpu_load = sum([item["cpu_load"] for item in frame_report]) / len(frame_report)
        
        summary_report.append(
            {
                "datetime": formatted_datetime,
                "filename": oriFilename,
                "model": args.model,
                "resolution": args.resolution,
                "total_time": total_time,
                "avg_fps": avg_fps,
                "avg_cpu_load": avg_cpu_load,
                "frame_report": frame_report,
            }
        )
    
    with open(f"{exportDirPath}/summary_report.json", "w") as file:
        json.dump(summary_report, file, indent=4)

    cv2.destroyAllWindows()
logger.debug("finished+")
