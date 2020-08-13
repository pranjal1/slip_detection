import os
import sys

import cv2
import skvideo
from skvideo import datasets, io, measure
import numpy as np
from loguru import logger
from pytube import YouTube
import matplotlib.pyplot as plt


def convert_vid_to_imgs(vid_path: str, save_path: str) -> bool:
    """
    Converts the video to sequence of images
    """
    if not os.path.isfile(vid_path):
        raise FileNotFoundError
    logger.debug(f"Found video in path {vid_path}")
    vidcap = cv2.VideoCapture(vid_path)
    success, image = vidcap.read()
    count = 0
    logger.debug(f"Saving images in path {save_path}")
    logger.debug(f"Converting...")
    while success:
        cv2.imwrite(
            os.path.join(save_path, f"frame{count}.jpg"), image
        )  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1
    vidcap.release()
    logger.debug(f"Done!")
    return True


def detect_scene_boundaries(
    file_path: str = None,
    use_edge: bool = True,
    use_luminance: bool = True,
    debug: bool = False,
) -> dict:
    """
    Detects the scene boundaries in compilation videos
    """
    if debug:
        file_path = datasets.bikes()
    elif not os.path.isfile(file_path):
        raise FileNotFoundError
    result_dct = dict()
    print(file_path)
    videodata = io.vread(file_path)
    videometadata = io.ffprobe(file_path)
    frame_rate = videometadata["video"]["@avg_frame_rate"]
    num_frames = np.int(videometadata["video"]["@nb_frames"])
    width = np.int(videometadata["video"]["@width"])
    height = np.int(videometadata["video"]["@height"])

    if use_edge:
        # using the "edge" algorithm
        logger.info("Using edge algorithm...")
        scene_edge_idx = skvideo.measure.scenedet(videodata, method="edges")
        scene_edge = np.zeros((num_frames,))
        scene_edge[scene_edge_idx] = 1
        result_dct.update({"edge_algorithm": scene_edge})

    if use_luminance:
        # using the "luminance" algorithm
        logger.info("Using luminance algorithm...")
        scene_lum_idx = skvideo.measure.scenedet(
            videodata, method="histogram", parameter1=1.0
        )
        scene_lum = np.zeros((num_frames,))
        scene_lum[scene_lum_idx] = 1
        result_dct.update({"luminance_algorithm": scene_lum})
    return result_dct


def temporal_crop():
    pass


def download_video(
    video_url: str, save_dir_path: str = "", convert_to_img: bool = False
) -> bool:
    """
    Downloads youtube video given the url.
    """
    logger.info(f"Quering for video {video_url}...")
    yt = YouTube(url=video_url)
    vid_title = yt.title
    logger.info(f"Found Video titled {vid_title}.")
    vid_save_path = os.path.join(save_dir_path, vid_title)
    os.makedirs(save_dir_path)
    logger.info(f"The video will be saved to {vid_save_path}.")
    logger.info(f"Downloading...")
    yt.streams.filter(file_extension="mp4").get_by_resolution("144p").download(
        vid_save_path
    )
    logger.info(f"Download complete!")

    if convert_to_img:
        imgs_folder = os.path.join(save_dir_path, imgs)
        os.makedirs(imgs_folder)
        logger.info(f"Converting video to sequence of images in path {imgs_folder}...")
        if not convert_vid_to_imgs(vid_save_path, imgs_folder):
            raise Exception(
                f"Error in converting video {vid_save_path} to sequence of images!"
            )
        logger.info("Conversion to image complete!")


if __name__ == "__main__":
    # download_video("http://www.youtube.com/watch?v=yhb04aZYyXY", "data/slip1", True)
    # convert_vid_to_imgs(sys.argv[1], sys.argv[2])
    print(detect_scene_boundaries(debug=True))
