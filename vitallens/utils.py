

import importlib.resources
import logging
import math
import numpy as np
import os
from prpy.ffmpeg.probe import probe_video
from prpy.ffmpeg.readwrite import read_video_from_path
from prpy.numpy.image import crop_slice_resize
from typing import Union, Tuple
import urllib.request
import yaml

from vitallens.constants import API_MIN_FRAMES, VIDEO_PARSE_ERROR

def load_config(filename: str) -> dict:
  """Load a yaml config file.

  Args:
    filename: The filename of the yaml config file
  Returns:
    loaded: The contents of the yaml config file
  """
  with importlib.resources.open_binary('vitallens.configs', filename) as f:
    loaded = yaml.load(f, Loader=yaml.Loader)
  return loaded

def download_file(url: str, dest: str):
  """Download a file if necessary.

  Args:
    url: The url to download the file from
    dest: The path to write the downloaded file to
  """
  if not os.path.exists(dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    logging.info("Downloading {} to {}".format(url, dest))
    urllib.request.urlretrieve(url, dest)
  else:
    logging.info("{} already exists, skipping download.".format(dest))

def probe_video_inputs(
    video: Union[np.ndarray, str],
    fps: float = None
) -> Tuple[tuple, float]:
    """Check the video inputs and probe to extract metadata.

    Args:
        video: The video to analyze. Either a np.ndarray of shape (n_frames, h, w, 3)
            with a sequence of frames in unscaled uint8 RGB format, or a path to a
            video file.
        fps: Sampling frequency of the input video. Required if type(video)==np.ndarray.

    Returns:
        video_shape: The shape of the input video as (n_frames, h, w, 3)
        fps: Sampling frequency of the input video.
    """
    ds_factor = 1  # Set default downsampling factor to 1 (adjust as needed)

    # Check that fps is correct type
    if not (fps is None or isinstance(fps, (int, float))):
        raise ValueError("fps should be a number, but got {}".format(type(fps)))
    
    if isinstance(video, str):
        if os.path.isfile(video):
            try:
                fps_, n, w_, h_, _, _, r, _ = probe_video(video)
                if fps is None: fps = fps_
                if abs(r) == 90: h = w_; w = h_
                else: h = h_; w = w_
                return (n, h, w, 3), fps, ds_factor
            except Exception as e:
                raise ValueError("Problem probing video at {}: {}".format(video, e))
        else:
            raise ValueError("No file found at {}".format(video))
    elif isinstance(video, np.ndarray):
        if fps is None:
            raise ValueError("fps must be specified for ndarray input")
        if video.dtype != np.uint8:
            raise ValueError("video.dtype should be uint8, but got {}".format(video.dtype))
        if len(video.shape) != 4 or video.shape[0] < API_MIN_FRAMES or video.shape[3] != 3:
            raise ValueError("video should have shape (n_frames [>= {}], h, w, 3), but found {}".format(API_MIN_FRAMES, video.shape))
        return video.shape, fps, ds_factor
    else:
        raise ValueError("Invalid video {}, type {}".format(video, type(video)))
def merge_faces(faces: np.ndarray) -> np.ndarray:
    """Merge face detections across frames into a single region of interest.

    Args:
        faces: The face detections as np.int64. Shape (n_frames, 4) in form (x0, y0, x1, y1).

    Returns:
        u_roi: The union of all face detections as np.int64. Shape (4,) in form (x0, y0, x1, y1).
    """
    if len(faces) == 0:
        raise ValueError("No faces provided for merging.")
    # Ensure faces have the correct shape
    if len(faces.shape) != 2 or faces.shape[1] != 4:
        raise ValueError(f"Expected faces to have shape (n_frames, 4), but got {faces.shape}")
    # Calculate the union of all faces
    x0 = np.min(faces[:, 0])
    y0 = np.min(faces[:, 1])
    x1 = np.max(faces[:, 2])
    y1 = np.max(faces[:, 3])

    return np.array([x0, y0, x1, y1], dtype=np.int64)
def check_faces(
    faces: Union[list, np.ndarray],
    inputs_shape: tuple
) -> np.ndarray:
    """Make sure the face detections are in a correct format.

    Args:
        faces: The specified faces in form [x0, y0, x1, y1]. Either
            - list/ndarray of shape (n_faces, n_frames, 4) for multiple faces detected on multiple frames
            - list/ndarray of shape (n_frames, 4) for single face detected on multiple frames
            - list/ndarray of shape (4,) for single face detected globally
            - None to assume frames already cropped to single face
        inputs_shape: The shape of the inputs.

    Returns:
        faces: The faces. np.ndarray of shape (n_faces, n_frames, 4) in form [x_0, y_0, x_1, y_1]
    """
    n_frames, h, w, _ = inputs_shape
    if faces is None:
        # Assume that each entire frame is a single face
        logging.info("No faces given - assuming that frames have been cropped to a single face")
        faces = np.tile(np.asarray([0, 0, w, h], dtype=np.int64), (n_frames, 1))[np.newaxis]  # (1, n_frames, 4)
    else:
        faces = np.asarray(faces, dtype=np.int64)
        if faces.shape[-1] != 4: 
            raise ValueError("Face detections must be in flat point form")
        if len(faces.shape) == 1:
            # Single face detection given - repeat for n_frames
            faces = np.tile(faces, (n_frames, 1))[np.newaxis]  # (1, n_frames, 4)
        elif len(faces.shape) == 2:
            # Single face detections for multiple frames given
            if faces.shape[0] != n_frames:
                raise ValueError("Assuming detections of a single face for multiple frames given, but number of frames ({}) did not match number of face detections ({})".format(
                    n_frames, faces.shape[0]))
            faces = faces[np.newaxis]
        elif len(faces.shape) == 3:
            if faces.shape[1] == 1:
                # Multiple face detections for single frame given
                faces = np.tile(faces, (1, n_frames, 1))  # (n_faces, n_frames, 4)
            else:
                # Multiple face detections for multiple frames given
                if faces.shape[1] != n_frames:
                    raise ValueError("Assuming detections of multiple faces for multiple frames given, but number of frames ({}) did not match number of detections for each face ({})".format(
                        n_frames, faces.shape[1]))
    # Check that x0 < x1 and y0 < y1 for all faces
    if not (np.all((faces[...,2] - faces[...,0]) > 0) and np.all((faces[...,3] - faces[...,1]) > 0)):
        raise ValueError("Face detections are invalid, should be in form [x0, y0, x1, y1]")
    return faces

def check_faces_in_roi(
    faces: np.ndarray,
    roi: Union[np.ndarray, tuple, list],
    percentage_required_inside_roi: tuple = (0.5, 0.5)
) -> bool:
    """Check whether all faces are sufficiently inside the ROI.

    Args:
        faces: The faces. Shape (n_faces, 4) in form (x0, y0, x1, y1)
        roi: The region of interest. Shape (4,) in form (x0, y0, x1, y1)
        percentage_required_inside_roi: Tuple (w, h) indicating what percentage
            of width/height of face is required to remain inside the ROI.

    Returns:
        out: True if all faces are sufficiently inside the ROI.
    """
    faces_w = faces[:,2] - faces[:,0]
    faces_h = faces[:,3] - faces[:,1]
    faces_inside_roi = np.logical_and(
        np.logical_and(faces[:,2] - roi[0] > percentage_required_inside_roi[0] * faces_w,
                       roi[2] - faces[:,0] > percentage_required_inside_roi[0] * faces_w),
        np.logical_and(faces[:,3] - roi[1] > percentage_required_inside_roi[1] * faces_h,
                       roi[3] - faces[:,1] > percentage_required_inside_roi[1] * faces_h))
    return np.all(faces_inside_roi)

def convert_ndarray_to_list(d: Union[dict, list, np.ndarray]):
    """Recursively convert any np.ndarray to list in nested object.

    Args:
        d: Nested object consisting of list, dict, and np.ndarray

    Returns:
        out: The same object with any np.ndarray converted to list
    """
    if isinstance(d, np.ndarray):
        return d.tolist()
    elif isinstance(d, dict):
        return {k: convert_ndarray_to_list(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_ndarray_to_list(i) for i in d]
    else:
        return d