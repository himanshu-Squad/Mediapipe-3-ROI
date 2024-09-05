# Copyright (c) 2024 Rouast Labs
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import itertools
import logging
import math
import numpy as np
import os
from prpy.numpy.signal import interpolate_vals
import sys
from typing import Tuple
import mediapipe as mp
import cv2
#if sys.version_info >= (3, 9):
  #from importlib.resources import files
#else:
  #from importlib_resources import files

#from vitallens.utils import parse_video_inputs

INPUT_SIZE = (240, 320)
MAX_SCAN_FRAMES = 60

class FaceDetector:
  def __init__(
      self,
      max_faces: int,
      fs: float,
      score_threshold: float,
      iou_threshold: float):
    """Initialise the face detector.

    Args:
      max_faces: The maximum number of faces to detect.
      fs: Frequency [Hz] at which faces should be detected. Detections are
        linearly interpolated for remaining frames.
      score_threshold: Face detection score threshold.
      iou_threshold: Face detection iou threshold.
    """
    self.iou_threshold = iou_threshold
    self.score_threshold = score_threshold
    self.max_faces = max_faces
    self.fs = fs
    self.face_detection= mp.solutions.face_detection.FaceDetection(                       # New face Detector - Mediapipe
      model_selection=1,
      min_detection_confidence=score_threshold)
    
  def __call__(
      self,
      inputs: Tuple[np.ndarray, str],
      inputs_shape: Tuple[tuple, float],
      fps: float
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference.

    Args:
      inputs: The video to analyze. Either a np.ndarray of shape (n_frames, h, w, 3)
        with a sequence of frames in unscaled uint8 RGB format, or a path to a video file.
      inputs_shape: The shape of the input video as (n_frames, h, w, 3)
      fps: Sampling frequency of the input video.
    Returns:
      boxes: Detected face boxes in relative flat point form (n_frames, n_faces, 4)
      info: Tuple (idx, scanned, scan_found_face, interp_valid, confidence) (n_frames, n_faces, 5)
    """
    # Check if inputs is a string (file path)
    if isinstance(inputs, str):
        
        # Load video frames directly using OpenCV
        cap = cv2.VideoCapture(inputs)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB as MediaPipe uses RGB images
            # Implement error handling .......
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        inputs = np.array(frames)
        cap.release()

    # Now `inputs` should be a numpy array, and it's safe to check its shape
    print(f"Inputs type: {type(inputs)}") 
    print(f"Inputs shape: {inputs.shape}") 
    # Determine number of batches
    n_frames = inputs_shape[0]
    n_batches = math.ceil((n_frames / (fps / self.fs)) / MAX_SCAN_FRAMES)
    if n_batches > 1:
      logging.info("Running face detection in {} batches...".format(n_batches))
    # Determine frame offsets for batches
    offsets_lengths = [(i[0], len(i)) for i in np.array_split(np.arange(n_frames), n_batches)]
    # Process in batches
    results = [self.scan_batch(inputs=inputs, batch=i, n_batches=n_batches, start=int(s), end=int(s+l), fps=fps) for i, (s, l) in enumerate(offsets_lengths)]
    boxes = np.concatenate([r[0] for r in results], axis=0)
    classes = np.concatenate([r[1] for r in results], axis=0)
    scan_idxs = np.concatenate([r[2] for r in results], axis=0)
    scan_every = int(np.max(np.diff(scan_idxs)))
    n_frames_scan = boxes.shape[0]    
    # Add print statements to debug
    print(f"Number of frames processed: {inputs.shape[0]}")
    print(f"Number of face detections: {boxes.shape[0]}")
    print(f"Shape of boxes: {boxes.shape}")
    print(f"Shape of classes: {classes.shape}")
    print(f"Shape of idxs: {scan_idxs.shape}")
    # Ensure the number of detections matches the number of frames
    if boxes.shape[0] != inputs.shape[0]:
      raise ValueError(f"Number of detections ({boxes.shape[0]}) does not match number of frames ({inputs.shape[0]})")
    # check if any faces found
    if boxes.shape[1]==0:
      logging.warning("No faces found")
      return [],[]    
    # Assort info: idx, scanned, scan_found_face, confidence
    idxs = np.repeat(scan_idxs[:,np.newaxis],boxes.shape[1],axis=1)[...,np.newaxis]  # Frame index  and confidence score is added at last
    scanned = np.ones((n_frames_scan, boxes.shape[1], 1), dtype=np.int32) # Scanned =1 if not scanned =0
    scan_found_face = np.where(classes[...,1:2] < self.score_threshold, np.zeros([n_frames_scan, boxes.shape[1], 1], dtype=np.int32), scanned) # Scan_found_face =1 if not found =0
    info = np.r_['2', idxs, scanned, scan_found_face, classes[...,1:2]]
    print("boxes", boxes)
    print("info",info)
    # Return
    return boxes, info
  def scan_batch(
      self,
      batch: int,
      n_batches: int,
      inputs: Tuple[np.ndarray, str],
      start: int,
      end: int,
      fps: float = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Parse video and run inference for one batch.

    Args:
      batch: The number of this batch.
      n_batches: The total number of batches.
      inputs: The video to analyze. Either a np.ndarray of shape (n_frames, h, w, 3)
        with a sequence of frames in unscaled uint8 RGB format, or a path to a video file.
      start: The index of first frame of the video to analyze in this batch.
      end: The index of the last frame of the video to analyze in this batch.
      fps: Sampling frequency of the input video. Required if type(video) == np.ndarray.
    Returns:
      boxes: Scanned boxes in flat point form (n_frames, n_boxes, 4)
      classes: Detection scores for boxes (n_frames, n_boxes, 2)
      idxs: Indices of the scanned frames from the original video
    """

    logging.debug("Batch {}/{}...".format(batch, n_batches))
    # Slice the inputs to get the current batch of frames
    inputs_batch = inputs[start:end]
    boxes=[]
    classes=[]
    for frame in inputs_batch:
            results = self.face_detection.process(frame)
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    boxes.append([bbox.xmin, bbox.ymin, bbox.xmin + bbox.width, bbox.ymin + bbox.height])
                    classes.append([detection.score[0], detection.score[0]])
            else:
                boxes.append([0, 0, 0, 0])
                classes.append([0, 0])

    boxes = np.array(boxes).reshape([-1, self.max_faces, 4])
    classes = np.array(classes).reshape([-1, self.max_faces, 2])

    # Return the indices of the frames that were processed
    idxs = np.arange(start, end)

    return boxes, classes, idxs
    