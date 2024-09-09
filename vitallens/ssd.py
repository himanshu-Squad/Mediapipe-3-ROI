

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
# Define landmark indices for regions of interest (ROIs)
_left_cheek = [117, 118, 119, 120, 100, 142, 203, 206, 205, 50, 117]
_right_cheek = [346, 347, 348, 349, 329, 371, 423, 426, 425, 280, 346]
_forehead = [109, 10, 338, 337, 336, 285, 8, 55, 107, 108, 109]
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
    self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=max_faces)
  def __call__(
      self,
      inputs: np.ndarray,
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
    
    n_frames = inputs.shape[0]   
    results = [self.scan_frame(frame) for frame in inputs]
    boxes = np.array([r[0] for r in results])
    classes = np.array([r[1] for r in results])
       
    # Add print statements to debug
    print(f"Number of frames processed: {inputs.shape[0]}")
    print(f"Number of face detections: {boxes.shape[0]}")
    print(f"Shape of boxes (before reshape): {boxes.shape}")
    print(f"Shape of classes: {classes.shape}")
    # Ensure the number of detections matches the number of frames
    if boxes.shape[0] != inputs.shape[0]:
      raise ValueError(f"Number of detections ({boxes.shape[0]}) does not match number of frames ({inputs.shape[0]})")
    # If no faces found, return empty arrays
    if boxes.shape[1]==0:
      logging.warning("No faces found")
      return [],[]    
    # Assort info: idx, scanned, scan_found_face, confidence
    idxs = np.arange(n_frames)[..., np.newaxis]
    idxs = np.expand_dims(idxs, axis=1)  # Expand to (n_frames, 1, 1)
    scanned = np.ones((n_frames, boxes.shape[1], 1), dtype=np.int32)
    scan_found_face = np.where(classes[..., 1:2] < self.score_threshold, np.zeros((n_frames, boxes.shape[1], 1), dtype=np.int32), scanned)
    idxs = np.broadcast_to(idxs, scanned.shape)  # Ensure same shape as scanned
    info = np.concatenate([idxs, scanned, scan_found_face, classes[..., 1:2]], axis=-1)
    print("boxes", boxes)
    print("info",info)
    # Return
    return boxes, info
  def scan_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference on a batch of frames."""
        
        boxes = []
        classes = []
        results = self.face_mesh.process(frame)
       
        if results.multi_face_landmarks:

          for landmarks in results.multi_face_landmarks:
                    # Calculate the ROIs using the given landmark indices
                    left_cheek_bbox = self.get_roi_bbox(landmarks, _left_cheek, frame.shape)
                    right_cheek_bbox = self.get_roi_bbox(landmarks, _right_cheek, frame.shape)
                    forehead_bbox = self.get_roi_bbox(landmarks, _forehead, frame.shape)
                    # Combine the ROIs into one detection (flattening or combining them in some logical manner)
                    # Combine all three ROIs into a single bounding box
                    if np.all(left_cheek_bbox) and np.all(right_cheek_bbox) and np.all(forehead_bbox):
                      x_min = min(left_cheek_bbox[0], right_cheek_bbox[0], forehead_bbox[0])
                      y_min = min(left_cheek_bbox[1], right_cheek_bbox[1], forehead_bbox[1])
                      x_max = max(left_cheek_bbox[2], right_cheek_bbox[2], forehead_bbox[2])
                      y_max = max(left_cheek_bbox[3], right_cheek_bbox[3], forehead_bbox[3])
                      combined_bbox = [x_min, y_min, x_max, y_max]
                    else:
                      combined_bbox = [0, 0, 0, 0]  # Fallback to no detection
                    # Append all the ROIs as boxes
                    boxes.append(combined_bbox)
                    classes.append([1])  # Assume score of 1 for these ROIs
        else:
            # No faces detected, append empty bounding boxes
            boxes.append([[0, 0, 0, 0]])
            classes.append([0])
        # Debug print to confirm shape before reshaping
        #print(f"Boxes before reshaping: {np.array(boxes).shape}")
        if len(boxes)==0:
            boxes=np.array([[0, 0, 0, 0]]) # Default to no faces detected
        boxes = np.array(boxes).reshape([-1,4]) # (n_faces, 3 ROIs, 4 coords)
        classes = np.array(classes).reshape([-1,1])

        # Return the indices of the frames that were processed
        
        return boxes, classes

  def get_roi_bbox(self, landmarks, roi_indices, frame_shape):
        """Calculate bounding box for a region of interest (ROI) based on landmarks."""
        h, w, _ = frame_shape
        x_coords = [landmarks.landmark[i].x * w for i in roi_indices]
        y_coords = [landmarks.landmark[i].y * h for i in roi_indices]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    