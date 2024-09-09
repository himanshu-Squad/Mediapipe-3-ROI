

from datetime import datetime
from enum import IntEnum
import json
import logging
import numpy as np
import os
from prpy.constants import SECONDS_PER_MINUTE
from typing import Union

from vitallens.constants import DISCLAIMER
from vitallens.constants import CALC_HR_MIN, CALC_HR_MAX, CALC_HR_WINDOW_SIZE
from vitallens.constants import CALC_RR_MIN, CALC_RR_MAX, CALC_RR_WINDOW_SIZE
from vitallens.methods.g import GRPPGMethod
from vitallens.methods.chrom import CHROMRPPGMethod
from vitallens.methods.pos import POSRPPGMethod
from vitallens.methods.vitallens import VitalLensRPPGMethod
from vitallens.signal import windowed_freq, windowed_mean
from vitallens.ssd import FaceDetector
from vitallens.utils import load_config, probe_video_inputs, check_faces, convert_ndarray_to_list

class Method(IntEnum):
  VITALLENS = 1
  G = 2
  CHROM = 3
  POS = 4

logging.getLogger().setLevel("INFO")

class VitalLens:
    def __init__(
        self, 
        method: Method = Method.VITALLENS,
        api_key: str = None,
        detect_faces: bool = True,
        estimate_running_vitals: bool = True,
        export_to_json: bool = True,
        export_dir: str = "."
    ):
        """Initialisation.

        Args:
            method: The rPPG method to be used for inference.
            api_key: Usage key for the VitalLens API (required for Method.VITALLENS)
            detect_faces: `True` if faces need to be detected, otherwise `False`.
            estimate_running_vitals: Set `True` to compute running vitals (e.g., `running_heart_rate`).
            export_to_json: If `True`, write results to a json file.
            export_dir: The directory to which json files are written.
        """
        self.api_key = api_key
        # Load the config and model
        self.config = load_config(method.name.lower() + ".yaml")
        self.method = method
        if self.config['model'] == 'g':
            self.rppg = GRPPGMethod(self.config)
        elif self.config['model'] == 'chrom':
            self.rppg = CHROMRPPGMethod(self.config)
        elif self.config['model'] == 'pos':
            self.rppg = POSRPPGMethod(self.config)
        elif self.config['model'] == 'vitallens':
            if self.api_key is None or self.api_key == '':
                raise ValueError("An API key is required to use Method.VITALLENS, but was not provided. "
                                 "Get one for free at https://www.rouast.com/api.")
            self.rppg = VitalLensRPPGMethod(self.config, self.api_key)
        else:
            raise ValueError("Method {} not implemented!".format(self.config['model']))
        
        self.detect_faces = detect_faces
        self.estimate_running_vitals = estimate_running_vitals
        self.export_to_json = export_to_json
        self.export_dir = export_dir

    def __call__(
        self,
        video: Union[np.ndarray, str],
        faces: Union[np.ndarray, list] = None,
        fps: float = None,
        override_fps_target: float = None,
        override_global_parse: bool = None,
        export_filename: str = None
    ) -> list:
        """Run rPPG inference.

        Args:
            video: The video to analyze. Either a np.ndarray of shape (n_frames, h, w, 3)
                with a sequence of frames in unscaled uint8 RGB format, or a path to a
                video file.
            faces: Face boxes in flat point form, containing [x0, y0, x1, y1] coords.
                Ignored unless detect_faces=False.
            fps: Sampling frequency of the input video. Required if type(video) == np.ndarray. 
            override_fps_target: Target fps at which rPPG inference should be run (optional).
            override_global_parse: If True, always use global parse. If False, don't use global parse.
                If None, choose based on video.
            export_filename: Filename for json export if applicable.

        Returns:
            result: Analysis results as a list of faces with vital signs and other information.
        """
        # Probe inputs
        #inputs_shape, fps, _ = probe_video_inputs(video=video, fps=fps)
        inputs_shape, fps, ds_factor = probe_video_inputs(video=video, fps=fps)

        # Warning if using long video
        target_fps = override_fps_target if override_fps_target is not None else self.rppg.fps_target
        if self.method != Method.VITALLENS and inputs_shape[0]/fps*target_fps > 3600:
            logging.warning("Inference for long videos has yet to be optimized for POS / G / CHROM. This may run out of memory and crash.")
        
        _, height, width, _ = inputs_shape

        if self.detect_faces:
            # Use MediaPipe-based face detection (not implemented here)
            faces_rel = detect_faces(video, inputs_shape,fps)  # Placeholder function
            
            # If no faces detected: return empty list
            if len(faces_rel) == 0:
                logging.warning("No faces to analyze")
                return []
            # Convert relative face coordinates to absolute units
            faces = []
            for frame_faces in faces_rel:  # Loop over each frame's detections
                # Ensure frame_faces has shape (n, 4), where n is the number of faces
                frame_faces = np.array(frame_faces)  # Convert to numpy array if it's not already
                print(f"Original frame_faces shape: {frame_faces.shape}, contents: {frame_faces}")

                # Check if the shape is correct. It should be (n_faces, 4).
                if frame_faces.size % 4 != 0:  # If last dimension is not 4, reshape
                    logging.warning(f"Unexpected size of face data: {frame_faces.size}. Cannot reshape into bounding boxes.")
                    continue
                try:
                    # Reshape to (n_faces, 4)
                    frame_faces = frame_faces.reshape(-1, 4)  # Ensure it has the shape (n_faces, 4)
                    print(f"Reshaped frame_faces shape: {frame_faces.shape}, contents: {frame_faces}")
                except ValueError as e:
                    logging.error(f"Error reshaping face data: {e}")
                    continue

                # Now we can safely multiply by [width, height, width, height]
                abs_faces = frame_faces * np.array([width, height, width, height])
                faces.append(abs_faces)

            faces = np.array(faces).astype(np.int64)

            # Check the shape of faces before transposing
            print(f"Shape of faces before transpose: {faces.shape}")
            # Squeeze if the shape has an extra dimension
            faces = np.squeeze(faces, axis=0) if faces.shape[0] == 1 else faces
            print(f"Shape after squeezing: {faces.shape}")
        
            if faces.ndim == 2:
                faces = np.expand_dims(faces, axis=0)  # Adjust dimensions if necessary
            elif faces.ndim == 3 and faces.shape[0] == 1:
                faces = np.expand_dims(faces, axis=1)  # Ensuring the faces dimension is correctly handled
            if faces.ndim != 4:
                logging.error(f"Invalid shape for faces: {faces.shape}, expected 4 dimensions.")
                return []

            try:
                faces = np.transpose(faces, (1, 0, 2, 3))
                print(f"Shape of faces after transpose: {faces.shape}")
            except ValueError as e:
                logging.error(f"Error during transpose: {e}")
                return []
        else:
            faces = check_faces(faces, inputs_shape)  
        
        # Run separately for each face
        results = []
        for face in faces:
            # Run selected rPPG method
            data, unit, conf, note, live, rgb_ds = self.rppg(
                frames=video, faces=face, fps=fps,
                override_fps_target=override_fps_target,
                override_global_parse=override_global_parse
            )
            
            # Parse face results
            face_result = {'face': {
                'coordinates': face,
                'confidence': live,
                'note': "Face detection coordinates for this face, along with live confidence levels."
            }}
            
            # Parse vital signs results
            vital_signs_results = {}
            for name in self.config['signals']:
                vital_signs_results[name] = {
                    '{}'.format('data' if 'waveform' in name else 'value'): data[name],
                    'unit': unit[name],
                    'confidence': conf[name],
                    'note': note[name]
                }
            
            if self.estimate_running_vitals:
                try:
                    if 'ppg_waveform' in self.config['signals']:
                        window_size = int(CALC_HR_WINDOW_SIZE * fps)
                        running_hr = windowed_freq(
                            x=data['ppg_waveform'], f_s=fps, f_res=0.005,
                            f_range=(CALC_HR_MIN/SECONDS_PER_MINUTE, CALC_HR_MAX/SECONDS_PER_MINUTE),
                            window_size=window_size, overlap=window_size//2
                        ) * SECONDS_PER_MINUTE
                        running_conf = windowed_mean(
                            x=conf['ppg_waveform'], window_size=window_size, overlap=window_size//2
                        )
                        vital_signs_results['running_heart_rate'] = {
                            'data': running_hr,
                            'unit': 'bpm',
                            'confidence': running_conf,
                            'note': 'Estimate of the running heart rate using VitalLens, along with frame-wise confidences between 0 and 1.',
                        }
                    if 'respiratory_waveform' in self.config['signals']:
                        window_size = int(CALC_RR_WINDOW_SIZE * fps)
                        running_rr = windowed_freq(
                            x=data['respiratory_waveform'], f_s=fps, f_res=0.005,
                            f_range=(CALC_RR_MIN/SECONDS_PER_MINUTE, CALC_RR_MAX/SECONDS_PER_MINUTE),
                            window_size=window_size, overlap=window_size//2
                        ) * SECONDS_PER_MINUTE
                        running_conf = windowed_mean(
                            x=conf['respiratory_waveform'], window_size=window_size, overlap=window_size//2
                        )
                        vital_signs_results['running_respiratory_rate'] = {
                            'data': running_rr,
                            'unit': 'bpm',
                            'confidence': running_conf,
                            'note': 'Estimate of the running respiratory rate using VitalLens, along with frame-wise confidences between 0 and 1.',
                        }
                except ValueError as e:
                    logging.warning("Issue while computing running vitals: {}".format(e))
            
            face_result['vital_signs'] = vital_signs_results
            face_result['message'] = DISCLAIMER
            results.append(face_result)
        
        # Export to json
        if self.export_to_json:
            os.makedirs(self.export_dir, exist_ok=True)
            export_filename = "{}.json".format(export_filename) if export_filename is not None else 'vitallens_{}.json'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
            with open(os.path.join(self.export_dir, export_filename), 'w') as f:
                json.dump(convert_ndarray_to_list(results), f, indent=4)
        
        return results
# New detect_faces function that uses the FaceDetector
def detect_faces(video, inputs_shape, fps):
    face_detector = FaceDetector(max_faces=1, fs=fps, score_threshold=0.5, iou_threshold=0.5)
    return face_detector(video, inputs_shape, fps)