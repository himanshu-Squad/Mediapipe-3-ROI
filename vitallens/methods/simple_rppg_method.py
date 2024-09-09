

import abc
import numpy as np
from prpy.constants import SECONDS_PER_MINUTE
from prpy.numpy.face import get_roi_from_det
from prpy.numpy.image import reduce_roi
from prpy.numpy.signal import interpolate_cubic_spline, estimate_freq
from typing import Union, Tuple

from vitallens.constants import CALC_HR_MIN, CALC_HR_MAX
from vitallens.methods.rppg_method import RPPGMethod
from vitallens.utils import merge_faces

class SimpleRPPGMethod(RPPGMethod):
  def __init__(
      self,
      config: dict
    ):
    super(SimpleRPPGMethod, self).__init__(config=config)
    self.model = config['model']
    self.roi_method = config['roi_method']
    self.signals = config['signals']
  @abc.abstractmethod
  def algorithm(
      self,
      rgb: np.ndarray,
      fps: float
    ):
    pass
  @abc.abstractmethod
  def pulse_filter(self, 
      sig: np.ndarray,
      fps: float
    ) -> np.ndarray:
    pass
  def __call__(
      self,
      frames: Union[np.ndarray, str],
      faces: np.ndarray,
      fps: float,
      override_fps_target: float = None,
      override_global_parse: float = None,
    ) -> Tuple[dict, dict, dict, dict, np.ndarray]:
    """Estimate pulse signal from video frames using the subclass algorithm.

    Args:
      frames: The video frames. Shape (n_frames, h, w, c)
      faces: The face detection boxes as np.int64. Shape (n_frames, 4) in form (x0, y0, x1, y1)
      fps: The rate at which video was sampled.
      override_fps_target: Override the method's default inference fps (optional).
      override_global_parse: Has no effect here.
    Returns:
      data: A dictionary with the values of the estimated vital signs.
      unit: A dictionary with the units of the estimated vital signs.
      conf: A dictionary with the confidences of the estimated vital signs.
      note: A dictionary with notes on the estimated vital signs.
      live: Dummy live confidence estimation (set to always 1). Shape (1, n_frames)
    """
    # Compute temporal union of ROIs
    u_roi = merge_faces(faces)
    faces = faces - [u_roi[0], u_roi[1], u_roi[0], u_roi[1]]
    # Reduce ROI and extract RGB signal
    roi_ds = np.asarray([get_roi_from_det(f, roi_method=self.roi_method) for f in faces], dtype=np.int64)  # ROI for each frame (n, 4)
    rgb_ds = reduce_roi(video=frames, roi=roi_ds)  # RGB signal for each frame (n, 3)

    # Perform rPPG algorithm step
    sig_ds = self.algorithm(rgb_ds, fps)

    # Interpolate to original sampling rate
    sig = interpolate_cubic_spline(
            x=np.arange(rgb_ds.shape[0]), y=sig_ds, xs=np.arange(frames.shape[0]), axis=1)

    # Filter
    sig = self.pulse_filter(sig, fps)
    # Interpolate to original sampling rate (n_frames,)
    sig = interpolate_cubic_spline(
    x=np.arange(rgb_ds.shape[0]), y=sig_ds, xs=np.arange(frames.shape[0]), axis=1)
    # Filter (n_frames,)
    sig = self.pulse_filter(sig, fps)
    # Estimate HR
    hr = estimate_freq(
      sig, f_s=fps, f_res=0.1/SECONDS_PER_MINUTE,
      f_range=(CALC_HR_MIN/SECONDS_PER_MINUTE, CALC_HR_MAX/SECONDS_PER_MINUTE),
      method='periodogram') * SECONDS_PER_MINUTE
    # Assemble results
    data, unit, conf, note = {}, {}, {}, {}
    for name in self.signals:
      if name == 'heart_rate':
        data[name] = hr
        unit[name] = 'bpm'
        conf[name] = 1.0
        note[name] = 'Estimate of the global heart rate using {} method. This method is not capable of providing a confidence estimate, hence returning 1.'.format(self.model)
      elif name == 'ppg_waveform':
        data[name] = sig
        unit[name] = 'unitless'
        conf[name] = np.ones_like(sig)
        note[name] = 'Estimate of the ppg waveform using {} method. This method is not capable of providing a confidence estimate, hence returning 1.'.format(self.model)
    # Return results
    #return data, unit, conf, note, np.ones_like(sig)
    return data, unit, conf, note, np.ones_like(sig),rgb_ds
  
