#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:liufeng
# datetime:2022/12/27 4:52 PM
# software: PyCharm

import ctypes
import os
import numpy as np
from typeguard import check_argument_types, typechecked

dir_path = os.path.dirname(os.path.realpath(__file__))
so_path = os.path.join(dir_path, "../build/libFastCqt.so")
handle = ctypes.CDLL(so_path)

handle.new_fastcqt.argtypes = [ctypes.c_int, ctypes.c_float]
handle.get_freq_dims.argtypes = [ctypes.c_void_p]
handle.compute_cqt.argtypes = [ctypes.c_void_p,
                               ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                               ctypes.POINTER(ctypes.c_float), ctypes.c_int]


def convert_type(python_input):
  """change python-type to c-type, support int, float and string"""
  ctypes_map = {int: ctypes.c_int,
                float: ctypes.c_float,
                str: ctypes.c_char_p
                }
  input_type = type(python_input)
  if input_type in ctypes_map:
    return ctypes_map[input_type](bytes(python_input, encoding="utf-8") if type(
      python_input) is str else python_input)
  else:
    raise Exception("Unsupported input type")


@typechecked
class FastCqt(object):
  def __init__(self, sample_rate: int, hop_size: float):
    self._sample_rate = sample_rate
    self._hop_size = hop_size
    self.obj = handle.new_fastcqt(convert_type(sample_rate),
                                  convert_type(hop_size))
    self._freq_dims = handle.get_freq_dims(self.obj)
    return

  def compute_cqt(self, signal_float: np.ndarray) -> np.ndarray:
    assert check_argument_types()
    signal_c = np.ctypeslib.as_ctypes(signal_float)
    length = int(len(signal_float) / self._sample_rate * (1 / self._hop_size))
    cqt_numpy = np.zeros(length * self._freq_dims, dtype=np.float32)
    cqt_c_point = np.ctypeslib.as_ctypes(cqt_numpy)
    handle.compute_cqt(self.obj, signal_c, len(signal_float), cqt_c_point,
                       len(cqt_numpy))
    cqt_feat = cqt_numpy.reshape([length, self._freq_dims])
    return cqt_feat


def _example():
  import librosa
  wav_path = os.path.join(dir_path, "../demo/demo.wav")
  sig, _ = librosa.load(wav_path, sr=16000)
  sig = np.ascontiguousarray(sig, dtype=np.float32)
  cpp_cqt_ext = FastCqt(16000, 0.04)
  cpp_cqt = cpp_cqt_ext.compute_cqt(sig)
  print("input length:{}".format(np.shape(sig)))
  print("output shape:{}".format(np.shape(cpp_cqt)))
  return


if __name__ == '__main__':
  _example()
