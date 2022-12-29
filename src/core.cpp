/**
 * @file core.cpp
 * @author liufeng (liufeng1@huya.com)
 * @brief
 * @version 0.1
 * @date 2022-12-29
 *
 * @copyright Copyright (c) 2022
 *
 * Core program for computing cqt feature.
 *
 */

#include "Eigen/Core"
#include <stdio.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "Eigen/Sparse"
#include "unsupported/Eigen/FFT"

typedef Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorf;
typedef Eigen::Matrix<std::complex<float>, 1, Eigen::Dynamic, Eigen::RowMajor>
    Vectorcf;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Matrixf;
typedef Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic,
                      Eigen::RowMajor>
    Matrixcf;

static inline float hamming(int i, int nn) {
  return (0.54 - 0.46 * cos(2.0 * M_PI * (double)i * 1.0 / (double)(nn - 1.0)));
}

static std::vector<float> generate_hamming_window(int window_length) {
  std::vector<float> hamming_window;
  for (int j = 0; j < window_length; j++) {
    hamming_window.push_back(hamming(j, window_length));
  }
  return hamming_window;
}

/**
 * @brief engine to compute Constant-Q Transform
 */
class FastCqt {
 public:
  /**
   * @brief Construct a new Fast Cqt object
   * @param sample_rate
   * @param hop_size usually set 0.04, that means 1s has 25 frame
   */
  FastCqt(const int sample_rate, const float hop_size)
      : sample_rate_(sample_rate), hop_size_(hop_size) {
    int octave_resolution = 12;  // to get 96 bins
    int minimum_frequency = 32;
    int maximum_frequency = -1;
    if (maximum_frequency == -1) {
      maximum_frequency = sample_rate / 2;
    }
    // cqt_kernel_.resize(num_freq_, fft_length_);
    ComputeCqtKernel(sample_rate, octave_resolution, minimum_frequency,
                     maximum_frequency, &cqt_kernel_);
  }

  /**
   * @brief output Constant-Q Transform feature
   * @param signal_float input signal from 0~1
   * @param feat_dim_first True for output matrix with [feat_dim x time_dim],
   * else for [time_dime x feat_dim]
   * @return Matrixf result, note that is RowMajor
   */
  Matrixf Forward(const Vectorf &signal_float, bool feat_dim_first = false) {
    Matrixf cqt_feat = CoreCompute(signal_float);
    cqt_feat = cqt_feat.array() + 1e-9;
    float ref_value = cqt_feat.maxCoeff();
    cqt_feat = 20 * cqt_feat.array().max(1e-10).log10() - 20 * log10(ref_value);
    if (!feat_dim_first) {
      return cqt_feat.transpose();
    } else {
      return cqt_feat;
    }
  }

  int GetFreqDims() { return num_freq_; }

 private:
  bool ComputeCqtKernel(
      const int sample_rate, const int octave_resolution,
      const int min_frequency, const int max_frequency,
      Eigen::SparseMatrix<std::complex<float>> *res_cqt_kernel) {
    // Compute the constant ratio of frequency to resolution (= fk/(fk+1-fk))
    float quality_factor = 1.0 / (pow(2.0, 1.0 / octave_resolution) - 1.0);

    //# Compute the number of frequency channels for the CQT
    num_freq_ = round(octave_resolution *
                      log2(max_frequency * 1.0 / (min_frequency * 1.0)));

    // Compute the window length for the FFT
    fft_length_ = static_cast<int>(pow(
        2.0, ceil(log2(quality_factor * sample_rate / (min_frequency * 1.0)) +
                  1e-6)));

    Matrixcf cqt_kernel = Matrixcf::Zero(num_freq_, fft_length_);
    for (int i = 0; i < num_freq_; i++) {
      float freq_value =
          min_frequency * pow(2.0, i * 1.0 / (octave_resolution * 1.0));
      // Compute the window length in samples(nearest odd value to center the
      // temporal kernel on 0)
      int window_length =
          2 * round(quality_factor * sample_rate * 1.0 / freq_value / 2.0) + 1;
      Vectorcf temporal_kernel(window_length);
      auto hamming_win = generate_hamming_window(window_length);

      std::vector<float> arange_win;
      for (int j = -(window_length - 1) / 2; j < (window_length - 1) / 2 + 1;
           j++) {
        arange_win.push_back(j);
      }

      for (int j = 0; j < window_length; j++) {
        std::complex<float> tmp(0.0, 2 * M_PI * quality_factor * arange_win[j] /
                                         (window_length * 1.0));
        temporal_kernel[j] =
            std::complex<float>(hamming_win[j] / (window_length * 1.0), 0.0) *
            exp(tmp);
      }

      // Derive the pad width to center the temporal kernels
      int pad_width = (fft_length_ - window_length + 1) / 2;

      // Save the current temporal kernel at the center
      // (the zero-padded temporal kernels are not perfectly symmetric anymore
      // because of the even length here)
      cqt_kernel.block(i, pad_width, 1, window_length) = temporal_kernel;
    }

    Eigen::FFT<float> fft;
    for (int i = 0; i < num_freq_; i++) {
      Vectorcf x_frame = cqt_kernel.row(i).array();
      Vectorcf fft_out = fft.fwd(x_frame);
      cqt_kernel.row(i) = fft_out;
    }

    for (int i = 0; i < num_freq_; i++) {
      for (int j = 0; j < fft_length_; j++) {
        if (cqt_kernel.cwiseAbs().array()(i, j) < 0.01) {
          cqt_kernel(i, j) = std::complex<float>(0.0, 0.0);
        }
      }
    }

    for (int i = 0; i < num_freq_; i++) {
      for (int j = 0; j < fft_length_; j++) {
        if (cqt_kernel(i, j) != std::complex<float>(0.0, 0.0)) {
          cqt_kernel(i, j) = std::complex<float>(1 / (fft_length_ * 1.0), 0.0) *
                             conj(cqt_kernel(i, j));
        }
      }
    }

    res_cqt_kernel->resize(num_freq_, fft_length_);
    std::vector<Eigen::Triplet<std::complex<float>>> tripletList;
    for (int i = 0; i < num_freq_; i++) {
      for (int j = 0; j < fft_length_; j++) {
        if (cqt_kernel(i, j) != std::complex<float>(0.0, 0.0)) {
          tripletList.push_back(
              Eigen::Triplet<std::complex<float>>(i, j, cqt_kernel(i, j)));
        }
      }
    }

    res_cqt_kernel->setFromTriplets(tripletList.begin(), tripletList.end());
    res_cqt_kernel->makeCompressed();
    return true;
  }

  Matrixf CoreCompute(const Vectorf &audio_signal) {
    int time_resolution = 1.0 / hop_size_;
    int step_length = round(sample_rate_ * 1.0 / (time_resolution * 1.0));
    int number_times = audio_signal.size() * 1.0 / (step_length * 1.0);
    int left = ceil((fft_length_ - step_length) / 2.0);
    int right = floor((fft_length_ - step_length) / 2.0);
    Vectorf audio_paded = Vectorf::Zero(left + audio_signal.size() + right);
    audio_paded.segment(left, audio_signal.size()) = audio_signal;

    Eigen::FFT<float> fft;
    Matrixcf fft_out_mat(number_times, fft_length_);
    int i = 0;
    for (int j = 0; j < number_times; j++) {
      Vectorf x_frame = audio_paded.segment(i, fft_length_).array();
      fft_out_mat.row(j) = fft.fwd(x_frame);
      i = i + step_length;
    }

    Matrixcf mult_out = cqt_kernel_ * fft_out_mat.transpose();
    Matrixf cqt_feat = mult_out.cwiseAbs().array();
    return cqt_feat;
  }

 private:
  const int sample_rate_;
  const float hop_size_;
  int num_freq_ = -1;    // default is 96
  int fft_length_ = -1;  // default is 16384
  Eigen::SparseMatrix<std::complex<float>> cqt_kernel_;
};

extern "C" {

FastCqt *new_fastcqt(int sample_rate, float hop_size) {
  return new FastCqt(sample_rate, hop_size);
}

int get_freq_dims(FastCqt *t) { return t->GetFreqDims(); }

void compute_cqt(FastCqt *t, float *signal_p, int signal_len, float *res_p,
                 int res_len) {
  Eigen::VectorXf signal = Eigen::Map<Eigen::VectorXf>(signal_p, signal_len);
  Eigen::MatrixXf cqt = t->Forward(signal, false);
  int idx = 0;
  for (int i = 0; i < cqt.rows(); i++) {
    for (int j = 0; j < cqt.cols(); j++) {
      res_p[idx] = cqt(i, j);
      idx++;
    }
  }
  return;
}
}
