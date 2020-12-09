from abc import ABC
from pathlib import Path

import librosa
import numpy as np

from trainkit.datasets.base import BaseDataset


class AudioBaseDataset(BaseDataset, ABC):
    def __init__(self, run_params: dict,
                 hyper_params: dict,
                 **_ignored):
        super().__init__(run_params=run_params,
                         hyper_params=hyper_params,
                         **_ignored)

        self.audio_sr = hyper_params['audio_sr']
        self.spec_kws = hyper_params['spec_kws']

    def _read_audio(self, audio_path: Path,
                    return_mono_flag: bool = True,
                    start_sec: float = 0,
                    duration: float = None,
                    res_type: str = 'kaiser_best') -> np.ndarray:
        """Reads audio file and returns array with read signal

        Args:
            audio_path: path to audio file to read
            return_mono_flag: convert signal to mono
            start_sec: second to start reading
            duration: duration to read
            res_type: resample type if needed

        Returns:
            Read audio
        """
        audio, _ = librosa.load(audio_path,
                                sr=self.audio_sr,
                                mono=return_mono_flag,
                                offset=start_sec,
                                duration=duration,
                                res_type=res_type)

        return audio

    def _mel_spectrogram(self, audio: np.ndarray,
                         decibel_scale: bool) -> np.ndarray:
        """Converts given signal into mel spectrogram (amplitude, power = 1)

        Args:
            audio: input signal
            decibel_scale: if True than amplitude spectrogram will be converted
                to dB-scaled spectrogram

        Returns:
            Mel spectrogram, shape: [n_mels, t]
        """
        spec = librosa.feature.melspectrogram(audio, sr=self.audio_sr, power=1, **self.spec_kws)

        if decibel_scale:
            spec = librosa.amplitude_to_db(spec, ref=np.max)

        return spec

    def _spectrogram(self, audio: np.ndarray,
                     decibel_scale: bool) -> np.ndarray:
        """Converts given signal into general spectrogram (STFT)

        Args:
            audio: input signal
            decibel_scale: if True than amplitude spectrogram will be converted
                to dB-scaled spectrogram

        Returns:
            Spectrogram, shape=[1 + n_fft/2, n_frames]
        """
        spec = np.abs(librosa.stft(audio, **self.spec_kws))

        if decibel_scale:
            spec = librosa.amplitude_to_db(spec, ref=np.max)

        return spec

    def _constant_pad_audio(self, audio: np.ndarray,
                            left_pad_frames: int = None,
                            right_pad_frames: int = None,
                            left_pad_sec: float = None,
                            right_pad_sec: float = None,
                            pad_value: float = 0) -> np.ndarray:
        """Adds paddings before and after given `audio` array.

        Method uses sample rate from `init` to convert pad seconds into audio frames.
        If pad_frames and pad_sec given than pad_frames will be used.

        Args:
            audio: array with audio signal
            left_pad_frames: padding size in frames before given audio
            right_pad_frames: padding size in frames after given audio
            left_pad_sec: padding size in seconds before given audio
            right_pad_sec: padding size in seconds after given audio
            pad_value: padding value

        Returns:
            Padded audio
        """
        if left_pad_sec is None and left_pad_frames is None:
            raise ValueError('Param left_pad_sec or left_pad_frames must be given')
        if right_pad_sec is None and right_pad_frames is None:
            raise ValueError('Param right_pad_sec or right_pad_frames must be given')

        if left_pad_frames is None and right_pad_frames is None:
            left_pad_frames = int(left_pad_sec * self.audio_sr)
            right_pad_frames = int(right_pad_sec * self.audio_sr)

        audio_padded = np.pad(array=audio,
                              pad_width=(left_pad_frames, right_pad_frames),
                              constant_values=(pad_value, pad_value))

        return audio_padded
