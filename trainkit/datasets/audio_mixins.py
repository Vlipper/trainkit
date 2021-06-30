from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING

import librosa
import numpy as np

if TYPE_CHECKING:
    from typing import Optional


__all__ = [
    'AudioBaseOperationsMixin',
    'AudioSpectrogramMixin'
]


class AudioBaseOperationsMixin(ABC):
    """Mixin with base operations on audios
    """

    def __init__(self, run_params: dict,
                 hyper_params: dict,
                 **_ignored):

        self.audio_sr = hyper_params['audio_sr']

    def _read_audio(self, audio_path: Path,
                    return_mono_flag: bool = True,
                    start_sec: float = 0,
                    duration: 'Optional[float]' = None,
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

    def _constant_pad_audio(self, audio: np.ndarray,
                            left_pad_samples: 'Optional[int]' = None,
                            right_pad_samples: 'Optional[int]' = None,
                            left_pad_sec: 'Optional[float]' = None,
                            right_pad_sec: 'Optional[float]' = None,
                            pad_value: float = 0) -> np.ndarray:
        """Adds paddings before and after given `audio` array

        Method uses sample rate from `init` to convert pad seconds into audio samples.
        If pad_samples and pad_sec given than pad_samples will be used.

        Args:
            audio: array with audio signal
            left_pad_samples: padding size in samples before given audio
            right_pad_samples: padding size in samples after given audio
            left_pad_sec: padding size in seconds before given audio
            right_pad_sec: padding size in seconds after given audio
            pad_value: padding value

        Returns:
            Padded audio
        """
        if left_pad_sec is None and left_pad_samples is None:
            raise ValueError('Param "left_pad_sec" or "left_pad_samples" must be given')
        if right_pad_sec is None and right_pad_samples is None:
            raise ValueError('Param "right_pad_sec" or "right_pad_samples" must be given')

        if left_pad_samples is None and right_pad_samples is None:
            left_pad_samples = int(left_pad_sec * self.audio_sr)
            right_pad_samples = int(right_pad_sec * self.audio_sr)

        audio_padded = np.pad(array=audio,
                              pad_width=(left_pad_samples, right_pad_samples),
                              constant_values=(pad_value, pad_value))

        return audio_padded

    # ToDo add method _truncate_or_pad to specified size
    #   left_pad_samples, right_pad_samples -- make default value as 0


class AudioSpectrogramMixin(ABC):
    """Mixin with spectrogram mining operations
    """

    def __init__(self, run_params: dict,
                 hyper_params: dict,
                 **_ignored):

        self.audio_sr = hyper_params['audio_sr']
        self.spec_kws = hyper_params['spec_kws']

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
        spec = librosa.feature.melspectrogram(audio,
                                              sr=self.audio_sr,
                                              power=1,
                                              **self.spec_kws)

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
            Spectrogram, shape=[1 + n_fft/2, n_samples]
        """
        spec = np.abs(librosa.stft(audio, **self.spec_kws))

        if decibel_scale:
            spec = librosa.amplitude_to_db(spec, ref=np.max)

        return spec
