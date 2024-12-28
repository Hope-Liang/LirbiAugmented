"""Dataset loader for LibriSpeech."""

import functools
import os
import random
import sys
from typing import Callable, Optional, Sequence, Union

import gin
import numpy as np
import librosa
import pandas as pd
import pesq
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import tqdm

import audio
import augmentations
import utils


# Disable tqdm globally.
# tqdm.tqdm.__init__ = functools.partialmethod(tqdm.tqdm.__init__, disable=True)

_AugmentationFn = Callable[[audio.Audio], audio.Audio]

class NoiseDataset:
   
    def __init__(
        self,
        data_paths: Sequence[str] = ('../../datasets/noise_demand', '../../datasets/noise_freesound'),
        sample_rate: int = 16_000,
        clip_duration: float = 10.0,
    ):
        self._sample_rate = sample_rate
        self._clip_duration = clip_duration
        self._clip_length = int(self._clip_duration * self._sample_rate)
        self._data_wavs = {}
        self._data_wavs_paths = {}
        print('Loading noises...')
        for data_path in data_paths:
            for cur_dir in os.listdir(data_path):
                cur_dir_path = os.path.join(data_path, cur_dir)
                wavs, wav_paths = self._load_clips(cur_dir_path)
                if wav_paths != []:
                    self._data_wavs[cur_dir_path] = wavs
                    self._data_wavs_paths[cur_dir_path] = wav_paths
           
    def _load_clips(self, path: str) -> Sequence[np.ndarray]:
        """loaded, croped/padded, resampled"""
        wavs = []
        wav_paths = []
        for dp, _, filenames in os.walk(path):
            for filename in tqdm.tqdm(
                filenames, total=len(filenames), desc=dp):
                filepath = os.path.join(dp, filename)
                if os.path.splitext(filename)[1] == '.flac':
                    sig = audio.Audio.read_flac(filepath)
                elif os.path.splitext(filename)[1] == '.wav':
                    sig = audio.Audio.read_wav(filepath)
                else:
                    continue
                sig = sig.repetitive_crop(
                    length=int(self._clip_duration * sig.rate))
                samples = librosa.resample(
                    sig.samples[:, 0],
                    orig_sr=sig.rate,
                    target_sr=self._sample_rate,
                    res_type='scipy',
                )
                if sig.rate == self._sample_rate:
                    wavs.append(samples)
                    wav_paths.append(filepath)
        return wavs, wav_paths

    def get_random_sample_path(self):
        _, paths = random.choice(list(self._data_wavs_paths.items()))
        return random.choice(paths)

    def get_random_sample(self):
        _, wavs = random.choice(list(self._data_wavs.items()))
        return random.choice(wavs)


class LibriSpeech(Dataset):
   
    def __init__(
        self,
        data_path: str = '../../datasets/LibriSpeech',
        valid: str = 'train',
        sample_rate: int = 16_000,
        clip_duration: float = 10.0,
        max_num_clips: Union[int, None] = 10_000,
    ):
        self._max_num_clips = max_num_clips
        self._sample_rate = sample_rate
        self._clip_duration = clip_duration
        self._clip_length = int(self._clip_duration * self._sample_rate)
        self._paths = self._load_paths(data_path, valid)
   
    def _load_paths(self, data_path: str, valid: str):
        rel_paths = pd.read_csv(os.path.join(data_path, valid) + '.csv').filepaths
        paths = [os.path.join(data_path, rel_path) for rel_path in rel_paths]
        if self._max_num_clips is not None:
            if len(paths) > self._max_num_clips:
                paths = paths[:self._max_num_clips]
        return paths

    def _stft(self, wav):
        return np.abs(librosa.stft(wav, n_fft=512)).T

    def load_wav(self, file_path: str):
        sig = audio.Audio.read_flac(file_path)
        sig = sig.repetitive_crop(length=int(self._clip_duration * sig.rate))
        sig = sig.resample(self._sample_rate)
        return sig

    def __getitem__(self, idx: int):
        return self._wavs[idx], self._labels[idx]
   
    def __len__(self):
        return len(self._paths)
   
    # def _random_pad(self, samples: np.ndarray) -> np.ndarray:
    #     cur_length = samples.shape[0]
    #     max_length = self._clip_length - cur_length
    #     length_beginning = np.random.randint(0, max_length + 1)
    #     return np.pad(
    #         samples,
    #         (length_beginning, max_length - length_beginning),
    #         'constant',
    #     )

@gin.configurable
class LibriAugmented(Dataset):
    
    def __init__(
        self,
        data_path: str = '../../datasets/LibriAugmented',
        valid: str = 'train',
        label_type: str = 'visqol',
        sample_rate: int = 16000,
        use_multi_augmentations: bool = False,
        q_inv_augmentations: Union[Sequence[_AugmentationFn], None] = None,
        num_samples_per_class: Optional[int] = None,
    ):
        self._data_path = data_path
        self._df = pd.read_csv(os.path.join(data_path, f'{valid}.csv'))
        if use_multi_augmentations:
            tmp_df = pd.read_csv(os.path.join(data_path, f'{valid}2.csv'))
            self._df = pd.concat([self._df, tmp_df])
        self._num_samples = len(self._df)
        self._valid = valid
        self._label_type = label_type
        self._sample_rate = sample_rate
        if q_inv_augmentations is None:
            self._q_inv_augmentations = list()
        else:
            self._q_inv_augmentations = q_inv_augmentations

        self._num_samples_per_class = num_samples_per_class
        self._labels, self._augmentations, self._aug_map = self._load_labels()
        self._mag_specs = self._load_clips()

    @property
    def num_augmentations(self):
        return len(self._aug_map)
    
    @property
    def label_type(self):
        return self._label_type

    def _load_labels(self):
        labels, augmentations = [], []
        # Map augmentations to integers.
        aug_map = {}
        num_samples_extracted = {}
        for label, augmentation in tqdm.tqdm(
            zip(self._df[self._label_type], self._df['augmentation']),
            total=self._num_samples,
            desc='Loading labels...',
        ):
            if augmentation in num_samples_extracted:
                num_samples_extracted[augmentation] += 1
                if self._num_samples_per_class is not None and num_samples_extracted[augmentation] > self._num_samples_per_class:
                    continue
            else:
                num_samples_extracted[augmentation] = 1
            labels.append(label)
            if augmentation not in aug_map:
                aug_map[augmentation] = len(aug_map)
            augmentations.append(aug_map[augmentation])

        return labels, augmentations, aug_map

    def _load_clips(self):
        mag_specs = []
        num_samples_extracted = {}
        for path, augmentation in tqdm.tqdm(
            zip(self._df['processed'], self._df['augmentation']),
            total=self._num_samples,
            desc='Loading clips...',
        ):
            if augmentation in num_samples_extracted:
                num_samples_extracted[augmentation] += 1
                if self._num_samples_per_class is not None and num_samples_extracted[augmentation] > self._num_samples_per_class:
                    continue
            else:
                num_samples_extracted[augmentation] = 1
            wav, _ = librosa.load(path, sr=self._sample_rate)
            signal = audio.Audio(wav, self._sample_rate)
            if self._valid == 'train':
                for q_inv_augmentation in self._q_inv_augmentations:
                    signal = q_inv_augmentation(signal)
            samples = np.squeeze(signal.samples)
            spec = utils.stft(samples)
            mag_specs.append(spec)

        return mag_specs
    
   
    def __getitem__(self, idx: int):
        return self._mag_specs[idx], self._labels[idx], self._augmentations[idx]
   
    def __len__(self):
        return len(self._mag_specs)
 
    def collate_fn(self, batch: list):
        mag_specs, labels, augmentations = zip(*batch)
        mag_specs = torch.FloatTensor(np.array(mag_specs))
        labels = torch.FloatTensor(labels)
        augmentations = torch.LongTensor(augmentations)
        return mag_specs, labels, augmentations


@gin.configurable
def get_dataloader(
    dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
    )
