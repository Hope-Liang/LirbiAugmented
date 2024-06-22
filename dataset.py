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
class Nisqa(Dataset):
    
    def __init__(
        self,
        data_path: str = '../../datasets/NISQA_Corpus',
        valid: str = 'train',
        label_type: str = 'mos',
        sample_rate: int = 16000,
    ):
        self._data_path = data_path
        if valid == 'train':
            self._df = pd.read_csv(os.path.join(data_path, 'NISQA_TRAIN_SIM', 'NISQA_TRAIN_SIM_file.csv'))
        else:
            self._df = pd.read_csv(os.path.join(data_path, 'NISQA_VAL_SIM', 'NISQA_VAL_SIM_file.csv'))
        self._num_samples = len(self._df)
        self._valid = valid
        self._sample_rate = sample_rate
        self._label_type = label_type

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
        # TODO: Add so that the distortion is extracted.
        for label, augmentation in tqdm.tqdm(
            zip(self._df[self._label_type], self._df['source']),
            total=self._num_samples,
            desc='Loading labels...',
        ):
            labels.append(label)
            if augmentation not in aug_map:
                aug_map[augmentation] = len(aug_map)
            augmentations.append(aug_map[augmentation])

        return labels, augmentations, aug_map

    def _load_clips(self):
        mag_specs = []
        for path in tqdm.tqdm(
            self._df['filepath_deg'], total=self._num_samples, desc='Loading clips...',
        ):
            wav, _ = librosa.load(os.path.join(self._data_path, path), sr=self._sample_rate)
            signal = audio.Audio(wav, self._sample_rate)
            signal = signal.repetitive_crop(10 * self._sample_rate)
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
        augmentations = torch.IntTensor(augmentations)
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
