import random
import re
import cv2
import torch
import numpy as np
import cv2
import sys
import gc

from AMNLT.configs.smt_dan_config.ExperimentConfig import ExperimentConfig
from AMNLT.utils.smt_dan_utils.data_augmentation import augment, convert_img_to_tensor
from AMNLT.utils.smt_dan_utils.utils import check_and_retrieveVocabulary
from rich import progress
from lightning import LightningDataModule
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.v2 as v2

from datasets import load_dataset

def batch_preparation_img2seq(data):
    images = [sample[0] for sample in data]
    dec_in = [sample[1] for sample in data]
    gt = [sample[2] for sample in data]

    max_image_width = max(128, max([img.shape[2] for img in images]))
    max_image_height = max(256, max([img.shape[1] for img in images]))

    X_train = torch.ones(size=[len(images), 1, max_image_height, max_image_width], dtype=torch.float32)

    for i, img in enumerate(images):
        _, h, w = img.size()
        X_train[i, :, :h, :w] = img
    
    max_length_seq = max([len(w) for w in gt])

    decoder_input = torch.zeros(size=[len(dec_in),max_length_seq])
    y = torch.zeros(size=[len(gt),max_length_seq])

    for i, seq in enumerate(dec_in):
        decoder_input[i, 0:len(seq)-1] = torch.from_numpy(np.asarray([char for char in seq[:-1]]))
    
    for i, seq in enumerate(gt):
        y[i, 0:len(seq)-1] = torch.from_numpy(np.asarray([char for char in seq[1:]]))
    
    return X_train, decoder_input.long(), y.long()

class OMRIMG2SEQDataset(Dataset):
    def __init__(self, augment=False) -> None:
        self.teacher_forcing_error_rate = 0.2
        self.x = None
        self.y = None
        self.augment = augment

        super().__init__()
    
    def apply_teacher_forcing(self, sequence):
        errored_sequence = sequence.clone()
        for token in range(1, len(sequence)):
            if np.random.rand() < self.teacher_forcing_error_rate and sequence[token] != self.padding_token:
                errored_sequence[token] = np.random.randint(0, len(self.w2i))
        
        return errored_sequence

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.augment:
            x = augment(self.x[index])
        else:
            x = convert_img_to_tensor(self.x[index])
        
        y = torch.from_numpy(np.asarray([self.w2i[token] for token in self.y[index]]))
        decoder_input = self.apply_teacher_forcing(y)
        return x, decoder_input, y

    def get_max_hw(self):
        m_width = np.max([img.shape[1] for img in self.x])
        m_height = np.max([img.shape[0] for img in self.x])

        return m_height, m_width
    
    def get_max_seqlen(self):
        return np.max([len(seq) for seq in self.y])

    def vocab_size(self):
        return len(self.w2i)

    def get_gt(self):
        return self.y
    
    def set_dictionaries(self, w2i, i2w):
        self.w2i = w2i
        self.i2w = i2w
        self.padding_token = w2i['<pad>']
    
    def get_dictionaries(self):
        return self.w2i, self.i2w
    
    def get_i2w(self):
        return self.i2w
    
class AMNLTSingleSystem(OMRIMG2SEQDataset):
    IMAGE = "image"
    TRANSCRIPT = "transcription"

    def __init__(self, dataset_name, split, notation, reduce_ratio, augment=False) -> None:
        self.augment = augment
        self.teacher_forcing_error_rate = 0.2
        self.reduce_ratio = reduce_ratio
        self.notation = notation
        self.tensorTransform = transforms.ToTensor()
        self.num_sys_gen = 1
        self.fixed_systems_num = False
        self.fixed_size = None

        self.dataset = load_dataset(dataset_name, split=split)
        self.dataset = self.dataset.map(self.preprocess_gt)
        self.x, self.y = self.dataset[AMNLTSingleSystem.IMAGE], self.dataset[AMNLTSingleSystem.TRANSCRIPT]

    @staticmethod
    def _load_dataset(dataset_name, split):
        ds = load_dataset(dataset_name, split=split)
        return ds[AMNLTSingleSystem.IMAGE], ds[AMNLTSingleSystem.TRANSCRIPT]

    def get_width_avgs(self):
        widths = [self._get_new_img_size(img.size[1], img.size[0])[1] for img in self.x]
        return np.average(widths), np.max(widths), np.min(widths)
    
    def get_max_hw(self):
        max_size = np.max([self._get_new_img_size(img.size[1], img.size[0]) for img in self.x], axis=0)
        return max_size[0], max_size[1]
    
    def _get_new_img_size(self, height, width):
        if self.fixed_size != None:
            width = self.fixed_size[1]
            height = self.fixed_size[0]
        else:
            width = int(np.ceil(min(width, 3056) * self.reduce_ratio))
            height = int(np.ceil(max(height, 256) * self.reduce_ratio))
        return height, width

    def __getitem__(self, index):
        x = v2.functional.to_image(self.x[index])
        y = self.y[index]

        height, width = self._get_new_img_size(x.shape[1], x.shape[2])
        x = v2.functional.resize_image(x, size=[height, width])

        if self.augment:
            x = augment(x)
        else:
            x = convert_img_to_tensor(x)
        
        y = torch.from_numpy(np.asarray([self.w2i[token] for token in y]))
        decoder_input = self.apply_teacher_forcing(y)
        return x, decoder_input, y
    
    def __len__(self):
        return len(self.x)

    def preprocess_gt(self, sample):
        gabc = sample[AMNLTSingleSystem.TRANSCRIPT]
        result = None
        if self.notation == "char":
            gabc = list(gabc.strip())
            
        elif self.notation == "music_aware":
            muaw_gabc = []
            i = 0
            while i < len(gabc):
                if gabc[i:i+3] == "<m>":
                    if i + 3 < len(gabc):
                        muaw_gabc.append(gabc[i:i+4])
                        i += 4
                    else:
                        break
                else:
                    muaw_gabc.append(gabc[i])
                    i += 1
                    
            result = muaw_gabc
            
        elif self.notation == "new_gabc":
            new_gabc = []
            i = 0
            while i < len(gabc):
                if gabc[i] == "(":
                    new_gabc.append(gabc[i])
                    i += 1
                    temp = ""
                    while i < len(gabc) and gabc[i] != ")":
                        temp += gabc[i]
                        i += 1
                        
                    for token in temp.split():
                        new_gabc.append(token)
                        
                    if i < len(gabc):
                        new_gabc.append(gabc[i])
                        i += 1
                else:
                    new_gabc.append(gabc[i])
                    i += 1
            
            result = new_gabc
            
        return {AMNLTSingleSystem.TRANSCRIPT: ['<bos>'] + result + ['<eos>']}

class AMNLTDataset(LightningDataModule):
    def __init__(self, config:ExperimentConfig) -> None:
        super().__init__()
        self.dataset_name = config.dataset_name
        self.vocab_name = config.vocab_name
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.notation = config.transcript_format
        self.reduce_ratio = config.reduce_ratio

        self.train_set = AMNLTSingleSystem(self.dataset_name, "train", self.notation, self.reduce_ratio, augment=True)
        self.val_set = AMNLTSingleSystem(self.dataset_name, "validation", self.notation, self.reduce_ratio)
        self.test_set = AMNLTSingleSystem(self.dataset_name, "test", self.notation, self.reduce_ratio)

        if self.notation == "music_aware":
            vocab_name = self.vocab_name + "_music_aware"
        else:
            vocab_name = self.vocab_name
            
        w2i, i2w = check_and_retrieveVocabulary([self.train_set.get_gt(), self.val_set.get_gt(), self.test_set.get_gt()], "vocab", f"{vocab_name}")

        self.train_set.set_dictionaries(w2i, i2w)
        self.val_set.set_dictionaries(w2i, i2w)
        self.test_set.set_dictionaries(w2i, i2w)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=batch_preparation_img2seq)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)