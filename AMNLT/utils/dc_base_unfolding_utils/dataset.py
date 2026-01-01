import json
import os
import re
import sys
from PIL import Image

import torch
from torch.utils.data import Dataset

from datasets import load_dataset

from AMNLT.utils.dc_base_unfolding_utils.data_preprocessing import (
    preprocess_image,
    preprocess_transcript,
)

from gabcparser import GabcParser
from gabcparser.utils import separate_lyrics_music
import gabcparser.grammars as grammars

from enum import Enum

################################################################################################ Single-source:

class Separation(Enum):
    NONE = None
    LYRIC = "lyric"
    MUSIC = "music"

# ds_name = name of HuggingFace dataset
# encoding_type = ["char", "new_gabc", "music_aware"]
# transcription_separation = [None, "lyric", "music"]
def make_vocabulary(ds_name, encoding_type, transcription_separation: Separation = Separation.NONE):
        vocab = set()
        ds = load_dataset(ds_name)

        if transcription_separation != Separation.NONE:
            if ds_name == "PRAIG/GregoSynth_staffLevel":
                gabc_variation = grammars.GABC
            elif ds_name == "PRAIG/Solesmes_staffLevel":
                gabc_variation = grammars.S_GABC
            elif ds_name in ["PRAIG/Einsiedeln_staffLevel", "PRAIG/Salzinnes_staffLevel"]:
                gabc_variation = grammars.MEI_GABC
            else:
                raise ValueError(f"Could not infer gabc variation for the unknown dataset '{ds_name}'")
        
            parser = GabcParser.load_parser(gabc_variation)

            error_indices = []
            for split in ds.keys():
                ds[split]["transcription_lyric"] = []
                ds[split]["transcription_music"] = []
                for i,transcript in enumerate(ds[split]["transcription"]):
                    lyric, music = separate_lyrics_music.separate_lyrics_music(transcript, parser)
                    if lyric is None or music is None:
                        error_indices.append(i)
                    ds[split]["transcription_lyric"] = [].append(lyric)
                    ds[split]["transcription_music"] = [].append(music)
            if len(error_indices) > 0:
                print(f"Could not separate lyrics and musics for the following samples: {error_indices}")

        key = CTCDataset.TRANSCRIPT
        if transcription_separation == Separation.LYRIC:
            key = "transcription_lyric"
        elif transcription_separation == Separation.MUSIC:
            key = "transcription_music"

        if (encoding_type == "char") or (
            encoding_type == "new_gabc" 
            and ds_name in ["PRAIG/Einsiedeln_staffLevel", "PRAIG/Salzinnes_staffLevel"] 
            and transcription_separation == Separation.LYRIC
            ):
            for split in ds.keys():
                for y in ds[split][key]:
                    vocab.update(y.strip())
        elif encoding_type == "music_aware":
            for split in ds.keys():
                for y in ds[split][CTCDataset.TRANSCRIPT]:
                    # Leer todo el contenido del archivo
                    content = y.strip()
                    i = 0
                    while i < len(content):
                        if content[i:i+3] == "<m>":  # Check if the token starts with the musical tag
                            # If so, extract the musical character with the tag
                            if i + 3 < len(content):  # Ensure it doesn't go out of range
                                vocab.add("<m>" + content[i+3])
                                i += 3  # Skip to the next character after <m>
                            else:
                                break
                        else:
                            # Add the normal character to the vocabulary
                            vocab.add(content[i])
                        i += 1
        elif encoding_type == "new_gabc" and ds_name in ["PRAIG/Einsiedeln_staffLevel", "PRAIG/Salzinnes_staffLevel"]:
            for split in ds.keys():
                for y in ds[split][CTCDataset.TRANSCRIPT]:
                    content = y
                    i = 0
                    while i < len(content):
                        if content[i] == '(':
                            # Tokenize '('
                            vocab.add(content[i])
                            i += 1
                            # Collect the characters inside the parentheses
                            temp = ''
                            while i < len(content) and content[i] != ')':
                                temp += content[i]
                                i += 1
                            # Split the collected characters by spaces and add them as tokens
                            for token in temp.split():
                                vocab.add(token)
                            # Tokenize ')'
                            if i < len(content):
                                vocab.add(content[i])
                                i += 1
                        else:
                            # Tokenize each character outside parentheses
                            vocab.add(content[i])
                            i += 1
                            
        elif encoding_type == "new_gabc" and ds_name in ["PRAIG/Einsiedeln_staffLevel", "PRAIG/Salzinnes_staffLevel"] and transcription_separation == Separation.MUSIC:
            for split in ds.keys():
                for y in ds[split][key]:
                    content = y.strip()
                    i = 0
                    temp = ''
                    while i < len(content):
                        if content[i] == " " or content[i] == ")":
                            if temp:
                                vocab.add(temp)
                            vocab.add(content[i])
                            temp = ''
                        elif content[i] == "(":
                            vocab.add(content[i])
                        else:
                            temp += content[i]
                        i += 1
                    if temp:
                        vocab.add(temp)
                    
        vocab = sorted(vocab)

        w2i = {}
        i2w = {}
        for i, w in enumerate(vocab):
            w2i[w] = i + 1
            i2w[i + 1] = w
        w2i["<PAD>"] = 0
        i2w[0] = "<PAD>"

        return w2i, i2w

class CTCDataset(Dataset):
    IMAGE = "image"
    TRANSCRIPT = "transcription"

    def __init__(
        self,
        ds_name,
        split,
        model_name,
        train=True,
        da_train=False,
        width_reduction=2,
        encoding_type="char",
    ):
        self.name = ds_name
        self.split = split
        self.model_name = model_name
        self.train = train
        self.da_train = da_train
        self.width_reduction = width_reduction
        self.encoding_type = encoding_type

        # Get image paths and transcripts
        self.X, self.Y = self.load_data(ds_name, split)
        
        self.printbatch = False

        # Check and retrieve vocabulary
        vocab_name = f"w2i_{self.encoding_type}.json"
        vocab_folder = os.path.join(os.path.join("data", self.name), "vocab")
        os.makedirs(vocab_folder, exist_ok=True)
        self.w2i_path = os.path.join(vocab_folder, vocab_name)
        self.w2i, self.i2w = self.check_and_retrieve_vocabulary()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.model_name == "trocr":
            # TrOCR expects PIL image resized to 224x224 and tokenized label
            image = Image.open(self.X[idx]).convert("RGB").resize((224, 224))
            label = open(self.Y[idx], "r", encoding="utf-8").read().strip()

            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
            label_ids = self.processor.tokenizer(label, return_tensors="pt").input_ids.squeeze()
            return {"pixel_values": pixel_values, "labels": label_ids}

        
        unfolding = False
        if self.model_name == "fcn" or self.model_name == "crnnunfolding" or self.model_name == "cnnt2d" or self.model_name == "van":
            unfolding = True
            
        if "gregoeli" in self.name and self.model_name in ["fcn", "crnnunfolding", "cnnt2d", "van"]:
            reduce = True
        else:
            reduce = False

        # CTC Training setting
        x = preprocess_image(self.X[idx], unfolding=unfolding, reduce=reduce)
        y = preprocess_transcript(self.Y[idx], self.w2i, self.name, self.encoding_type)
        
        img_id = f"{self.split}_{idx}"
        
        if self.train:
            # x.shape = [channels, height, width]
            if self.model_name == "fcn" or self.model_name == "crnnunfolding" or self.model_name == "cnnt2d" or self.model_name == "van":
                return x, (x.shape[2] // 8) * (x.shape[1] // 32), y, len(y), img_id
            elif self.model_name == "crnn":
                return x, x.shape[2] // self.width_reduction, y, len(y), img_id
            elif self.model_name == "ctc_van":
                return x, x.shape[2] // 8, y, len(y), img_id
            
        if self.printbatch:
            try:
                # Open the image from the provided path
                image = self.X[img_id]
                # Save the image to a new location, e.g., as 'output_image.png'
                image.save("output_image.png")
                print(f"Image saved as 'output_image.png'")
            except Exception as e:
                print(f"Error loading or saving image: {e}")
            
            # Save the transcript (y) as a text file
            with open("output_transcript.txt", "w") as f:
                f.write(str(y))
            
            print(f"Transcript saved as 'output_transcript.txt'")
            
            # Set printbatch to False after saving
            self.printbatch = False
            
        return x, y, img_id
    
    def get_mx_hw(self):
        
        reduce = False
        
        if self.model_name == "cnnt2d" and self.name.startswith("gregoeli"):
            reduce = True
        
        max_height = max_width = 0
        for img in self.X:
            x = preprocess_image(img, unfolding=True, reduce=reduce)
            max_height = max(max_height, x.shape[1])
            max_width = max(max_width, x.shape[2])
        return max_height, max_width

    def load_data(self, ds_name, split):
        ds = load_dataset(ds_name, split=split)
        return ds[self.IMAGE], ds[self.TRANSCRIPT]

    def check_and_retrieve_vocabulary(self):
        w2i = {}
        i2w = {}

        if os.path.isfile(self.w2i_path):
            with open(self.w2i_path, "r") as file:
                w2i = json.load(file)
            i2w = {v: k for k, v in w2i.items()}
        else:
            w2i, i2w = self.make_vocabulary()
            with open(self.w2i_path, "w") as file:
                json.dump(w2i, file)

        return w2i, i2w
    
    def make_vocabulary(self):
        # TODO: add support for transcription separation
        return make_vocabulary(self.name, self.encoding_type)