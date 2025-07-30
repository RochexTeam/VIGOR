import json
# from typing import io

import PIL
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor

import re
from torch.nn import functional as F
import torch.nn as nn
import os
import torch
import torch.nn as nn

import torch
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class MMDataset(torch.utils.data.Dataset):

    def __init__(self, path, is_eval=False):
        self.data = json.load(open(path, encoding='utf-8'))
        self.generator_text_maxlength = 512
        self.answer_maxlength = 64
        self.pad_id = 0
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.is_eval = is_eval

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data = self.data[i]
        return data

    def collate_fn(self, data):

        for dialogue in data:
            text_context = ""
            str = ""
            for i, utt in enumerate(dialogue["history"]):
                if text_context == "":
                    text_context = utt
                else:
                    text_context += " " + utt
                if i == len(dialogue["history"])-1:
                    user_question = utt
                
                if i % 2 == 0:
                    if str == "":
                        str += "<user> " + utt
                    else:
                        str += " <user> " + utt
                else:
                    str += " <system> " + utt
            
            context_token = str
            
            knowledge = dialogue["knowledge"]

            response = dialogue["response"]

            image_response = dialogue["image"] 

        return text_context, knowledge, response, image_response, user_question, context_token


if __name__ == "__main__":
    train_set = MMDataset(path="../../../data/m-rest/used/train.json")
    train_dataloader = torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=1,
                                                   collate_fn=train_set.collate_fn,
                                                   shuffle=True,
                                                   drop_last=False)
    progress = tqdm(total=len(train_dataloader))
    for idx, (text_context, knowledge, response, image_response, user_question, context_token) in enumerate(
            train_dataloader):
        pass
