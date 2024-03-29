import os
import cv2
import sys
import torch
import random
import itertools
import numpy as np
import pandas as pd
import ujson as json
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from modules.basic_utils import load_json
from torch.utils.data import Dataset
from config.base_config import Config

db_file = 'data/MSRVTT/MSRVTT_data.json'
test_csv = 'data/MSRVTT/MSRVTT_JSFUSION_test.csv'


def load_caption(db_file):
    db = load_json(db_file)
    vid2caption = defaultdict(list)
    for annotation in db['sentences']:
        caption = annotation['caption']
        vid = annotation['video_id']
        vid2caption[vid].append(caption)
    return vid2caption
vid2caption = load_caption(db_file)
def load_testcaption_csv(db_file):
    # db = load_json(db_file)
    db = pandas.read_csv(db_file)
    vid2caption = defaultdict(list)
    for i in range(len(db)):
        caption = db['sentence'][i]
        vid = db['video_id'][i]
        vid2caption[vid].append(caption)
    return vid2caption

