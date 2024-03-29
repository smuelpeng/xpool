import os
import torch
import random
import numpy as np
from config.all_config import AllConfig
from model.model_factory import ModelFactory
import pandas
from modules.basic_utils import load_json
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing
# multiprocessing.set_start_method('spawn')   
multiprocessing.set_start_method('spawn', force=True)
from multiprocessing import get_context
import pandas
import cv2
from PIL import Image
import clip

class VideoCapture:

    @staticmethod
    def load_frames_from_video(video_path,
                               num_frames,
                               sample='rand'):
        """
            video_path: str/os.path
            num_frames: int - number of frames to sample
            sample: 'rand' | 'uniform' how to sample
            returns: frames: torch.tensor of stacked sampled video frames 
                             of dim (num_frames, C, H, W)
                     idxs: list(int) indices of where the frames where sampled
        """
        cap = cv2.VideoCapture(video_path)
        assert (cap.isOpened()), video_path
        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # get indexes of sampled frames
        acc_samples = min(num_frames, vlen)
        intervals = np.linspace(
            start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []

        # ranges constructs equal spaced intervals (start, end)
        # we can either choose a random image in the interval with 'rand'
        # or choose the middle frame with 'uniform'
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
        else:  # sample == 'uniform':
            frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

        frames = []
        for index in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()
            if not ret:
                n_tries = 5
                for _ in range(n_tries):
                    ret, frame = cap.read()
                    if ret:
                        break
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                raise ValueError

        while len(frames) < num_frames:
            frames.append(frames[-1].clone())
        cap.release()
        return frames, frame_idxs

def load_caption(db_file):
    db = load_json(db_file)
    vid2caption = defaultdict(list)
    for annotation in db['sentences']:
        caption = annotation['caption']
        vid = annotation['video_id']
        vid2caption[vid].append(caption)
    return vid2caption

def load_testcaption_csv(db_file):
    # db = load_json(db_file)
    db = pandas.read_csv(db_file)
    vid2caption = defaultdict(list)
    for i in range(len(db)):
        caption = db['sentence'][i]
        vid = db['video_id'][i]
        vid2caption[vid].append(caption)
    return vid2caption
    

def extraxct_clip_features(model, imgs, v2c_dict, device, preprocess,tokenizer):
    with torch.no_grad():
        raw_imgs = [preprocess(Image.fromarray(img)).to(device) for img in imgs]
        raw_imgs = torch.stack(raw_imgs)
        image_features = model.clip.get_image_features(raw_imgs)
        text_features = []
        for j, caption in enumerate(v2c_dict):
            text = tokenizer(caption, return_tensors='pt', padding=True, truncation=True)
            text = {k: v.to(device) for k, v in text.items()}
            text_feat = model.clip.get_text_features(**text)
            text_features.append(text_feat)
        del raw_imgs
        text_features = torch.stack(text_features)
    return image_features, text_features


def task_list2(video_ids, gpu_id):
    config = AllConfig()
    # gpu_id = random.sample([1,2,3,6,7], 1)[0]
    gpus = [4,5,3,6,7]
    gpu_id = gpus[gpu_id%len(gpus)]
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    device = torch.device("cuda", gpu_id) if torch.cuda.is_available() else "cpu"
    print(device)
    _, preprocess = clip.load("ViT-B/32", device=device)
    model = ModelFactory.get_model(config).to(device)
    checkpoint_path = "/data1/zhipeng/workspace/github/xpool/outputs/xpoolMR9k_compress/model_best.pth"
    # checkpoint_path = "/data1/zhipeng/workspace/github/xpool/outputs/xpoolMR9k_compress/checkpoint-epoch5.pth"
    print("Loading checkpoint: {} ...".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)

    video_prefix = 'preprocess/all_compress/'
    vid2caption_dict = load_testcaption_csv('data/MSRVTT/MSRVTT_JSFUSION_test.csv')
    # print('load caption finished!!')    
    clip_res = {}
    image_feats = []
    text_feats = []
    for video_id in tqdm(video_ids):
        video_path = f'{video_prefix}{video_id}.mp4'
        imgs, idxs = VideoCapture.load_frames_from_video(video_path,
                                                         12,
                                                         'uniform')
        img_feat,text_feat = extraxct_clip_features(model, imgs, vid2caption_dict[video_id], device, preprocess,tokenizer)
        image_feats.append(img_feat)
        text_feats.append(text_feat)
    image_feats = torch.cat(image_feats).cpu().numpy()
    text_feats = torch.cat(text_feats).cpu().numpy()
    return image_feats, text_feats



def task_list(video_ids, gpu_id):
    config = AllConfig()
    gpus = [4,5,3,6,7]
    gpu_id = gpus[gpu_id%len(gpus)]
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    device = torch.device("cuda", gpu_id) if torch.cuda.is_available() else "cpu"
    print(device)
    _, preprocess = clip.load("ViT-B/32", device=device)
    model = ModelFactory.get_model(config).to(device)
    checkpoint_path = "/data1/zhipeng/workspace/github/xpool/outputs/xpoolMR9k_compress/model_best.pth"
    # checkpoint_path = "/data1/zhipeng/workspace/github/xpool/outputs/xpoolMR9k_compress/checkpoint-epoch5.pth"
    print("Loading checkpoint: {} ...".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)

    video_prefix = 'preprocess/all_compress/'
    vid2caption_dict = load_testcaption_csv('data/MSRVTT/MSRVTT_JSFUSION_test.csv')
    # print('load caption finished!!')    
    clip_res = {}
    for video_id in tqdm(video_ids):
        video_path = f'{video_prefix}{video_id}.mp4'
        imgs, idxs = VideoCapture.load_frames_from_video(video_path,
                                                         12,
                                                         'uniform')
        with torch.no_grad():
            clip_scores = torch.zeros((len(imgs), len(vid2caption_dict[video_id])))
            cap_embeds = {}
            for i, img in enumerate(imgs):
                raw_img = Image.fromarray(img)
                raw_img = preprocess(raw_img).unsqueeze(0).to(device)
                # image_features = model.encode_image(raw_img)
                image_features = model.clip.get_image_features(raw_img)                    
                for j, caption in enumerate(vid2caption_dict[video_id]):
                    text = tokenizer(caption, return_tensors='pt', padding=True, truncation=True)
                    text = {k: v.to(device) for k, v in text.items()}
                    text_features = model.clip.get_text_features(**text)
                    similarity = torch.cosine_similarity(image_features, text_features)
                    clip_scores[i, j] = similarity.item()
                del raw_img
        clip_res[video_id] = clip_scores.cpu().numpy()
    return clip_res

if __name__ == '__main__':
    # vid2caption_dict = load_caption('data/MSRVTT/MSRVTT_data.json')
    vid2caption_dict = load_testcaption_csv('data/MSRVTT/MSRVTT_JSFUSION_test.csv')
    clip_result_dict = {}
    video_ids = list(vid2caption_dict.keys())#[:100]
    num_gpus = torch.cuda.device_count()
    # num_gpus = 1
    with get_context("spawn").Pool(num_gpus) as pool:
        n = len(video_ids)
        procs = []
        num_per_task = n // num_gpus + 1
        for i in range(num_gpus):
            procs.append(
                pool.apply_async(task_list2, args=(
                    video_ids[i*num_per_task:(i+1)*num_per_task], i,))
            )
        for proc in procs:
            # clip_res = proc.get()
            # clip_result_dict.update(clip_res)
            image_feats, text_feats = proc.get()
            clip_result_dict.setdefault('image_feats', []).append(image_feats)
            clip_result_dict.setdefault('text_feats', []).append(text_feats)
    pool.close()
    pool.join()
    clip_result_dict['image_feats'] = np.concatenate(clip_result_dict['image_feats'], axis=0)
    clip_result_dict['text_feats'] = np.concatenate(clip_result_dict['text_feats'], axis=0)
    torch.save(clip_result_dict, 'data/MSRVTT/test_XPOOL_feature_dict.pt')

