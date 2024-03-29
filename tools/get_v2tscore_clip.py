import torch
import clip
from PIL import Image

import torch
from PIL import Image

from modules.basic_utils import load_json
from collections import defaultdict

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor

import cv2
import random
import numpy as np
import torch
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing
# multiprocessing.set_start_method('spawn')   
multiprocessing.set_start_method('spawn', force=True)
from multiprocessing import get_context
import pandas

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

# def get_itm_score(raw_image, caption):
#     with torch.no_grad():
#         img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
#         txt = text_processors["eval"](caption)
#         itm_output = model({"image": img, "text_input": txt}, match_head="itm")
#         itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
#         #print(
#         #     f'The image and text are matched with a probability of {itm_scores[:, 1].item():.3%}')
#         itc_score = model({"image": img, "text_input": txt}, match_head='itc')
#         #print('The image feature and text feature has a cosine similarity of %.4f' % itc_score)
#         del img
#         return itm_scores[:,1].item(), itc_score


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
    

# def get_video_caption_scores(video_id):
#     video_path = f'{video_prefix}{video_id}.mp4'
#     imgs, idxs = VideoCapture.load_frames_from_video(video_path,
#                                                      12,
#                                                      'uniform')
#     itm_scores = torch.zeros( (len(imgs), len(vid2caption_dict[video_id])) )
#     itc_scores = torch.zeros( (len(imgs), len(vid2caption_dict[video_id])) )
#     for i,img in enumerate(imgs):
#         img = Image.fromarray(img)
#         for j,caption in enumerate(vid2caption_dict[video_id]):
#             # itm_score, itc_score = get_itm_score(img, caption)
#             with torch.no_grad():
#                 img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
#                 txt = text_processors["eval"](caption)
#                 itm_output = model({"image": img, "text_input": txt}, match_head="itm")
#                 itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
#                 #print(
#                 #     f'The image and text are matched with a probability of {itm_scores[:, 1].item():.3%}')
#                 itc_score = model({"image": img, "text_input": txt}, match_head='itc')
#                 #print('The image feature and text feature has a cosine similarity of %.4f' % itc_score)
#                 del img
#                 return itm_scores[:,1].item(), itc_score
#             itm_scores[i,j] = itm_score
#             itc_scores[i,j] = itc_score
#     # print(itm_scores.shape)
#     return itm_scores, itc_scores


def task_list(video_ids, gpu_id):
    # 加载预训练的CLIP模型
    # gpu_id = 7 if gpu_id >3  else 3
    gpu_id = 3
    device = torch.device("cuda", gpu_id) if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    video_prefix = 'preprocess/all_compress/'
    # vid2caption_dict = load_caption('data/MSRVTT/MSRVTT_data.json')
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
                image_features = model.encode_image(raw_img)
                for j, caption in enumerate(vid2caption_dict[video_id]):
                    text = clip.tokenize(caption).to(device)
                    text_features = model.encode_text(text)
                    similarity = torch.cosine_similarity(image_features, text_features)
                    clip_scores[i, j] = similarity.item()
                del raw_img
        clip_res[video_id] = clip_scores.cpu().numpy()
    return clip_res

if __name__ == '__main__':
    # vid2caption_dict = load_caption('data/MSRVTT/MSRVTT_data.json')
    vid2caption_dict = load_testcaption_csv('data/MSRVTT/MSRVTT_JSFUSION_test.csv')
    clip_result_dict = {}
    video_ids = list(vid2caption_dict.keys())
    num_gpus = torch.cuda.device_count()
    with get_context("spawn").Pool(num_gpus) as pool:
        n = len(video_ids)
        procs = []
        num_per_task = n // num_gpus + 1
        for i in range(num_gpus):
            procs.append(
                pool.apply_async(task_list, args=(
                    video_ids[i*num_per_task:(i+1)*num_per_task], i,))
            )
        for proc in procs:
            clip_res = proc.get()
            clip_result_dict.update(clip_res)
    pool.close()
    pool.join()
    torch.save(clip_result_dict, 'data/MSRVTT/test_CLIP_result_dict.pt')
    # video_caption_scores = {}
    # for video_id in tqdm(vid2caption_dict.keys()):
    #     itm_scores, itc_scores = get_video_caption_scores(video_id)
    #     video_caption_scores[video_id] = (itm_scores, itc_scores)
    # torch.save(video_caption_scores, 'data/MSRVTT/video_caption_scores.pt')
