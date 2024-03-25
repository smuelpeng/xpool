import torch
from PIL import Image
from datasets.video_capture import VideoCapture
from modules.basic_utils import load_json
from collections import defaultdict

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor

def get_itm_score(raw_image, caption):
    img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    txt = text_processors["eval"](caption)
    itm_output = model({"image": img, "text_input": txt}, match_head="itm")
    itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
    # print(
    #     f'The image and text are matched with a probability of {itm_scores[:, 1].item():.3%}')
    itc_score = model({"image": img, "text_input": txt}, match_head='itc')
    # print('The image feature and text feature has a cosine similarity of %.4f' % itc_score)
    return itm_scores[:,1].item(), itc_score


def load_caption(db_file):
    db = load_json(db_file)
    vid2caption = defaultdict(list)
    for annotation in db['sentences']:
        caption = annotation['caption']
        vid = annotation['video_id']
        vid2caption[vid].append(caption)
    return vid2caption


def get_video_caption_scores(video_id):
    video_path = f'{video_prefix}{video_id}.mp4'
    imgs, idxs = VideoCapture.load_frames_from_video(video_path,
                                                     12,
                                                     'uniform')
    itm_scores = torch.zeros( (len(imgs), len(vid2caption_dict[video_id])) )
    itc_scores = torch.zeros( (len(imgs), len(vid2caption_dict[video_id])) )
    for i,img in enumerate(imgs):
        img = Image.fromarray(img)
        for j,caption in enumerate(vid2caption_dict[video_id]):
            itm_score, itc_score = get_itm_score(img, caption)
            itm_scores[i,j] = itm_score
            itc_scores[i,j] = itc_score
    return itm_scores, itc_scores

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, vis_processors, text_processors = load_model_and_preprocess(
    "blip2_image_text_matching", "pretrain", device=device, is_eval=True)
video_prefix = 'preprocess/all_compress/'
vid2caption_dict = load_caption('data/MSRVTT/MSRVTT_data.json')


video_caption_scores = {}
for video_id in vid2caption_dict.keys():
    itm_scores, itc_scores = get_video_caption_scores(video_id)
    video_caption_scores[video_id] = (itm_scores, itc_scores)

    # print(f'Video {video_id} has {len(vid2caption_dict[video_id])} captions')
    # print(f'ITM scores: {itm_scores}')
    # print(f'ITC scores: {itc_scores}')
    # break
torch.save(video_caption_scores, 'data/MSRVTT/video_caption_scores.pt')
        
