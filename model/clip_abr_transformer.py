import torch
import torch.nn as nn
from config.base_config import Config
from modules.transformer import Transformer,Perceiver

class CLIPABRTransformer(nn.Module):
    def __init__(self, config: Config):
        super(CLIPABRTransformer, self).__init__()
        self.config = config
        
        if self.config.huggingface:
            from transformers import CLIPModel
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        else:
            from model.clip_model import load_clip
            self.clip = load_clip(config.clip_arch)

        config.pooling_type = 'transformer'
        self.pool_frames = Transformer(config)
        if config.visual_proj == 'perceiver':
            self.visual_proj = Perceiver(config)
        else:
            self.visual_proj = nn.Identity()


    def forward(self, data, return_all_frames=False, frame_caption=False):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        if frame_caption:
            frame_cap_data = data['frame_caption']

        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        if self.config.huggingface:
            text_features = self.clip.get_text_features(**text_data)
            video_features = self.clip.get_image_features(video_data)
            if frame_caption:
                cap_features = self.clip.get_text_features(**frame_cap_data)
        else:
            text_features = self.clip.encode_text(text_data)
            video_features = self.clip.encode_image(video_data)
            if frame_caption:
                cap_features = self.clip.encode_text(frame_cap_data)

        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)
        video_features_proj = self.visual_proj(video_embeds=video_features)
        # frame_features = video_features
        video_features_pooled = self.pool_frames(text_features, video_features)
        if return_all_frames:
            return text_features, video_features, video_features_pooled
        if frame_caption:
            return text_features, video_features_pooled, cap_features, video_features_proj #frame_features
        else:
            return text_features, video_features_pooled
    
