import torch
import torch.nn as nn
from config.base_config import Config
from modules.transformer import Transformer


class CLIPITMTransformer(nn.Module):
    def __init__(self, config: Config):
        super(CLIPITMTransformer, self).__init__()
        self.config = config
        
        if self.config.huggingface:
            from transformers import CLIPModel
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        else:
            from model.clip_model import load_clip
            self.clip = load_clip(config.clip_arch)

        config.pooling_type = 'transformer'
        self.pool_frames = Transformer(config)
        # self.itm_module = 
        # self.visual_proj =

    def build_text_encoder(self):
        """build text_encoder and possiblly video-to-text multimodal fusion encoder.
        Returns: nn.Module. The text encoder

        """
        # encoder_name = self.config.model.get('itm_module', '')
        self.text_encoder = self.build
        
    def build_bert(model_config, pretrain, checkpoint):
        """build text encoder.

        Args:
            model_config (dict): model config.
            pretrain (bool): Whether to do pretrain or finetuning.
            checkpoint (bool): whether to do gradient_checkpointing.

        Returns: TODO

        """
        bert_config = BertConfig.from_json_file(model_config.text_encoder.config)
        bert_config.encoder_width = model_config.vision_encoder.d_model
        bert_config.gradient_checkpointing = checkpoint
        bert_config.fusion_layer = model_config.text_encoder.fusion_layer

        if not model_config.multimodal.enable:
            bert_config.fusion_layer = bert_config.num_hidden_layers

        if pretrain:
            text_encoder, loading_info = BertForMaskedLM.from_pretrained(
                model_config.text_encoder.pretrained,
                config=bert_config,
                output_loading_info=True,
            )
        else:
            text_encoder, loading_info = BertModel.from_pretrained(
                model_config.text_encoder.pretrained,
                config=bert_config,
                add_pooling_layer=False,
                output_loading_info=True,
            )
        return text_encoder
    
    def forward(self, data, return_all_frames=False):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        
        if self.config.huggingface:
            text_features = self.clip.get_text_features(**text_data)
            video_features = self.clip.get_image_features(video_data)
        else:
            text_features = self.clip.encode_text(text_data)
            video_features = self.clip.encode_image(video_data)
   
        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)

        video_features_pooled = self.pool_frames(text_features, video_features)
            
        if return_all_frames:
            return text_features, video_features, video_features_pooled

        return text_features, video_features_pooled



