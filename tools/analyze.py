import torch
import numpy as np
import torch.nn as nn
import json
import os
from tqdm import tqdm
from modules.metrics import sim_matrix_training, sim_matrix_inference, generate_embeds_per_video_id
from tqdm import tqdm

def _valid_epoch_step(self, epoch, step, num_steps):
    """
    Validate at a step when training an epoch at a certain step
    :return: A log that contains information about validation
    """
    total_val_loss = 0.0
    text_embed_arr = []
    vid_embed_arr = []
    all_vid_ids = []
    
    with torch.no_grad():
        for _, data in tqdm(enumerate(self.valid_data_loader)):
            if self.tokenizer is not None:
                data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
            if isinstance(data['text'], torch.Tensor):
                data['text'] = data['text'].to(self.device)
            else:
                data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}

            data['video'] = data['video'].to(self.device)
            
            text_embed, vid_embed, vid_embed_pooled = self.model(data, return_all_frames=True)
            text_embed_arr.append(text_embed.cpu())
            vid_embed_arr.append(vid_embed.cpu())
            sims_batch = sim_matrix_training(text_embed, vid_embed_pooled, self.pooling_type)

            curr_loss = self.loss(sims_batch, self.model.clip.logit_scale)
            total_val_loss += curr_loss.item()

            for v_id in data['video_id']:
                all_vid_ids.append(v_id)
            
        text_embeds = torch.cat(text_embed_arr)
        vid_embeds = torch.cat(vid_embed_arr)

        # Since we have all pairs, remove duplicate videos when there's multiple captions per video
        vid_embeds_per_video_id = {}
        for idx, v_id in enumerate(all_vid_ids):
            if v_id not in vid_embeds_per_video_id:
                vid_embeds_per_video_id[v_id] = vid_embeds[idx]
        
        vid_embeds = torch.stack([vid_embeds_per_video_id[v_id] for v_id in vid_embeds_per_video_id])
            
        # Pool frames for inference once we have all texts and videos
        self.model.pool_frames.cpu()
        vid_embeds_pooled = self.model.pool_frames(text_embeds, vid_embeds)
        self.model.pool_frames.cuda()

        text_embeds_per_video_id, vid_embeds_pooled_per_video_id = generate_embeds_per_video_id(text_embeds, 
                vid_embeds_pooled, all_vid_ids, self.pooling_type)
        
        sims = sim_matrix_inference(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, self.pooling_type)

        total_val_loss = total_val_loss / len(self.valid_data_loader)

        metrics = self.metrics
        res = metrics(sims)
        
        # Compute window metrics
        for m in res:
            self.window_metric[m].append(res[m])

        # Compute average of window metrics
        for m in self.window_metric:
            res[m + "-window"] = np.mean(self.window_metric[m])

        print(f"-----Val Epoch: {epoch}, dl: {step}/{num_steps}-----\n",
                f"R@1: {res['R1']} (window: {res['R1-window']})\n", 
                f"R@5: {res['R5']} (window: {res['R5-window']})\n", 
                f"R@10: {res['R10']} (window: {res['R10-window']})\n",
                f"MedR: {res['MedR']} (window: {res['MedR-window']})\n",
                f"MeanR: {res['MeanR']} (window: {res['MeanR-window']})\n",
                f"Loss: {total_val_loss}")
        
        res['loss_val'] =  total_val_loss

        if self.writer is not None:
            for m in res:
                self.writer.add_scalar(f'val/{m}', res[m], self.global_step)

        return res
