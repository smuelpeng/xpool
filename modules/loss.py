import torch.nn as nn
import torch
import torch.nn.functional as F
from config.base_config import Config

class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims, logit_scale):
        """
        Inputs: cosine similarities
            sims: n x n (text is dim-0)
            logit_scale: 1 x 1
        """
        logit_scale = logit_scale.exp()
        logits = sims * logit_scale
        
        t2v_log_sm = F.log_softmax(logits, dim=1)
        t2v_neg_ce = torch.diag(t2v_log_sm)
        t2v_loss = -t2v_neg_ce.mean()

        v2t_log_sm = F.log_softmax(logits, dim=0)
        v2t_neg_ce = torch.diag(v2t_log_sm)
        v2t_loss = -v2t_neg_ce.mean()

        return (t2v_loss + v2t_loss) / 2.0
    
class FrameCapNCELoss(nn.Module):
    def __init__(self, config,tau=0.07):
        super().__init__()   
        self.config = config
        self.tau = tau
        self.num_frames = self.config.num_frames
        self.batch_size = self.config.batch_size
        
    def forward(self, feat_frame, feat_cap, label_frame, label_cap=None):

        label_frame = label_frame.reshape(-1).expand(self.num_frames, self.batch_size).t().reshape(-1)
        if label_cap is None:
            label_cap = label_frame
        
        feat_frame = feat_frame.reshape(-1,  feat_frame.size(-1))
        assert feat_frame.size(0) == self.num_frames * self.batch_size
        feat_frame=F.normalize(feat_frame, p=2, dim=1, eps=1e-12)
        
        feat_cap = feat_cap.reshape(-1,  feat_cap.size(-1))
        assert feat_cap.size(0) == self.num_frames * self.batch_size
        feat_cap=F.normalize(feat_cap, p=2, dim=1, eps=1e-12)
        
        
        sim_matrix = torch.matmul(feat_frame, feat_cap.t()) / self.tau

        neg_mask = ~torch.eq(label_frame.reshape(-1,1), label_cap.reshape(1,-1))
        matrix_mask = neg_mask | torch.eye(neg_mask.size(0), dtype=bool)
        matrix_mask = matrix_mask.cuda()

        sim_matrix = sim_matrix * matrix_mask.float()

        # matrix_mask_exp = sim_matrix.exp() * matrix_mask.float()
        # t2v_log_sm = matrix_mask_exp / matrix_mask_exp.sum(1, keepdim=True)
        # t2v_loss = -torch.diag(t2v_log_sm.log()).mean()
        
        # v2t_log_sm = matrix_mask_exp / matrix_mask_exp.sum(0, keepdim=True)
        # v2t_loss = -torch.diag(v2t_log_sm.log()).mean()

        t2v_log_sm = F.log_softmax(sim_matrix, dim=1)
        t2v_neg_ce = torch.diag(t2v_log_sm)
        t2v_loss = -t2v_neg_ce.mean()
        
        v2t_log_sm = F.log_softmax(sim_matrix, dim=0)
        v2t_neg_ce = torch.diag(v2t_log_sm)
        v2t_loss = -v2t_neg_ce.mean()
        return (t2v_loss + v2t_loss) / 2.0
    
    
class ABRLoss(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.clip_loss = CLIPLoss()
        self.framecap_loss = FrameCapNCELoss(config)
        self.config = config
        
    def forward(self, sims, frame_embeds, cap_embeds, frame_labels, cap_labels, logit_scale ):
        clip_loss = self.clip_loss(sims, logit_scale)
        frame_loss = self.framecap_loss(frame_embeds, cap_embeds, frame_labels, cap_labels)
        print(f'clip_loss: {clip_loss}, frame_loss: {frame_loss}')
        return clip_loss + self.config.framecap_loss_weight * frame_loss
    
        
class LossFactory:
    @staticmethod
    def get_loss(config: Config):
        if config.loss == 'clip':
            return CLIPLoss()
        if config.loss == 'ABR':
            return ABRLoss(config)
        else:
            raise NotImplemented
