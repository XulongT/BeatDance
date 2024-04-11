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

class DBCLIPLoss(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        # self.threshold = 0.05
        # self.threshold = -0.1054
        self.threshold = threshold
        self.k = 5

    def forward(self, sims, logit_scale):
        """
        Inputs: cosine similarities
            sims: n x n (text is dim-0)
            logit_scale: 1 x 1
        """

        n = sims.shape[0]
        eye_m = torch.eye(n, dtype=torch.bool).cuda()
        logit_scale = logit_scale.exp()

        row_sims = sims.clone()
        row_diag = torch.diag(sims).view(sims.shape[0], -1)
        row_mask = (~eye_m) & (row_sims > (row_diag * self.threshold)) & (row_sims > 0.05)
        row_sims[row_mask] = -1e5

        logits = row_sims * logit_scale
        
        t2v_log_sm = F.log_softmax(logits, dim=1)
        t2v_neg_ce = torch.diag(t2v_log_sm)
        t2v_loss = -t2v_neg_ce.mean()

        column_sims = sims.clone()
        column_diag = torch.diag(sims)
        mask = (~eye_m) &  (column_sims > (column_diag * self.threshold)) & (column_sims > 0.05)
        column_sims[mask] = -1e5

        logits = column_sims * logit_scale

        v2t_log_sm = F.log_softmax(logits, dim=0)
        v2t_neg_ce = torch.diag(v2t_log_sm)
        v2t_loss = -v2t_neg_ce.mean()

        return (t2v_loss + v2t_loss) / 2.0



class LossFactory:
    @staticmethod
    def get_loss(config: Config):
        if config.loss == 'clip':
            return CLIPLoss()
        if config.loss == 'dbclip':
            return DBCLIPLoss(config.threshold)
        else:
            raise NotImplemented
