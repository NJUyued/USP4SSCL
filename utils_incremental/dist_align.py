import torch
import numpy as np

class DistAlignQueueHook():
    """
    Distribution Alignment Hook for conducting distribution alignment
    """
    def __init__(self, num_classes, queue_length=128, p_target_type='uniform', p_target=None):
        super().__init__()
        self.num_classes = num_classes
        self.queue_length = queue_length

        # p_target
        self.p_target_ptr, self.p_target = self.set_p_target(p_target_type, p_target)    
        print('distribution alignment p_target:', self.p_target.mean(dim=0))
        # p_model
        self.p_model = torch.zeros(self.queue_length, self.num_classes, dtype=torch.float)
        self.p_model_ptr = torch.zeros(1, dtype=torch.long)

    @torch.no_grad()
    def dist_align(self, probs_x_ulb, probs_x_lb=None):
        """
        Args:
            probs_x_ulb: unlabeled batch probs
        """

        # update queue
        self.update_p(probs_x_ulb, probs_x_lb)

        # dist align
        probs_x_ulb_aligned = probs_x_ulb * (self.p_target.mean(dim=0) + 1e-6) / (self.p_model.mean(dim=0) + 1e-6)
        probs_x_ulb_aligned = probs_x_ulb_aligned / probs_x_ulb_aligned.sum(dim=-1, keepdim=True)
        return probs_x_ulb_aligned
    
    @torch.no_grad()
    def update_p(self, probs_x_ulb, probs_x_lb):
        # TODO: think better way?
        # check device
        if not self.p_target.is_cuda:
            self.p_target = self.p_target.to(probs_x_ulb.device)
            if self.p_target_ptr is not None:
                self.p_target_ptr = self.p_target_ptr.to(probs_x_ulb.device)
        
        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(probs_x_ulb.device)
            self.p_model_ptr = self.p_model_ptr.to(probs_x_ulb.device)

        probs_x_ulb = probs_x_ulb.detach()
        p_model_ptr = int(self.p_model_ptr)
        self.p_model[p_model_ptr] = probs_x_ulb.mean(dim=0)
        self.p_model_ptr[0] = (p_model_ptr + 1) % self.queue_length

        if self.p_target_ptr is not None:
            assert probs_x_lb is not None
            p_target_ptr = int(self.p_target_ptr)
            self.p_target[p_target_ptr] = probs_x_lb.mean(dim=0)
            self.p_target_ptr[0] = (p_target_ptr + 1) % self.queue_length
    
    def set_p_target(self, p_target_type='uniform', p_target=None):
        assert p_target_type in ['uniform', 'gt', 'model']

        # p_target
        p_target_ptr = None
        if p_target_type == 'uniform':
            p_target = torch.ones(self.queue_length, self.num_classes, dtype=torch.float) / self.num_classes
        elif p_target_type == 'model':
            p_target = torch.zeros((self.queue_length, self.num_classes), dtype=torch.float)
            p_target_ptr =  torch.zeros(1, dtype=torch.long)
        else:
            assert p_target is not None
            if isinstance(p_target, np.ndarray):
                p_target = torch.from_numpy(p_target)
            p_target = p_target.unsqueeze(0).repeat((self.queue_length, 1))
        
        return p_target_ptr, p_target