import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn

def set_all_seeds(SEED, deterministic=False):
    """Set random seed.
    Args:
        SEED(int): Seed to be used.
        deterministic(bool): Whether to set the deterministic option for CUDNN backend. Default: False
    """
    # REPRODUCIBILITY
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    cudnn.benchmark = True
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

class Trainsave():
    def __init__(self):
        self.best_loss = 1000
        self.best_loss_epoch = None
        self.best_loss_result = None

        self.best_cindex = 0
        self.best_cindex_epoch = None
        self.best_cindex_result = None

    def save_loss(self, loss, epoch, result_all):
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_loss_epoch = epoch
            self.best_loss_result = result_all
            return 'best_loss_model.pth'
        else:
            return None

    def save_cindex(self, Cindex, epoch, result_all):
        if Cindex > self.best_cindex:
            self.best_cindex = Cindex
            self.best_cindex_epoch = epoch
            self.best_cindex_result = result_all
            return 'best_cindex_model.pth'
        else:
            return None