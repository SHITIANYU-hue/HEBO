import numpy as np
from bo.base import TestFunction
from task.tools import Absolut
import torch
import json
class BOTask(TestFunction):
    """
    BO Task Class
    """
    # this should be changed if we are tackling a mixed, or continuous problem, for e.g.
    problem_type = 'categorical'
    def __init__(self,
                 device,
                 n_categories,
                 seq_len,
                 bbox=None,
                 normalise=True,
                 mean=None,
                 std=None):
        super(BOTask, self).__init__(normalise)
        self.device = device
        self.bbox = bbox
        self.n_vertices = n_categories
        self.config = self.n_vertices
        self.dim = seq_len
        self.dataset='/home/tianyu/code/biodrug/unify-length/data_ADQ_A.json'
        self.categorical_dims = np.arange(self.dim)
        if self.bbox['tool'] == 'Absolut':
            self.fbox = Absolut(self.bbox)
        else:
            assert 0,f"{self.config['tool']} Not Implemented"


    def compute(self, x):
        '''
        x: categorical vector
        '''
        # we can remove this line and load correspoinding energy from our dataset
        energy, _ = self.fbox.Energy_custom(x,data=json.load(f'{self.dataset}')) ## x will not be list but will be sequecne 
        
        energy = torch.tensor(energy, dtype=torch.float32).to(self.device)
        return energy

    def idx_to_seq(self, x):
        seqs = []
        for seq in x:
            seqs.append(''.join(self.fbox.idx_to_AA[int(aa)] for aa in seq))
        return seqs