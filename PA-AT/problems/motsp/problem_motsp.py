from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.motsp.state_motsp import StateTSP
from utils.beam_search import beam_search


class MOTSP(object):

    NAME = 'motsp'

    @staticmethod
    def get_costs(dataset, pi, w1, w2):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))
        
        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        o1 = (d[:, 1:,:2] - d[:, :-1,:2]).norm(p=2, dim=2).sum(1) + (d[:, 0, :2] - d[:, -1, :2]).norm(p=2, dim=1)
        o2 = (d[: , 1:, 2:] - d[:, :-1, 2:]).norm(p=2, dim=2).sum(1) + (d[:, 0, 2:] - d[:, -1, 2:]).norm(p=2, dim=1)
        
        obj31 = torch.sqrt(o1*o1+o2*o2)
        obj32 = w1*o1 + w2*o2

        a = torch.div(obj32,obj31)
        cstr = 500*(1-a)
        

        return o1,o2,obj31,cstr, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class TSPDataset(Dataset):
    
    def __init__(self, filename=None, size=100, num_samples=500000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            # Sample points randomly in [0, 1] square
            self.data = [torch.FloatTensor(size, 4).uniform_(0, 1) for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
