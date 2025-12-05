import argparse
from mmengine.config import DictAction

import xtuner.tools.train as train

def parse_args():
    parser = argparse.ArgumentParser(description='Train LLM')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--deepspeed',
        type=str,
        default='deepspeed_zero2',
        help='the path to the .json file for deepspeed')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='specify checkpoint path to be resumed from.')
    parser.add_argument(
        '--seed', type=int, default=None, help='Random seed for the training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    return args

train.parse_args = parse_args


def disable_torch_repr():
    """Disable torch.Tensor.__repr__ to avoid printing large tensors."""
    import torch
    torch.Tensor.buf_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = lambda self: f'Tensor({self.size()})'

def enable_torch_repr():
    """Enable torch.Tensor.__repr__ to print large tensors."""
    import torch
    torch.Tensor.__repr__ = torch.Tensor.buf_repr
    del torch.Tensor.buf_repr


def enable_short_torch_repr():
    import torch
    ori_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = lambda self: f'Shape of Tensor({self.size()})' if self.numel() > 100 else ori_repr(self)

enable_short_torch_repr()

if __name__ == '__main__':
    train.main()
