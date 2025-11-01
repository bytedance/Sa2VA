from itertools import zip_longest, chain
import os.path as osp
import subprocess
import torch
import os
from torch import distributed as torch_dist
from torch.distributed import ProcessGroup
import functools
from typing import Callable, Optional, Tuple
import pickle
import shutil


def _init_dist_pytorch(backend, **kwargs) -> None:
    """Initialize distributed environment with PyTorch launcher.

    Args:
        backend (str): Backend of torch.distributed. Supported backends are
            'nccl', 'gloo' and 'mpi'. Defaults to 'nccl'.
        **kwargs: keyword arguments are passed to ``init_process_group``.
    """
    # LOCAL_RANK is set by `torch.distributed.launch` since PyTorch 1.1
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    torch_dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend,
                     port=None,
                     init_backend='torch',
                     **kwargs) -> None:
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    # Not sure when this environment variable could be None, so use a fallback
    local_rank_env = os.environ.get('SLURM_LOCALID', None)
    if local_rank_env is not None:
        local_rank = int(local_rank_env)
    else:
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['RANK'] = str(proc_id)

    torch.cuda.set_device(local_rank)

    if init_backend == 'torch':
        torch_dist.init_process_group(backend=backend, **kwargs)
    elif init_backend == 'deepspeed':
        import deepspeed
        deepspeed.init_distributed(dist_backend=backend, **kwargs)
    elif init_backend == 'colossalai':
        import colossalai
        colossalai.launch_from_slurm(
            backend=backend,
            host=os.environ['MASTER_ADDR'],
            port=os.environ['MASTER_PORT'],
            **kwargs,
        )
    else:
        raise ValueError(
            'supported "init_backend" is "torch" or "deepspeed", '
            f'but got {init_backend}')
        

def get_dist_info(group=None) -> Tuple[int, int]:
    """Get distributed information of the given process group.

    Note:
        Calling ``get_dist_info`` in non-distributed environment will return
        (0, 1).

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        tuple[int, int]: Return a tuple containing the ``rank`` and
        ``world_size``.
    """
    world_size = get_world_size(group)
    rank = get_rank(group)
    return rank, world_size

def get_world_size(group: Optional[ProcessGroup] = None) -> int:
    """Return the number of the given process group.

    Note:
        Calling ``get_world_size`` in non-distributed environment will return
        1.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Return the number of processes of the given process group if in
        distributed environment, otherwise 1.
    """
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_world_size(group)
    else:
        return 1


def get_rank(group: Optional[ProcessGroup] = None) -> int:
    """Return the rank of the given process group.

    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.

    Note:
        Calling ``get_rank`` in non-distributed environment will return 0.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Return the rank of the process group if in distributed
        environment, otherwise 0.
    """

    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_rank(group)
    else:
        return 0

def is_distributed() -> bool:
    """Return True if distributed environment has been initialized."""
    return torch_dist.is_available() and torch_dist.is_initialized()

def get_default_group() -> Optional[ProcessGroup]:
    """Return default process group."""

    return torch_dist.distributed_c10d._get_default_group()

def is_main_process(group: Optional[ProcessGroup] = None) -> bool:
    """Whether the current rank of the given process group is equal to 0.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        bool: Return True if the current rank of the given process group is
        equal to 0, otherwise False.
    """
    return get_rank(group) == 0

def master_only(func: Callable) -> Callable:
    """Decorate those methods which should be executed in master process.

    Args:
        func (callable): Function to be decorated.

    Returns:
        callable: Return decorated function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)
    return wrapper

def collect_results_cpu(result_part: list,
                        size: int,
                        tmpdir='./dist_test_temp'):
    """Collect results under cpu mode.

    On cpu mode, this function will save the results on different gpus to
    ``tmpdir`` and collect them by the rank 0 worker.

    Args:
        result_part (list): Result list containing result parts
            to be collected. Each item of ``result_part`` should be a picklable
            object.
        size (int): Size of the results, commonly equal to length of
            the results.
        tmpdir (str | None): Temporal directory for collected results to
            store. If set to None, it will create a random temporal directory
            for it. Defaults to None.

    Returns:
        list or None: The collected results.
    """
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part[:size]

    # create a tmp dir if it is not specified
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)

    # dump the part result to the dir
    with open(osp.join(tmpdir, f'part_{rank}.pkl'), 'wb') as f:  # type: ignore
        pickle.dump(result_part, f, protocol=2)

    barrier()

    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            path = osp.join(tmpdir, f'part_{i}.pkl')  # type: ignore
            if not osp.exists(path):
                raise FileNotFoundError(
                    f'{tmpdir} is not an shared directory for '
                    f'rank {i}, please make sure {tmpdir} is a shared '
                    'directory for all ranks!')
            with open(path, 'rb') as f:
                part_list.append(pickle.load(f))
        # sort the results
        ordered_results = []
        zipped_results = zip_longest(*part_list)
        ordered_results = [
            i for i in chain.from_iterable(zipped_results) if i is not None
        ]
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)  # type: ignore
        return ordered_results


def barrier(group: Optional[ProcessGroup] = None) -> None:
    """Synchronize all processes from the given process group.

    This collective blocks processes until the whole group enters this
    function.

    Note:
        Calling ``barrier`` in non-distributed environment will do nothing.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.
    """
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        torch_dist.barrier(group)