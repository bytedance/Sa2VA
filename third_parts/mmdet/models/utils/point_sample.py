# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor
import torch.nn.functional as F


def normalize(grid: Tensor) -> Tensor:
    """Normalize input grid from [-1, 1] to [0, 1]

    Args:
        grid (torch.Tensor): The grid to be normalize, range [-1, 1].

    Returns:
        torch.Tensor: Normalized grid, range [0, 1].
    """

    return (grid + 1.0) / 2.0


def denormalize(grid: Tensor) -> Tensor:
    """Denormalize input grid from range [0, 1] to [-1, 1]

    Args:
        grid (torch.Tensor): The grid to be denormalize, range [0, 1].

    Returns:
        torch.Tensor: Denormalized grid, range [-1, 1].
    """

    return grid * 2.0 - 1.0


def point_sample(input: Tensor,
                 points: Tensor,
                 align_corners: bool = False,
                 **kwargs) -> Tensor:
    """A wrapper around :func:`grid_sample` to support 3D point_coords tensors
    Unlike :func:`torch.nn.functional.grid_sample` it assumes point_coords to
    lie inside ``[0, 1] x [0, 1]`` square.

    Args:
        input (torch.Tensor): Feature map, shape (N, C, H, W).
        points (torch.Tensor): Image based absolute point coordinates
            (normalized), range [0, 1] x [0, 1], shape (N, P, 2) or
            (N, Hgrid, Wgrid, 2).
        align_corners (bool, optional): Whether align_corners.
            Default: False

    Returns:
        torch.Tensor: Features of `point` on `input`, shape (N, C, P) or
        (N, C, Hgrid, Wgrid).
    """

    add_dim = False
    if points.dim() == 3:
        add_dim = True
        points = points.unsqueeze(2)
    output = F.grid_sample(
        input, denormalize(points), align_corners=align_corners, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def get_uncertainty(mask_preds: Tensor, labels: Tensor) -> Tensor:
    """Estimate uncertainty based on pred logits.

    We estimate uncertainty as L1 distance between 0.0 and the logits
    prediction in 'mask_preds' for the foreground class in `classes`.

    Args:
        mask_preds (Tensor): mask predication logits, shape (num_rois,
            num_classes, mask_height, mask_width).

        labels (Tensor): Either predicted or ground truth label for
            each predicted mask, of length num_rois.

    Returns:
        scores (Tensor): Uncertainty scores with the most uncertain
            locations having the highest uncertainty score,
            shape (num_rois, 1, mask_height, mask_width)
    """
    if mask_preds.shape[1] == 1:
        gt_class_logits = mask_preds.clone()
    else:
        inds = torch.arange(mask_preds.shape[0], device=mask_preds.device)
        gt_class_logits = mask_preds[inds, labels].unsqueeze(1)
    return -torch.abs(gt_class_logits)


def get_uncertain_point_coords_with_randomness(
        mask_preds: Tensor, labels: Tensor, num_points: int,
        oversample_ratio: float, importance_sample_ratio: float) -> Tensor:
    """Get ``num_points`` most uncertain points with random points during
    train.

    Sample points in [0, 1] x [0, 1] coordinate space based on their
    uncertainty. The uncertainties are calculated for each point using
    'get_uncertainty()' function that takes point's logit prediction as
    input.

    Args:
        mask_preds (Tensor): A tensor of shape (num_rois, num_classes,
            mask_height, mask_width) for class-specific or class-agnostic
            prediction.
        labels (Tensor): The ground truth class for each instance.
        num_points (int): The number of points to sample.
        oversample_ratio (float): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled
            via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (num_rois, num_points, 2)
            that contains the coordinates sampled points.
    """
    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1
    batch_size = mask_preds.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(
        batch_size, num_sampled, 2, device=mask_preds.device)
    point_logits = point_sample(mask_preds, point_coords)
    # It is crucial to calculate uncertainty based on the sampled
    # prediction value for the points. Calculating uncertainties of the
    # coarse predictions first and sampling them for points leads to
    # incorrect results.  To illustrate this: assume uncertainty func(
    # logits)=-abs(logits), a sampled point between two coarse
    # predictions with -1 and 1 logits has 0 logits, and therefore 0
    # uncertainty value. However, if we calculate uncertainties for the
    # coarse predictions first, both will have -1 uncertainty,
    # and sampled point will get -1 uncertainty.
    point_uncertainties = get_uncertainty(point_logits, labels)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(
        point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(
        batch_size, dtype=torch.long, device=mask_preds.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        batch_size, num_uncertain_points, 2)
    if num_random_points > 0:
        rand_roi_coords = torch.rand(
            batch_size, num_random_points, 2, device=mask_preds.device)
        point_coords = torch.cat((point_coords, rand_roi_coords), dim=1)
    return point_coords
