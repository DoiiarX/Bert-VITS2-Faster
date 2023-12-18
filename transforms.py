import torch
from torch.nn import functional as F

import numpy as np


DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def piecewise_rational_quadratic_transform(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails=None,
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):

    spline_fn = unconstrained_rational_quadratic_spline
    spline_kwargs = {"tails": tails, "tail_bound": tail_bound}

    outputs, logabsdet = spline_fn(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
        **spline_kwargs
    )
    return outputs, logabsdet


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    
    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    unnormalized_widths = unnormalized_widths.contiguous().view(-1, 10)
    unnormalized_heights = unnormalized_heights.contiguous().view(-1, 10)
    inside_interval_mask = inside_interval_mask.contiguous().view(-1, 1)
    unnormalized_derivatives = unnormalized_derivatives.contiguous().view(-1, 11)
    inputs = inputs.contiguous().view(-1)
    unnormalized_widths_ = unnormalized_widths * inside_interval_mask
    unnormalized_heights_ = unnormalized_heights* inside_interval_mask
    unnormalized_derivatives_ = unnormalized_derivatives * inside_interval_mask
    inputs_ = inputs * inside_interval_mask.view(-1)

    (outputs, logabsdet) = rational_quadratic_spline(
        inputs=inputs_,
        unnormalized_widths=unnormalized_widths_,
        unnormalized_heights=unnormalized_heights_,
        unnormalized_derivatives=unnormalized_derivatives_,
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )
    
    outputs = outputs.unsqueeze(0).unsqueeze(0)
    logabsdet = logabsdet.unsqueeze(0).unsqueeze(0)
    
    return outputs, logabsdet

def diag_index(tensorx):
    x_1 = tensorx.shape[0]
    diagonal_indices = torch.arange(x_1)
    diagonal_elements = tensorx[diagonal_indices, diagonal_indices]
    return diagonal_elements

def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError("Input to a transform is not within its domain")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    bin_idx = searchsorted(cumheights, inputs)[..., None]

    # 4 replace torch.diag
    input_cumwidths = diag_index(cumwidths[:, bin_idx[:,0]])
    input_bin_widths = diag_index(widths[:, bin_idx[:,0]])
    input_cumheights = diag_index(cumheights[:, bin_idx[:,0]])
    delta = heights / widths
    input_delta = diag_index(delta[:, bin_idx[:,0]])
    input_derivatives = diag_index(derivatives[:, bin_idx[:,0]])
    input_derivatives_plus_one = diag_index(derivatives[..., 1:][:, bin_idx[:,0]])
    input_heights = diag_index(heights[:, bin_idx[:,0]])
    
    a = (inputs - input_cumheights) * (
        input_derivatives + input_derivatives_plus_one - 2 * input_delta
    ) + input_heights * (input_delta - input_derivatives)
    b = input_heights * input_derivatives - (inputs - input_cumheights) * (
        input_derivatives + input_derivatives_plus_one - 2 * input_delta
    )
    c = -input_delta * (inputs - input_cumheights)

    discriminant = b.pow(2) - 4 * a * c
    assert (discriminant >= 0).all()

    root = (2 * c) / (-b - torch.sqrt(discriminant))
    outputs = root * input_bin_widths + input_cumwidths

    theta_one_minus_theta = root * (1 - root)
    denominator = input_delta + (
        (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
        * theta_one_minus_theta
    )
    derivative_numerator = input_delta.pow(2) * (
        input_derivatives_plus_one * root.pow(2)
        + 2 * input_delta * theta_one_minus_theta
        + input_derivatives * (1 - root).pow(2)
    )
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    return outputs, -logabsdet
