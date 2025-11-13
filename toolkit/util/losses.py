import torch
import torch.nn.functional as F

import torchvision.utils
import os
import time

_dwt = None


def _get_wavelet_loss(device, dtype):
    global _dwt
    if _dwt is not None:
        return _dwt

    # init wavelets
    from pytorch_wavelets import DWTForward

    # wave='db1'  wave='haar'
    dwt = DWTForward(J=1, mode="zero", wave="haar").to(device=device, dtype=dtype)
    _dwt = dwt
    return dwt


def wavelet_loss(model_pred, latents, noise):
    model_pred = model_pred.float()
    latents = latents.float()
    noise = noise.float()
    dwt = _get_wavelet_loss(model_pred.device, model_pred.dtype)
    with torch.no_grad():
        model_input_xll, model_input_xh = dwt(latents)
        model_input_xlh, model_input_xhl, model_input_xhh = torch.unbind(
            model_input_xh[0], dim=2
        )
        model_input = torch.cat(
            [model_input_xll, model_input_xlh, model_input_xhl, model_input_xhh], dim=1
        )

    # reverse the noise to get the model prediction of the pure latents
    model_pred = noise - model_pred

    model_pred_xll, model_pred_xh = dwt(model_pred)
    model_pred_xlh, model_pred_xhl, model_pred_xhh = torch.unbind(
        model_pred_xh[0], dim=2
    )
    model_pred = torch.cat(
        [model_pred_xll, model_pred_xlh, model_pred_xhl, model_pred_xhh], dim=1
    )

    return torch.nn.functional.mse_loss(model_pred, model_input, reduction="none")


_gaussian_kernel_cache = {}


def _get_gaussian_kernel(device, dtype, channels):
    key = (device, dtype, channels)
    kernel = _gaussian_kernel_cache.get(key)
    if kernel is None:
        base = torch.tensor(
            [
                [1.0, 4.0, 6.0, 4.0, 1.0],
                [4.0, 16.0, 24.0, 16.0, 4.0],
                [6.0, 24.0, 36.0, 24.0, 6.0],
                [4.0, 16.0, 24.0, 16.0, 4.0],
                [1.0, 4.0, 6.0, 4.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )
        base = base / 256.0
        kernel = base.view(1, 1, 5, 5).repeat(channels, 1, 1, 1)
        _gaussian_kernel_cache[key] = kernel
    return kernel


def _conv_gauss(img, kernel):
    kw = kernel.shape[-1]
    pad = kw // 2
    img = torch.nn.functional.pad(img, (pad, pad, pad, pad), mode="replicate")
    return torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])


def _pyr_downsample(x):
    return x[:, :, ::2, ::2]


def _pyr_upsample(x, kernel, filtered_height, filtered_width):
    n_channels = kernel.shape[0]
    op0 = 1 - (filtered_height % 2)
    op1 = 1 - (filtered_width % 2)
    return torch.nn.functional.conv_transpose2d(
        x,
        kernel,
        groups=n_channels,
        stride=2,
        padding=2,
        output_padding=(op0, op1),
    )


def _laplacian_pyramid_expand(img, kernel, max_levels):
    current = img
    pyr = []
    for _ in range(max_levels):
        filtered = _conv_gauss(current, kernel)
        down = _pyr_downsample(filtered)
        up = _pyr_upsample(down, 4 * kernel, filtered.shape[-2], filtered.shape[-1])
        diff = current - up
        pyr.append(diff)

        current = down

    return pyr

def laplacian_loss(pred, target, *, pred_latents=None, target_latents=None, max_levels=5):
    pred = pred.float()
    target = target.float()

    channels = pred.shape[1]
    kernel = _get_gaussian_kernel(pred.device, pred.dtype, channels)

    pyr_pred = _laplacian_pyramid_expand(pred, kernel, max_levels)
    pyr_target = _laplacian_pyramid_expand(target, kernel, max_levels)

    base_size = pred.shape[-2:]
    weights = [2 ** i for i in range(len(pyr_pred))]
    weight_total = float(sum(weights)) if weights else 1.0

    loss_map = 0.0
    for weight, pred_level, target_level in zip(weights, pyr_pred, pyr_target):
        if pred_level.shape[-2:] != base_size:
            pred_level = torch.nn.functional.interpolate(
                pred_level, size=base_size, mode="bilinear", align_corners=False
            )
            target_level = torch.nn.functional.interpolate(
                target_level, size=base_size, mode="bilinear", align_corners=False
            )
        level_loss = weight * torch.nn.functional.l1_loss(
            pred_level, target_level, reduction="none"
        )
        loss_map = loss_map + level_loss

    loss_map = loss_map / weight_total * 10

    return loss_map


def stepped_loss(model_pred, latents, noise, noisy_latents, timesteps, scheduler):
    # this steps the on a 20 step timescale from the current step (50 idx steps ahead)
    # and then reconstructs the original image at that timestep. This should lessen the error
    # possible in high noise timesteps and make the flow smoother.
    bs = model_pred.shape[0]

    noise_pred_chunks = torch.chunk(model_pred, bs)
    timestep_chunks = torch.chunk(timesteps, bs)
    noisy_latent_chunks = torch.chunk(noisy_latents, bs)
    noise_chunks = torch.chunk(noise, bs)

    x0_pred_chunks = []

    for idx in range(bs):
        model_output = noise_pred_chunks[idx]  # predicted noise (same shape as latent)
        timestep = timestep_chunks[idx]  # scalar tensor per sample (e.g., [t])
        sample = noisy_latent_chunks[idx].to(torch.float32)
        noise_i = noise_chunks[idx].to(sample.dtype).to(sample.device)

        # Initialize scheduler step index for this sample
        scheduler._step_index = None
        scheduler._init_step_index(timestep)

        # ---- Step +50 indices (or to the end) in sigma-space ----
        sigma = scheduler.sigmas[scheduler.step_index]
        target_idx = min(scheduler.step_index + 50, len(scheduler.sigmas) - 1)
        sigma_next = scheduler.sigmas[target_idx]

        # One-step update along the model-predicted direction
        stepped = sample + (sigma_next - sigma) * model_output

        # ---- Inverse-Gaussian recovery at the target timestep ----
        t_01 = (
            (scheduler.sigmas[target_idx]).to(stepped.device).to(stepped.dtype)
        )
        original_samples = (stepped - t_01 * noise_i) / (1.0 - t_01)
        x0_pred_chunks.append(original_samples)

    predicted_images = torch.cat(x0_pred_chunks, dim=0)

    return torch.nn.functional.mse_loss(
        predicted_images.float(),
        latents.float().to(device=predicted_images.device),
        reduction="none",
    )
