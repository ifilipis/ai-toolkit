import torch
import torch.nn.functional as F


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


def _get_gaussian_kernel5(channels, device, dtype):
    key = (channels, device, dtype)
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
            dtype=torch.float32,
        )
        base = base / 256.0
        base = base.to(dtype=dtype)
        kernel = base.view(1, 1, 5, 5).repeat(channels, 1, 1, 1)
        _gaussian_kernel_cache[key] = kernel
    return kernel


def _conv_gaussian(img, kernel):
    n_channels = img.shape[1]
    padding = kernel.shape[-1] // 2
    img = F.pad(img, (padding, padding, padding, padding), mode="replicate")
    return F.conv2d(img, kernel, groups=n_channels)


def _pyr_downsample(x):
    return x[:, :, ::2, ::2]


def _pyr_upsample(x, kernel, target_size):
    n_channels = x.shape[1]
    op_h = target_size[0] % 2
    op_w = target_size[1] % 2
    return F.conv_transpose2d(
        x,
        kernel,
        groups=n_channels,
        stride=2,
        padding=2,
        output_padding=(1 - op_h, 1 - op_w),
    )


def _laplacian_pyramid_expand(img, kernel, max_levels=5):
    current = img
    pyr = []
    expanded_kernel = 4 * kernel
    for level in range(max_levels):
        filtered = _conv_gaussian(current, kernel)
        down = _pyr_downsample(filtered)
        up = _pyr_upsample(down, expanded_kernel, filtered.shape[-2:])
        diff = current - up
        pyr.append(diff)
        if down.shape[-2] < 2 or down.shape[-1] < 2:
            break
        current = down
    return pyr


def laplacian_pyramid_loss(pred, target, max_levels=5):
    pred = pred.float()
    target = target.float()

    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape for laplacian loss")

    kernel = _get_gaussian_kernel5(pred.shape[1], pred.device, pred.dtype)
    pyr_pred = _laplacian_pyramid_expand(pred, kernel, max_levels=max_levels)
    pyr_target = _laplacian_pyramid_expand(target, kernel, max_levels=max_levels)

    levels = min(len(pyr_pred), len(pyr_target))
    weights = [2 ** i for i in range(levels)]

    losses = []
    reduce_dims = tuple(range(1, pred.ndim))
    for weight, pred_level, target_level in zip(weights, pyr_pred[:levels], pyr_target[:levels]):
        level_loss = torch.abs(pred_level - target_level)
        losses.append(weight * level_loss.mean(dim=reduce_dims, keepdim=True))

    if not losses:
        return torch.zeros((pred.shape[0], 1, 1, 1), device=pred.device, dtype=pred.dtype)

    return torch.stack(losses, dim=0).sum(dim=0)


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
