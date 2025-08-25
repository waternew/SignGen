import torch
from typing import Optional, List
from torch.distributions import Normal
from functools import partial
from .pndm import gen_order_1, gen_order_4


def _to_labels(x: torch.Tensor, t: int) -> torch.Tensor:
    return torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)


def ddpm_sampler(x, model, n_steps_each: int = 1, step_lr: float = 0.00002,
                 final_only: bool = True, denoise: bool = True,
                 subsample_steps: Optional[List[int]] = None,
                 clip_before: bool = True, cond=None, cond_mask=None,
                 verbose: bool = False, log: bool = False, gamma: bool = False,
                 texts=None, deepth_video=None, ori_X=None, motion=None,
                 pose_video=None, **kwargs):
    """Simplified DDPM sampler.

    Args:
        x: Initial noise sample.
        model: Score network predicting noise epsilon.
    Returns:
        List of samples during sampling (only final if final_only).
    """
    net = model.module if hasattr(model, 'module') else model
    betas = net.betas
    alphas = net.alphas
    alphas_prev = getattr(net, 'alphas_prev', alphas)
    T = betas.shape[0]
    if subsample_steps is not None:
        timesteps = list(subsample_steps)
    else:
        timesteps = list(range(T - 1, -1, -1))

    sample_list = []
    x_t = x
    for t in timesteps:
        if verbose:
            print(f"t={t}")
        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_prev = alphas_prev[t] if t < len(alphas_prev) else alphas[0]
        # model predicts noise
        labels = _to_labels(x_t, t)
        eps = model(x_t, labels, cond=cond, cond_mask=cond_mask,
                    texts=texts, deepth_video=deepth_video, ori_X=ori_X,
                    motion=motion, pose_video=pose_video)
        sqrt_alpha_t = alpha_t.sqrt()
        sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt()
        x0_hat = (x_t - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t
        sqrt_alpha_prev = alpha_prev.sqrt()
        sqrt_one_minus_alpha_prev = (1 - alpha_prev).sqrt()
        mean = sqrt_alpha_prev * x0_hat + sqrt_one_minus_alpha_prev * eps
        if t > 0:
            noise = torch.randn_like(x_t)
            x_t = mean + noise * beta_t.sqrt()
        else:
            x_t = mean
        if clip_before:
            x_t = x_t.clamp_(-1., 1.)
        if not final_only:
            sample_list.append(x_t.clone())
    if denoise:
        labels = _to_labels(x_t, 0)
        eps = model(x_t, labels, cond=cond, cond_mask=cond_mask,
                    texts=texts, deepth_video=deepth_video, ori_X=ori_X,
                    motion=motion, pose_video=pose_video)
        x_t = (x_t - (1 - alphas[0]).sqrt() * eps) / alphas[0].sqrt()
    if final_only:
        return torch.stack([x_t])
    sample_list.append(x_t)
    return torch.stack(sample_list)


def ddim_sampler(x, model, n_steps_each: int = 1, step_lr: float = 0.00002,
                 final_only: bool = True, denoise: bool = True,
                 subsample_steps: Optional[List[int]] = None,
                 clip_before: bool = True, cond=None, cond_mask=None,
                 verbose: bool = False, log: bool = False, gamma: bool = False,
                 texts=None, deepth_video=None, ori_X=None, motion=None,
                 pose_video=None, eta: float = 0.0, **kwargs):
    """Deterministic DDIM sampler."""
    net = model.module if hasattr(model, 'module') else model
    alphas = net.alphas
    alphas_prev = getattr(net, 'alphas_prev', alphas)
    T = alphas.shape[0]
    if subsample_steps is not None:
        timesteps = list(subsample_steps)
    else:
        timesteps = list(range(T - 1, -1, -1))
    sample_list = []
    x_t = x
    for t in timesteps:
        alpha_t = alphas[t]
        alpha_prev = alphas_prev[t] if t < len(alphas_prev) else alphas[0]
        labels = _to_labels(x_t, t)
        eps = model(x_t, labels, cond=cond, cond_mask=cond_mask,
                    texts=texts, deepth_video=deepth_video, ori_X=ori_X,
                    motion=motion, pose_video=pose_video)
        sqrt_alpha_t = alpha_t.sqrt()
        sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt()
        x0_hat = (x_t - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t
        sqrt_alpha_prev = alpha_prev.sqrt()
        sqrt_one_minus_alpha_prev = (1 - alpha_prev).sqrt()
        if eta > 0 and t > 0:
            noise = torch.randn_like(x_t)
            sigma_t = eta * ((1 - alpha_prev) / (1 - alpha_t)).sqrt() * (1 - alpha_t / alpha_prev).sqrt()
            mean = sqrt_alpha_prev * x0_hat + (sqrt_one_minus_alpha_prev - sigma_t) * eps
            x_t = mean + sigma_t * noise
        else:
            x_t = sqrt_alpha_prev * x0_hat + sqrt_one_minus_alpha_prev * eps
        if clip_before:
            x_t = x_t.clamp_(-1., 1.)
        if not final_only:
            sample_list.append(x_t.clone())
    if denoise:
        labels = _to_labels(x_t, 0)
        eps = model(x_t, labels, cond=cond, cond_mask=cond_mask,
                    texts=texts, deepth_video=deepth_video, ori_X=ori_X,
                    motion=motion, pose_video=pose_video)
        x_t = (x_t - (1 - alphas[0]).sqrt() * eps) / alphas[0].sqrt()
    if final_only:
        return torch.stack([x_t])
    sample_list.append(x_t)
    return torch.stack(sample_list)


def FPNDM_sampler(x, model, n_steps_each: int = 1, step_lr: float = 0.00002,
                  final_only: bool = True, denoise: bool = True,
                  subsample_steps: Optional[List[int]] = None,
                  clip_before: bool = True, cond=None, cond_mask=None,
                  verbose: bool = False, log: bool = False, gamma: bool = False,
                  texts=None, deepth_video=None, ori_X=None, motion=None,
                  pose_video=None, **kwargs):
    """Fourth-order PNDM sampler."""
    net = model.module if hasattr(model, 'module') else model
    alphas = net.alphas
    T = alphas.shape[0]
    alphas_cump = torch.cat([torch.ones(1, device=alphas.device), alphas])
    timesteps = list(range(T - 1, -1, -1))
    x_t = x
    sample_list = []
    ets: List[torch.Tensor] = []
    for t in timesteps:
        t_next = max(t - 1, -1)
        labels = _to_labels(x_t, t)
        model_fn = partial(model, cond=cond, cond_mask=cond_mask,
                           texts=texts, deepth_video=deepth_video,
                           ori_X=ori_X, motion=motion, pose_video=pose_video)
        x_t, ets = gen_order_4(x_t, labels, _to_labels(x_t, t_next), model_fn, alphas_cump, ets, clip_before=clip_before)
        if not final_only:
            sample_list.append(x_t.clone())
    if denoise:
        labels = _to_labels(x_t, 0)
        eps = model(x_t, labels, cond=cond, cond_mask=cond_mask,
                    texts=texts, deepth_video=deepth_video, ori_X=ori_X,
                    motion=motion, pose_video=pose_video)
        x_t = (x_t - (1 - alphas[0]).sqrt() * eps) / alphas[0].sqrt()
    if final_only:
        return torch.stack([x_t])
    sample_list.append(x_t)
    return torch.stack(sample_list)


def anneal_Langevin_dynamics(x_mod, score, n_steps_each, step_lr,
                              sigmas=None, final_only=True, denoise=True,
                              clip_before=True, cond=None, cond_mask=None,
                              verbose=False, log=False, just_beta=False,
                              texts=None, deepth_video=None, ori_X=None,
                              motion=None, pose_video=None, **kwargs):
    net = score.module if hasattr(score, 'module') else score
    if sigmas is None:
        sigmas = getattr(net, 'sigmas', None)
    if sigmas is None:
        raise RuntimeError('Score model must provide sigmas for Langevin dynamics')
    samples = []
    x = x_mod
    for c, sigma in enumerate(sigmas):
        sigma = torch.tensor(sigma, device=x.device)
        labels = _to_labels(x, c)
        for s in range(n_steps_each):
            grad = score(x, labels, cond=cond, cond_mask=cond_mask,
                         texts=texts, deepth_video=deepth_video, ori_X=ori_X,
                         motion=motion, pose_video=pose_video)
            noise = torch.randn_like(x)
            x = x + step_lr * grad + noise * (2 * step_lr) ** 0.5
            if clip_before:
                x = x.clamp_(-1., 1.)
        if not final_only:
            samples.append(x.clone())
    if denoise:
        labels = _to_labels(x, 0)
        grad = score(x, labels, cond=cond, cond_mask=cond_mask,
                      texts=texts, deepth_video=deepth_video, ori_X=ori_X,
                      motion=motion, pose_video=pose_video)
        x = x + sigmas[0] ** 2 * grad
    if final_only:
        return torch.stack([x])
    samples.append(x)
    return torch.stack(samples)


def anneal_Langevin_dynamics_consistent(*args, **kwargs):
    return anneal_Langevin_dynamics(*args, **kwargs)


def anneal_Langevin_dynamics_inpainting(init_samples, refer_image, score,
                                        image_size, n_steps_each, step_lr,
                                        cond=None, cond_mask=None, **kwargs):
    b1, b2 = init_samples.shape[0], init_samples.shape[1]
    x = init_samples.view(b1 * b2, *init_samples.shape[2:])
    ref = refer_image.view(b1 * b2, *refer_image.shape[2:])
    mask = (ref != 0).float()
    samples = anneal_Langevin_dynamics(x, score, n_steps_each, step_lr,
                                       cond=cond, cond_mask=cond_mask, **kwargs,
                                       final_only=True)
    x = samples[-1]
    x = x * (1 - mask) + ref * mask
    return x.view(b1, b2, *x.shape[1:])


def anneal_Langevin_dynamics_interpolation(init_samples, score, n_interpolations,
                                            n_steps_each, step_lr, cond=None,
                                            cond_mask=None, **kwargs):
    outputs = []
    for i in range(n_interpolations):
        x = init_samples.clone()
        samples = anneal_Langevin_dynamics(x, score, n_steps_each, step_lr,
                                           cond=cond, cond_mask=cond_mask,
                                           **kwargs, final_only=True)
        outputs.append(samples[-1])
    return torch.stack(outputs)
