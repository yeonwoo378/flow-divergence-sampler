import torch
import torch.nn as nn
from model_jit import JiT_models
from tqdm import tqdm
from einops import rearrange, repeat, reduce
import numpy as np
import math

class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels):
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten(), labels_dropped)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss

    @torch.no_grad()
    def generate(self, labels, args, noise=None, inverse=False):
        device = labels.device
        bsz = labels.size(0)
        if noise is None:
            z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        else:
            z = self.noise_scale * noise
        if inverse:
            start = 1.0
            end = 0.0
        else:
            start = 0.0
            end = 1.0
        timesteps = torch.linspace(start, end, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif 'euler_ours' in self.method:
            stepper = self._euler_step_ours

        elif self.method == "heun":
            stepper = self._heun_step

        elif 'heun_ours' in self.method:
            stepper = self._heun_step_ours
        else:
            raise NotImplementedError

        # ode
        for i in tqdm(range(self.steps - 1)):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            if 'ours' in self.method:
                z = stepper(z, t, t_next, labels, args=args, iter=args.iter, perturb=args.perturb_scale)
            else:
                z = stepper(z, t, t_next, labels)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    # @torch.no_grad()
    def _forward_sample(self, z, t, labels, grad=False):
        
        # if self.cfg_scale > 0:
        #     # conditional
        #     x_cond = self.net(z, t.flatten(), labels)
        #     v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # if self.cfg_scale != 1.0:
        #     # unconditional
        #     x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
        #     v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)
            
        batched_z_input = torch.cat([z, z], dim=0)
        batched_t_input = torch.cat([t.flatten(), t.flatten()], dim=0)
        batched_labels_input = torch.cat([labels, torch.full_like(labels, self.num_classes)], dim=0)
        
        if grad:
            x = self.net(batched_z_input, batched_t_input, batched_labels_input)
        else:
            with torch.no_grad():
                x = self.net(batched_z_input, batched_t_input, batched_labels_input)

        x_cond, x_uncond = torch.chunk(x, 2, dim=0)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)


        if self.cfg_scale == 1.0:
            return v_cond
        elif self.cfg_scale == 0.0:
            return v_uncond
        else:
            return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)

    
    def get_scheduled_value(self, total, cur, schedule_type):
        if schedule_type == 'constant': 
            return total
    
        elif schedule_type == 'linear': # 1 - (t/T)
            return total * (1. - cur / total)
        
        elif schedule_type == 'cosine': 
            return total * 0.5 * (1. + math.cos(math.pi * cur / total))
        
        elif schedule_type == 'sqrt':
           return total * math.sqrt(1. - cur / total)
        
        elif schedule_type == 'concave':
           return (1. - cur / total) ** 2
        
        elif schedule_type == 'convex':
           return 1. - (cur / total) ** 2
        
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")


class DenoiserCustom(Denoiser):
    @torch.no_grad()
    def generate(self, labels, args, noise=None, inverse=False):
        device = labels.device
        bsz = labels.size(0)
        if noise is None:
            z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        else:
            z = self.noise_scale * noise
        if inverse:
            start = 1.0
            end = 0.0
        else:
            start = 0.0
            end = 1.0
        timesteps = torch.linspace(start, end, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step

        elif self.method == "heun":
            stepper = self._heun_step
            
        elif 'ours' not in self.method:
            raise NotImplementedError
        
        improved = None
        delta = None
        # ode
        for i in tqdm(range(self.steps - 1)):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            if 'ours' in self.method:
                # z = stepper(z, t, t_next, labels, args=args, iter=args.iter, perturb=args.perturb_scale)
                v_func_kwargs = {
                    'z': z,
                    't': t,
                    't_next': t_next,
                    'labels': labels}
            
                iter = args.iter
                if t.mean().item() > args.stop_t:
                    iter = 0

                # hard-coded as euler ours -> TODO: generalize
                v_func = None
                update_func = None
                if 'euler' in self.method:
                    v_func = self._euler_get_v_pred
                    update_func = self._euler_update
                elif 'heun' in self.method:
                    v_func = self._heun_get_v_pred
                    update_func = self._heun_update    
                else:
                    raise NotImplementedError
                delta_schedule = self.get_scheduled_value(1., t.mean().item(), schedule_type=args.perturb_schedule)
                best_z, best_v_pred, improved, delta = self.divergence_stepper(v_func,
                                            v_func_kwargs,
                                            x_key='z',
                                            t_key='t',
                                            stop_t=args.stop_t,
                                            num_updates=iter,
                                            delta_scale=args.perturb_scale * delta_schedule, #/ iter if iter > 0 else args.perturb_scale * delta_schedule, # currently hard-coded
                                            # delta_scale=args.perturb_scale* 0.5 * (1. + math.cos(math.pi * t.mean().item())),
                                            # delta_scheduler=lambda n: 1,
                                            delta_scheduler=lambda n: 2**(-n),
                                            seed_delta=args.seed_delta,
                                            seed_eps=args.seed_eps,
                                            improved=improved,
                                            delta=delta,
                                            num_delta=args.num_delta
                )
                # z = best_z + (t_next - t) * best_v_pred # z_next
                z = update_func(best_z, t, t_next, best_v_pred)
        
            else:
                z = stepper(z, t, t_next, labels)
        # last step euler - (default setup of jit)
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    # @torch.no_grad()
    def _euler_get_v_pred(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)

        return v_pred
    
    def _euler_update(self, z, t, t_next, v_pred):
        z = z + (t_next - t) * v_pred # z_next
        return z
    
    def _heun_get_v_pred(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)

        return v_pred
    
    def _heun_update(self, z, t, t_next, v_pred):
        z_next = z + (t_next - t) * v_pred
        return z_next
    
    @torch.no_grad()
    def divergence_stepper(self, v_func,
                           v_func_kwargs,
                           x_key='z',
                           t_key='t',
                           stop_t=1.0,
                           num_updates=1,
                           num_delta=1,
                           num_eps=1,
                           delta_scale=1,
                           delta_scheduler=lambda t: 2 ** (-t),
                           seed_delta=None,
                           seed_eps=None,
                           delta=None,
                           improved=None,
                           sequential_hutchinson=True,
                           ):
        assert stop_t >= 0.0 and stop_t <= 1.0

        t = v_func_kwargs[t_key]
        if isinstance(t, torch.Tensor):
            assert (t == t.mean()).all().item(), "All timesteps in the batch must be the same for divergence_stepper."
            t = t.mean().item()

        if num_updates <= 0 or t > stop_t:
            return v_func_kwargs[x_key], v_func(**v_func_kwargs), improved, delta
        
        z = v_func_kwargs[x_key]        
        B = z.shape[0]
        D = np.prod(z.shape[1:])  # C * H * W
        
        delta_generator = None
        eps_generator = None
        
        if seed_delta is not None:
            delta_generator = torch.Generator(device=z.device).manual_seed(seed_delta + int(t * 1000))
        if seed_eps is not None:
            eps_generator = torch.Generator(device=z.device).manual_seed(seed_eps)
        sync_eps_with_delta = num_eps == 1 and seed_eps == seed_delta
        
        for update_idx in range(num_updates):
            # compute divergence and find the best perturbation
            for delta_idx in range(num_delta+1):

                if delta is None or improved is None:
                    assert improved is None and t == 0.0 and delta_idx <= 1
                    delta = torch.randn(z.shape, generator=delta_generator, device=z.device)
                elif update_idx > 0:
                    temp_delta_generator = torch.Generator(device=z.device).manual_seed(seed_delta + update_idx)
                    temp_delta = torch.randn(z.shape, generator=temp_delta_generator, device=z.device)
                    pass
                elif delta_idx > 0:
                    new_delta = torch.randn(z.shape, generator=delta_generator, device=z.device)
                    delta = torch.where(
                        improved.reshape(-1, *([1]*(z.ndim-1))), # hard-coded shape
                        delta, # True
                        new_delta # False
                    )
                # no update delta when delta_idx=0
                assert seed_delta != seed_eps, "Is a Biased Estimator"

                if sync_eps_with_delta and delta_idx != 0:
                    eps = delta.detach()
                    raise NotImplementedError # not using anymore!
                else:
                    eps = torch.randn(z.shape, generator=eps_generator, device=z.device)

                if delta_idx == 0:
                    perturbed_z = z
                elif update_idx == 0:
                    perturbed_z = z + delta_scale * delta_scheduler(update_idx) * delta # TODO: clarify
                else:
                    perturbed_z = z + delta_scale * delta_scheduler(update_idx) * temp_delta # TODO: clarify
                with torch.enable_grad():
                    perturbed_z = perturbed_z.detach().requires_grad_(True)
                    v_func_kwargs[x_key] = perturbed_z

                    v_pred = v_func(**v_func_kwargs)  # [B, C, H, W]
                    v_pred_eps = (v_pred * eps).flatten(1).sum(1)  # [B]
                    grad_v = torch.autograd.grad(
                        outputs=v_pred_eps,          # [B]
                        inputs=perturbed_z,                      # [B, C, H, W]
                        grad_outputs=torch.ones_like(v_pred_eps),  # [B]
                        create_graph=False,
                        retain_graph=False,
                    )[0].detach()  # [B, C, H, W]
                    divergence = (grad_v * eps).flatten(1).sum(1) / D  # [B]

                threshold = - (1 / (1 - t))

                if delta_idx == 0:
                    best_divergence = divergence.detach()
                    best_v_pred = v_pred.detach()
                    best_perturbed_z = perturbed_z.detach()
                elif update_idx == 0:
                    improved = (divergence < best_divergence) & (best_divergence >= threshold)
                    improved_shape = (B,) + (1,) * (len(z.shape) - 1)
                    best_divergence = torch.where(improved, divergence, best_divergence)
                    best_v_pred = torch.where(
                        improved.view(improved_shape),
                        v_pred,
                        best_v_pred,
                    )
                    best_perturbed_z = torch.where(
                        improved.view(improved_shape),
                        perturbed_z.detach(),
                        best_perturbed_z,
                    )
                else:
                    temp_improved = (divergence < best_divergence) & (best_divergence >= threshold)
                    improved_shape = (B,) + (1,) * (len(z.shape) - 1)
                    best_divergence = torch.where(temp_improved, divergence, best_divergence)
                    best_v_pred = torch.where(
                        temp_improved.view(improved_shape),
                        v_pred,
                        best_v_pred,
                    )
                    best_perturbed_z = torch.where(
                        temp_improved.view(improved_shape),
                        perturbed_z.detach(),
                        best_perturbed_z,
                    )

            # update iteration-wise
            z = best_perturbed_z # update z
            v_pred = best_v_pred
        return best_perturbed_z, best_v_pred, improved, delta
    

