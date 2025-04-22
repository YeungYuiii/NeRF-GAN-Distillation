# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing
from torchvision import models
import torch.nn as nn
from torchvision import transforms

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class VGGLoss(nn.Module):
    def __init__(self, device, n_layers=5):
        super().__init__()
        
        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (0.2, 0.3, 0.3, 0.3, 0.1)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer
        
        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss().to(device)
        
    def forward(self, source, target):
        source = self.transform(source)
        target = self.transform(target)
        loss = 0 
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight*self.criterion(source, target)
        return loss 


class StyleGAN2Loss(Loss):
    def __init__(self, device, G, G_eg3d, D, 
                    augment_pipe=None, 
                    r1_gamma=10, 
                    style_mixing_prob=0, 
                    pl_weight=0, 
                    pl_batch_shrink=2, 
                    pl_decay=0.01, 
                    pl_no_weight_grad=False, 
                    blur_init_sigma=0, 
                    blur_fade_kimg=0, 
                    r1_gamma_init=0, r1_gamma_fade_kimg=0, 
                    neural_rendering_resolution_initial=64, 
                    neural_rendering_resolution_final=None, 
                    neural_rendering_resolution_fade_kimg=0, 
                    gpc_reg_fade_kimg=1000, 
                    gpc_reg_prob=None, 
                    dual_discrimination=False, 
                    filter_mode='antialiased'
                    ):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.G_eg3d             = G_eg3d
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.r1_gamma_init      = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
        self.blur_raw_target = True
        self.vggloss = VGGLoss(device=device)
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)

    def run_G(self, z, c, swapping_prob, neural_rendering_resolution, update_emas=False, require_grad=False):
        if swapping_prob is not None:
            c_swapped = torch.roll(c.clone(), 1, 0)
            c_gen_conditioning = torch.where(torch.rand((c.shape[0], 1), device=c.device) < swapping_prob, c_swapped, c)
        else:
            c_gen_conditioning = torch.zeros_like(c)

        ws = self.G_eg3d.mapping(z, c_gen_conditioning, update_emas=update_emas)
        eg3d_output = self.G_eg3d.synthesis(ws, c_gen_conditioning, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas)
        ws = self.G.mapping(ws[:, 0, :], c_gen_conditioning, update_emas=update_emas)
        if require_grad:
            ws.requires_grad_(True)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        gen_output = self.G.synthesis(ws, c, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas)
        
        return gen_output, eg3d_output, ws

    def run_D(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                    torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear', antialias=True)],
                                                    dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear', antialias=True)

        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        real_img = {'image': real_img, 'image_raw': real_img_raw}

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, eg3d_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution)
                loss_feature = torch.nn.functional.smooth_l1_loss(gen_img['feature_image'], eg3d_img['feature_image'])
                loss_vgg_LR = self.vggloss(gen_img['image_raw'], eg3d_img['image_raw'])
                loss_vgg_HR = self.vggloss(gen_img['image'], eg3d_img['image'])
                training_stats.report('Loss/G/loss_feature', loss_feature)
                training_stats.report('Loss/G/loss_vgg_LR', loss_vgg_LR)
                training_stats.report('Loss/G/loss_vgg_HR', loss_vgg_HR)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_feature + loss_vgg_LR + loss_vgg_HR + loss_Gmain * 0.1).mean().mul(gain).backward()

        # Gmain: Maximize logits for generated images.
            # with torch.autograd.profiler.record_function('Gmain_forward'):
            #     gen_img, eg3d_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution)
            #     gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
            #     training_stats.report('Loss/scores/fake', gen_logits)
            #     training_stats.report('Loss/signs/fake', gen_logits.sign())
            #     loss_Gmain = torch.nn.functional.softplus(-gen_logits)
            #     training_stats.report('Loss/G/loss', loss_Gmain)
            # with torch.autograd.profiler.record_function('Gmain_backward'):
            #     (loss_Gmain).mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        # if ['Greg', 'Gboth']:
        #     with torch.autograd.profiler.record_function('Gpl_forward'):
        #         gen_img, eg3d_img, gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, require_grad=True)
        #         pl_noise = torch.randn_like(gen_img['image']) / np.sqrt(gen_img['image'].shape[2] * gen_img['image'].shape[3])
        #         with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
        #             pl_grads = torch.autograd.grad(outputs=[(gen_img['image'] * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
        #         pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
        #         pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
        #         self.pl_mean.copy_(pl_mean.detach())
        #         pl_penalty = (pl_lengths - pl_mean).square()
        #         training_stats.report('Loss/pl_penalty', pl_penalty)
        #         loss_Gpl = pl_penalty * self.pl_weight
        #         training_stats.report('Loss/G/reg', loss_Gpl)
        #     with torch.autograd.profiler.record_function('Gpl_backward'):
        #         (gen_img['image'][:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # # Density Regularization
        # if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'l1':
        #     if swapping_prob is not None:
        #         c_swapped = torch.roll(gen_c.clone(), 1, 0)
        #         c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
        #     else:
        #         c_gen_conditioning = torch.zeros_like(gen_c)

        #     ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
        #     if self.style_mixing_prob > 0:
        #         with torch.autograd.profiler.record_function('style_mixing'):
        #             cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
        #             cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
        #             ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        #     initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
        #     perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * self.G.rendering_kwargs['density_reg_p_dist']
        #     all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
        #     sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
        #     sigma_initial = sigma[:, :sigma.shape[1]//2]
        #     sigma_perturbed = sigma[:, sigma.shape[1]//2:]

        #     TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
        #     TVloss.mul(gain).backward()

        # # Alternative density regularization
        # if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-detach':
        #     if swapping_prob is not None:
        #         c_swapped = torch.roll(gen_c.clone(), 1, 0)
        #         c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
        #     else:
        #         c_gen_conditioning = torch.zeros_like(gen_c)

        #     ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

        #     initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front

        #     perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
        #     all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
        #     sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
        #     sigma_initial = sigma[:, :sigma.shape[1]//2]
        #     sigma_perturbed = sigma[:, sigma.shape[1]//2:]

        #     monotonic_loss = torch.relu(sigma_initial.detach() - sigma_perturbed).mean() * 10
        #     monotonic_loss.mul(gain).backward()


        #     if swapping_prob is not None:
        #         c_swapped = torch.roll(gen_c.clone(), 1, 0)
        #         c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
        #     else:
        #         c_gen_conditioning = torch.zeros_like(gen_c)

        #     ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
        #     if self.style_mixing_prob > 0:
        #         with torch.autograd.profiler.record_function('style_mixing'):
        #             cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
        #             cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
        #             ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        #     initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
        #     perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
        #     all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
        #     sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
        #     sigma_initial = sigma[:, :sigma.shape[1]//2]
        #     sigma_perturbed = sigma[:, sigma.shape[1]//2:]

        #     TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
        #     TVloss.mul(gain).backward()

        # # Alternative density regularization
        # if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-fixed':
        #     if swapping_prob is not None:
        #         c_swapped = torch.roll(gen_c.clone(), 1, 0)
        #         c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
        #     else:
        #         c_gen_conditioning = torch.zeros_like(gen_c)

        #     ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

        #     initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front

        #     perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
        #     all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
        #     sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
        #     sigma_initial = sigma[:, :sigma.shape[1]//2]
        #     sigma_perturbed = sigma[:, sigma.shape[1]//2:]

        #     monotonic_loss = torch.relu(sigma_initial - sigma_perturbed).mean() * 10
        #     monotonic_loss.mul(gain).backward()


        #     if swapping_prob is not None:
        #         c_swapped = torch.roll(gen_c.clone(), 1, 0)
        #         c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
        #     else:
        #         c_gen_conditioning = torch.zeros_like(gen_c)

        #     ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
        #     if self.style_mixing_prob > 0:
        #         with torch.autograd.profiler.record_function('style_mixing'):
        #             cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
        #             cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
        #             ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        #     initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
        #     perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
        #     all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
        #     sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
        #     sigma_initial = sigma[:, :sigma.shape[1]//2]
        #     sigma_perturbed = sigma[:, sigma.shape[1]//2:]

        #     TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
        #     TVloss.mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, eg3d_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                (loss_Dgen).mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw}

                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_raw = r1_grads[1]
                        r1_penalty = r1_grads_image.square().sum([1,2,3]) + r1_grads_image_raw.square().sum([1,2,3])
                    else: # single discrimination
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                        r1_penalty = r1_grads_image.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
