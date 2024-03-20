import os
import sys
import time

import lpips
import nerfacc
import numpy as np
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import tqdm
from easydict import EasyDict as edict

import camera
import util
import util_vis
from external.pohsun_ssim import pytorch_ssim
from util import debug, log

from . import base

class Model(base.Model):
    def __init__(self, opt):
        super().__init__(opt)
        self.lpips_loss = lpips.LPIPS(net="alex").to(opt.device)

    def load_dataset(self, opt, eval_split="val"):
        super().load_dataset(opt, eval_split=eval_split)
        # prefetch all training data
        self.train_data.prefetch_all_data(opt)
        self.train_data.all = edict(
            util.move_to_device(self.train_data.all, opt.device)
        )

    def setup_optimizer(self, opt):
        log.info("setting up optimizers...")
        optimizer = getattr(torch.optim, opt.optim.algo)
        self.optim = optimizer(
            [dict(params=self.graph.nerf.parameters(), lr=opt.optim.lr)]
        )
        if opt.nerf.fine_sampling:
            self.optim.add_param_group(
                dict(params=self.graph.nerf_fine.parameters(), lr=opt.optim.lr)
            )
        # set up scheduler
        if opt.optim.sched:
            scheduler = getattr(torch.optim.lr_scheduler, opt.optim.sched.type)
            if opt.optim.lr_end:
                assert opt.optim.sched.type == "ExponentialLR"
                opt.optim.sched.gamma = (opt.optim.lr_end / opt.optim.lr) ** (
                    1.0 / opt.max_iter
                )
            kwargs = {k: v for k, v in opt.optim.sched.items() if k != "type"}
            self.sched = scheduler(self.optim, **kwargs)

    def train(self, opt):
        # before training
        log.title("TRAINING START")
        self.timer = edict(start=time.time(), it_mean=None)
        self.graph.train()
        self.ep = 0  # dummy for timer
        # training
        # if self.iter_start == 0:
        #     self.validate(opt, 0)
        loader = tqdm.trange(opt.max_iter, desc="training", leave=False)
        for self.it in loader:
            if self.it < self.iter_start:
                continue
            # set var to all available images
            var = self.train_data.all
            self.train_iteration(opt, var, loader)
            if opt.optim.sched:
                self.sched.step()
            if self.it % opt.freq.val == 0:
                self.validate(opt, self.it)
            if self.it % opt.freq.ckpt == 0:
                self.save_checkpoint(opt, ep=None, it=self.it)
                
            # upsample the resolution
            if self.it in self.graph.nerf.upsample_list:
                # upsample the grid resolution
                n_voxels = self.graph.nerf.upsample_n_voxel_list.pop(0)
                current_resolution=self.graph.nerf.N_to_resolution(n_voxels, self.graph.nerf.occ_aabb)
                self.graph.nerf.upsample_grid(current_resolution)
                # register the optimizer for new tensor
                optimizer = getattr(torch.optim, opt.optim.algo)
                self.optim = optimizer(
                    [dict(params=self.graph.nerf.parameters(), lr=opt.optim.lr)]
                )
                if opt.optim.sched:
                    scheduler = getattr(torch.optim.lr_scheduler, opt.optim.sched.type)
                    if opt.optim.lr_end:
                        assert opt.optim.sched.type == "ExponentialLR"
                        opt.optim.sched.gamma = (opt.optim.lr_end / opt.optim.lr) ** (
                            1.0 / (opt.max_iter-self.it)
                        )
                    kwargs = {k: v for k, v in opt.optim.sched.items() if k != "type"}
                    self.sched = scheduler(self.optim, **kwargs)
                        
        # after training
        if opt.tb:
            self.tb.flush()
            self.tb.close()
        if opt.visdom:
            self.vis.close()
        log.title("TRAINING DONE")

    @torch.no_grad()
    def log_scalars(self, opt, var, loss, metric=None, step=0, split="train"):
        super().log_scalars(
            opt, var, loss, metric=metric, step=step, split=split
        )
        # log learning rate
        if split == "train":
            lr = self.optim.param_groups[0]["lr"]
            self.tb.add_scalar("{0}/{1}".format(split, "lr"), lr, step)
            if opt.nerf.fine_sampling:
                lr = self.optim.param_groups[1]["lr"]
                self.tb.add_scalar("{0}/{1}".format(split, "lr_fine"), lr, step)
        # compute PSNR
        psnr = -10 * loss.render.log10()
        self.tb.add_scalar("{0}/{1}".format(split, "PSNR"), psnr, step)
        if opt.nerf.fine_sampling:
            psnr = -10 * loss.render_fine.log10()
            self.tb.add_scalar("{0}/{1}".format(split, "PSNR_fine"), psnr, step)

    @torch.no_grad()
    def visualize(self, opt, var, step=0, split="train", eps=1e-10):
        if opt.tb:
            util_vis.tb_image(opt, self.tb, step, split, "image", var.image)
            if not opt.nerf.rand_rays or split != "train":
                invdepth = (
                    (1 - var.depth) / var.opacity
                    if opt.camera.ndc
                    else 1 / (var.depth / var.opacity + eps)
                )
                rgb_map = var.rgb.view(-1, opt.H, opt.W, 3).permute(
                    0, 3, 1, 2
                )  # [B,3,H,W]
                invdepth_map = invdepth.view(-1, opt.H, opt.W, 1).permute(
                    0, 3, 1, 2
                )  # [B,1,H,W]
                util_vis.tb_image(opt, self.tb, step, split, "rgb", rgb_map)
                util_vis.tb_image(
                    opt, self.tb, step, split, "invdepth", invdepth_map
                )
                if opt.nerf.fine_sampling:
                    invdepth = (
                        (1 - var.depth_fine) / var.opacity_fine
                        if opt.camera.ndc
                        else 1 / (var.depth_fine / var.opacity_fine + eps)
                    )
                    rgb_map = var.rgb_fine.view(-1, opt.H, opt.W, 3).permute(
                        0, 3, 1, 2
                    )  # [B,3,H,W]
                    invdepth_map = invdepth.view(-1, opt.H, opt.W, 1).permute(
                        0, 3, 1, 2
                    )  # [B,1,H,W]
                    util_vis.tb_image(
                        opt, self.tb, step, split, "rgb_fine", rgb_map
                    )
                    util_vis.tb_image(
                        opt, self.tb, step, split, "invdepth_fine", invdepth_map
                    )

    @torch.no_grad()
    def get_all_training_poses(self, opt):
        # get ground-truth (canonical) camera poses
        pose_GT = self.train_data.get_all_camera_poses(opt).to(opt.device)
        return None, pose_GT

    @torch.no_grad()
    def evaluate_full(self, opt, eps=1e-10):
        self.graph.eval()
        loader = tqdm.tqdm(self.test_loader, desc="evaluating", leave=False)
        res = []
        test_path = "{}/test_view".format(opt.output_path)
        os.makedirs(test_path, exist_ok=True)
        for i, batch in enumerate(loader):
            var = edict(batch)
            var = util.move_to_device(var, opt.device)
            if opt.model == "barf" and opt.optim.test_photo:
                # run test-time optimization to factorize imperfection in optimized poses from view synthesis evaluation
                var = self.evaluate_test_time_photometric_optim(opt, var)
            var = self.graph.forward(opt, var, mode="eval")
            # evaluate view synthesis
            invdepth = (
                (1 - var.depth) / var.opacity
                if opt.camera.ndc
                else 1 / (var.depth / var.opacity + eps)
            )
            rgb_map = var.rgb.view(-1, opt.H, opt.W, 3).permute(
                0, 3, 1, 2
            )  # [B,3,H,W]
            invdepth_map = invdepth.view(-1, opt.H, opt.W, 1).permute(
                0, 3, 1, 2
            )  # [B,1,H,W]
            psnr = -10 * self.graph.MSE_loss(rgb_map, var.image).log10().item()
            ssim = pytorch_ssim.ssim(rgb_map, var.image).item()
            lpips = self.lpips_loss(rgb_map * 2 - 1, var.image * 2 - 1).item()
            res.append(edict(psnr=psnr, ssim=ssim, lpips=lpips))
            # dump novel views
            torchvision_F.to_pil_image(rgb_map.cpu()[0]).save(
                "{}/rgb_{}.png".format(test_path, i)
            )
            torchvision_F.to_pil_image(var.image.cpu()[0]).save(
                "{}/rgb_GT_{}.png".format(test_path, i)
            )
            torchvision_F.to_pil_image(invdepth_map.cpu()[0]).save(
                "{}/depth_{}.png".format(test_path, i)
            )
        # show results in terminal
        print("--------------------------")
        print("PSNR:  {:8.2f}".format(np.mean([r.psnr for r in res])))
        print("SSIM:  {:8.2f}".format(np.mean([r.ssim for r in res])))
        print("LPIPS: {:8.2f}".format(np.mean([r.lpips for r in res])))
        print("--------------------------")
        # dump numbers to file
        quant_fname = "{}/quant.txt".format(opt.output_path)
        with open(quant_fname, "w") as file:
            for i, r in enumerate(res):
                file.write("{} {} {} {}\n".format(i, r.psnr, r.ssim, r.lpips))

    @torch.no_grad()
    def generate_videos_synthesis(self, opt, eps=1e-10):
        self.graph.eval()
        if opt.data.dataset == "blender":
            test_path = "{}/test_view".format(opt.output_path)
            # assume the test view synthesis are already generated
            print("writing videos...")
            rgb_vid_fname = "{}/test_view_rgb.mp4".format(opt.output_path)
            depth_vid_fname = "{}/test_view_depth.mp4".format(opt.output_path)
            os.system(
                "ffmpeg -y -framerate 30 -i {0}/rgb_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(
                    test_path, rgb_vid_fname
                )
            )
            os.system(
                "ffmpeg -y -framerate 30 -i {0}/depth_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(
                    test_path, depth_vid_fname
                )
            )
        else:
            pose_pred, pose_GT = self.get_all_training_poses(opt)
            poses = pose_pred if opt.model == "barf" else pose_GT
            if opt.model == "barf" and opt.data.dataset == "llff":
                _, sim3 = self.prealign_cameras(opt, pose_pred, pose_GT)
                scale = sim3.s1 / sim3.s0
            else:
                scale = 1
            # rotate novel views around the "center" camera of all poses
            idx_center = (
                (poses - poses.mean(dim=0, keepdim=True))[..., 3]
                .norm(dim=-1)
                .argmin()
            )
            pose_novel = camera.get_novel_view_poses(
                opt, poses[idx_center], N=60, scale=scale
            ).to(opt.device)
            # render the novel views
            novel_path = "{}/novel_view".format(opt.output_path)
            os.makedirs(novel_path, exist_ok=True)
            pose_novel_tqdm = tqdm.tqdm(
                pose_novel, desc="rendering novel views", leave=False
            )
            intr = (
                edict(next(iter(self.test_loader))).intr[:1].to(opt.device)
            )  # grab intrinsics
            for i, pose in enumerate(pose_novel_tqdm):
                ret = (
                    self.graph.render_by_slices(opt, pose[None], intr=intr)
                    if opt.nerf.rand_rays
                    else self.graph.render(opt, pose[None], intr=intr)
                )
                invdepth = (
                    (1 - ret.depth) / ret.opacity
                    if opt.camera.ndc
                    else 1 / (ret.depth / ret.opacity + eps)
                )
                rgb_map = ret.rgb.view(-1, opt.H, opt.W, 3).permute(
                    0, 3, 1, 2
                )  # [B,3,H,W]
                invdepth_map = invdepth.view(-1, opt.H, opt.W, 1).permute(
                    0, 3, 1, 2
                )  # [B,1,H,W]
                torchvision_F.to_pil_image(rgb_map.cpu()[0]).save(
                    "{}/rgb_{}.png".format(novel_path, i)
                )
                torchvision_F.to_pil_image(invdepth_map.cpu()[0]).save(
                    "{}/depth_{}.png".format(novel_path, i)
                )
            # write videos
            print("writing videos...")
            rgb_vid_fname = "{}/novel_view_rgb.mp4".format(opt.output_path)
            depth_vid_fname = "{}/novel_view_depth.mp4".format(opt.output_path)
            os.system(
                "ffmpeg -y -framerate 30 -i {0}/rgb_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(
                    novel_path, rgb_vid_fname
                )
            )
            os.system(
                "ffmpeg -y -framerate 30 -i {0}/depth_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(
                    novel_path, depth_vid_fname
                )
            )
            
            
            

class Graph(base.Graph):
    def __init__(self, opt):
        super().__init__(opt)
        self.nerf = NeRF(opt)
        if opt.nerf.fine_sampling:
            self.nerf_fine = NeRF(opt)
            
    def forward(self, opt, var, mode=None):
        batch_size = len(var.idx)
        pose = self.get_pose(opt, var, mode=mode)
        # render images
        if opt.nerf.rand_rays and mode in ["train", "test-optim"]:
            # sample random rays for optimization
            var.ray_idx = torch.randperm(opt.H * opt.W, device=opt.device)[
                : opt.nerf.rand_rays // batch_size
            ]
            ret = self.render(
                opt, pose, intr=var.intr, ray_idx=var.ray_idx, mode=mode
            )  # [B,N,3],[B,N,1]
        else:
            # render full image (process in slices)
            ret = (
                self.render_by_slices(opt, pose, intr=var.intr, mode=mode)
                if opt.nerf.rand_rays
                else self.render(opt, pose, intr=var.intr, mode=mode)
            )  # [B,HW,3],[B,HW,1]
        var.update(ret)
        return var
    def compute_loss(self, opt, var, mode=None):
        loss = edict()
        batch_size = len(var.idx)
        image = var.image.view(batch_size, 3, opt.H * opt.W).permute(0, 2, 1)
        if opt.nerf.rand_rays and mode in ["train", "test-optim"]:
            image = image[:, var.ray_idx]
        # compute image losses
        if opt.loss_weight.render is not None:
            loss.render = self.MSE_loss(var.rgb, image)
        if opt.loss_weight.render_fine is not None:
            assert opt.nerf.fine_sampling
            loss.render_fine = self.MSE_loss(var.rgb_fine, image)
        return loss
    def get_pose(self, opt, var, mode=None):
        return var.pose
    def render(self, opt, pose, intr=None, ray_idx=None, mode=None):
        batch_size = len(pose)
        center, ray = camera.get_center_and_ray(
            opt, pose, intr=intr, ray_idx=ray_idx
        )  # [B,HW,3]
        while (
            ray.isnan().any()
        ):  # TODO: weird bug, ray becomes NaN arbitrarily if batch_size>1, not deterministic reproducible
            center, ray = camera.get_center_and_ray(
                opt, pose, intr=intr, ray_idx=ray_idx
            )  # [B,HW,3]
        # if ray_idx is not None:
        #     # consider only subset of rays
        #     center,ray = center[:,ray_idx],ray[:,ray_idx]
        if opt.camera.ndc:
            # convert center/ray representations to NDC
            center, ray = camera.convert_NDC(opt, center, ray, intr=intr)
        # render with main MLP
        rgb = depth = opacity = None
        if self.nerf.use_occ_grid:
            assert not opt.nerf.fine_sampling
            flatten_center = center.reshape(-1, 3)
            flatten_ray = torch_F.normalize(ray.reshape(-1, 3), dim=-1)

            def sigma_fn(t_starts, t_ends, ray_indices):
                if ray_indices.shape[0] == 0:
                    return torch.zeros((0,), device=ray_indices.device)
                t_origins = flatten_center[ray_indices]
                t_dirs = flatten_ray[ray_indices]
                positions = (
                    t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
                )
                return self.nerf(
                    opt, positions, t_dirs, mode=mode, density_only=True
                )

            def rgb_sigma_fn(t_starts, t_ends, ray_indices):
                if ray_indices.shape[0] == 0:
                    return torch.zeros(
                        (0, 3), device=ray_indices.device
                    ), torch.zeros((0,), device=ray_indices.device)
                t_origins = flatten_center[ray_indices]
                t_dirs = flatten_ray[ray_indices]
                positions = (
                    t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
                )
                rgb, sigma = self.nerf(opt, positions, t_dirs, mode=mode)
                return rgb, sigma

            ray_indices, t_starts, t_ends = self.nerf.occ_grid.sampling(
                flatten_center,
                flatten_ray,
                sigma_fn=sigma_fn,
                near_plane=opt.nerf.depth.range[0],
                far_plane=opt.nerf.depth.range[1],
                render_step_size=self.nerf.occ_step_size,
                stratified=mode == "train",
                alpha_thre=self.nerf.occ_alpha_thres,
            )
            rgb, opacity, depth, _ = nerfacc.rendering(
                t_starts,
                t_ends,
                ray_indices=ray_indices,
                n_rays=flatten_center.shape[0],
                rgb_sigma_fn=rgb_sigma_fn,
                render_bkgd=opt.data.bgcolor,
            )
            rgb, opacity, depth = [
                t.reshape(batch_size, -1, t.shape[-1])
                for t in [rgb, opacity, depth]
            ]
        else:
            depth_samples = self.sample_depth(
                opt, batch_size, num_rays=ray.shape[1]
            )  # [B,HW,N,1]
            # NOTE(Hang Gao @ 02/25): center -> rays_o, ray -> rays_d.
            rgb_samples, density_samples = self.nerf.forward_samples(
                opt, center, ray, depth_samples, mode=mode
            )
            rgb, depth, opacity, prob = self.nerf.composite(
                opt, ray, rgb_samples, density_samples, depth_samples
            )
        ret = edict(rgb=rgb, depth=depth, opacity=opacity)  # [B,HW,K]
        # render with fine MLP from coarse MLP
        if opt.nerf.fine_sampling:
            with torch.no_grad():
                # resample depth acoording to coarse empirical distribution
                depth_samples_fine = self.sample_depth_from_pdf(
                    opt, pdf=prob[..., 0]
                )  # [B,HW,Nf,1]
                depth_samples = torch.cat(
                    [depth_samples, depth_samples_fine], dim=2
                )  # [B,HW,N+Nf,1]
                depth_samples = depth_samples.sort(dim=2).values
            rgb_samples, density_samples = self.nerf_fine.forward_samples(
                opt, center, ray, depth_samples, mode=mode
            )
            rgb_fine, depth_fine, opacity_fine, _ = self.nerf_fine.composite(
                opt, ray, rgb_samples, density_samples, depth_samples
            )
            ret.update(
                rgb_fine=rgb_fine,
                depth_fine=depth_fine,
                opacity_fine=opacity_fine,
            )  # [B,HW,K]
        return ret

    def render_by_slices(self, opt, pose, intr=None, mode=None):
        ret_all = edict(rgb=[], depth=[], opacity=[])
        if opt.nerf.fine_sampling:
            ret_all.update(rgb_fine=[], depth_fine=[], opacity_fine=[])
        # render the image by slices for memory considerations
        for c in range(0, opt.H * opt.W, opt.nerf.rand_rays):
            ray_idx = torch.arange(
                c, min(c + opt.nerf.rand_rays, opt.H * opt.W), device=opt.device
            )
            ret = self.render(
                opt, pose, intr=intr, ray_idx=ray_idx, mode=mode
            )  # [B,R,3],[B,R,1]
            for k in ret:
                ret_all[k].append(ret[k])
        # group all slices of images
        for k in ret_all:
            ret_all[k] = torch.cat(ret_all[k], dim=1)
        return ret_all
    
    
class NeRF(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.occ_grid_reso = opt.nerf.get("occ_grid_reso", -1)
        self.use_occ_grid = self.occ_grid_reso > 0
        self.occ_grid = None
        self.occ_aabb = torch.FloatTensor(opt.nerf.get(
            "occ_aabb", [-1.5, -1.5, -1.5, 1.5, 1.5, 1.5]
        ))
        self.occ_step_size = opt.nerf.get("occ_step_size", 5e-3)
        self.occ_alpha_thres = opt.nerf.get("occ_alpha_thres", 0.0)
        self.occ_thres = opt.nerf.get("occ_thres", 0.01)
        self.occ_ema_decay = opt.nerf.get("occ_ema_decay", 0.95)
        if self.use_occ_grid > 0:
            self.occ_grid = nerfacc.OccGridEstimator(
                roi_aabb=self.occ_aabb, resolution=self.occ_grid_reso
            )

            def occ_eval_fn(opt, x, mode):
                density = self(opt, x, mode=mode, density_only=True)
                return density * self.occ_step_size

            self.occ_eval_fn = occ_eval_fn

        # tensoRF
        self.init_scale=0.1
        self.color_feat_dim=opt.nerf.get("color_feat_dim", 27)
        self.color_n_comp=opt.nerf.get("color_n_comp", [16,16,16])
        self.density_n_comp=opt.nerf.get("density_n_comp", [16,16,16])
        self.N_voxel_init=opt.nerf.get("N_voxel_init", 128**3)
        self.N_voxel_final=opt.nerf.get("N_voxel_final", 640**3)
        self.upsample_list=opt.nerf.get("upsample_list", [2000,3000,4000,5500])
        self.upsample_n_voxel_list = (torch.round(torch.exp(torch.linspace(
            np.log(self.N_voxel_init),np.log(self.N_voxel_final),len(self.upsample_list) + 1))).long()).tolist()[1:]
        self.grid_size=self.N_to_resolution(self.N_voxel_init, self.occ_aabb)
        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        
        self.define_network(opt)
    def N_to_resolution(self,n_voxels, aabb):
        '''
        input : 
            n_voxels : int number of total voxels
            aabb : aabb bbox size
        output:
            reso : list of int, [x,y,z] resolution of voxel grid
        '''
        xyz_min, xyz_max = aabb.split(3)
        dim = len(xyz_min)
        voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
        return ((xyz_max - xyz_min) / voxel_size).long().tolist()
    def init_VM_decomposition(self, n_component, grid_size,init_scale):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                init_scale * torch.randn((1, n_component[i], grid_size[mat_id_1], grid_size[mat_id_0]))))  #
            line_coef.append(
                torch.nn.Parameter(init_scale * torch.randn((1, n_component[i], grid_size[vec_id], 1))))
        return torch.nn.ParameterList(plane_coef), torch.nn.ParameterList(line_coef)
    
    def define_network(self, opt):
        # TensorRF decompose tensor
        self.density_plane, self.density_line = self.init_VM_decomposition(self.density_n_comp, self.grid_size, 0.1) # list[3] [1,n_comp,grid_size,grid_size]  [1,n_comp,grid_size,1]
        self.color_plane, self.color_line = self.init_VM_decomposition(self.color_n_comp, self.grid_size, 0.1) # list[3] [1,n_comp,grid_size,grid_size]  [1,n_comp,grid_size,1]
        self.basis_mat = torch.nn.Linear(sum(self.color_n_comp), self.color_feat_dim, bias=False) # [3*color_n_comp, app_dim]
        # color network
        in_mlpC = (3+2*6*3)  + self.color_feat_dim #
        self.color_network = torch.nn.Sequential(
            torch.nn.Linear(in_mlpC, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 3)
        )
        
    def tensorflow_init_weights(self, opt, linear, out=None):
        raise NotImplementedError

    def query_density_feature(self, xyz):
        # plane + line basis
        coordinate_plane = torch.stack((xyz[..., self.matMode[0]], xyz[..., self.matMode[1]], xyz[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz[..., self.vecMode[0]], xyz[..., self.vecMode[1]], xyz[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        sigma_feature = torch.zeros((xyz.shape[0],), device=xyz.device)
        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = torch_F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz.shape[:1])
            line_coef_point = torch_F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz.shape[:1])
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature

    def query_color_feature(self, xyz):
        
        # plane + line basis
        coordinate_plane = torch.stack((xyz[..., self.matMode[0]], xyz[..., self.matMode[1]], xyz[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz[..., self.vecMode[0]], xyz[..., self.vecMode[1]], xyz[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(len(self.color_plane)):
            plane_coef_point.append(torch_F.grid_sample(self.color_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz.shape[:1]))
            line_coef_point.append(torch_F.grid_sample(self.color_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)


        return self.basis_mat((plane_coef_point * line_coef_point).T)
    def forward(
        self, opt, points_3D, ray_unit=None, mode=None, density_only=False
    ):  # [B,...,3]
        density=self.query_density_feature(points_3D)
        if density_only==True:
            return density
        
        color_feat=self.query_color_feature(points_3D)
        if opt.nerf.view_dep:
            assert ray_unit is not None
            ray_enc = self.positional_encoding(
                opt, ray_unit, L=6
            )
            ray_enc = torch.cat([ray_unit, ray_enc], dim=-1)  # [B,...,6L+3]
            feat = torch.cat([color_feat, ray_enc], dim=-1)
        feat=self.color_network(feat)
        rgb = feat.sigmoid_()  # [B,...,3]
        return rgb, density
    def positional_encoding(self, opt, input, L):  # [B,...,N]
        shape = input.shape
        freq = (
            2 ** torch.arange(L, dtype=torch.float32, device=opt.device) * np.pi
        )  # [L]
        spectrum = input[..., None] * freq  # [B,...,N,L]
        sin, cos = spectrum.sin(), spectrum.cos()  # [B,...,N,L]
        input_enc = torch.stack([sin, cos], dim=-2)  # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1], -1)  # [B,...,2NL]
        return input_enc
    
    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, resolution):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                torch_F.interpolate(plane_coef[i].data, size=(resolution[mat_id_1], resolution[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                torch_F.interpolate(line_coef[i].data, size=(resolution[vec_id], 1), mode='bilinear', align_corners=True))
        return plane_coef, line_coef
    @torch.no_grad()
    def upsample_grid(self,resolution):
        self.color_plane, self.color_line = self.up_sampling_VM(self.color_plane, self.color_line, resolution)
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, resolution)
        print(f'upsamping to {resolution}')