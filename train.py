#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import torch
import torchvision
import mmcv
from random import randint
from torch import Tensor

from utils.loss_utils import l1_loss, ssim, tv_loss, def_reg_loss, perceptual_loss, edge_aware_smoothness_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, depth2normal, normal2rgb, normal2curv, depth2rgb
from utils.initial_utils import imread
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import cos_loss
from torchvision.utils import save_image
import torch.nn.functional as F
import numpy as np
from PIL import Image


try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel()
    deform.train_setting(opt)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    if dataset.accurate_mask:
        invisible_mask_path = os.path.join(dataset.source_path, "dilated_invisible_mask.png")
        inpaint_mask = imread(invisible_mask_path) / 255.0
        np.savetxt("inpaint_mask.txt", inpaint_mask.astype(int), fmt='%d')
        inpaint_mask_tensor = torch.tensor(inpaint_mask, dtype=torch.float32, device="cuda")
        np.savetxt("inpaint_mask_tensor.txt", inpaint_mask_tensor.cpu().numpy().astype(int), fmt='%d')

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0

    progress_bar = tqdm(range(opt.iterations), desc="Training progress")

    for iteration in range(1, opt.iterations + 1):

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        N = gaussians.get_xyz.shape[0]
        time_input = fid.unsqueeze(0).expand(N, -1)
        d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input)

        # Render
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling)
        image, viewspace_point_tensor, visibility_filter, radii, depth, norm, alpha = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"], render_pkg_re["depth"], render_pkg_re["normal"], render_pkg_re["alpha"]

        image1 = image

        mask_vis = (alpha.detach() > 1e-5)
        normal1 = torch.nn.functional.normalize(norm, dim=0) * mask_vis
        mono = viewpoint_cam.mono if dataset.mono_normal else None


        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_image1 = gt_image
        if dataset.is_mask:
            mask = viewpoint_cam.mask.unsqueeze(0).cuda()
            gt_image = gt_image * mask

            if mono is not None:
                mono *= mask
                monoN = mono[:3]
                # monoD = mono[3:]
                #depth_ = depth2rgb(depth_np, mask_vis)
                # monoD_match, mask_match = match_depth(monoD, depth_, mask * mask_vis, 256, [viewpoint_cam.image_height, viewpoint_cam.image_width])

                loss_monoN = cos_loss(normal1, monoN, weight=mask)
                # loss_depth = l1_loss(depth_ * mask_match, monoD_match)

            #Image.fromarray((gt_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)).save('gt_image.png')
            #Image.fromarray((image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)).save('rendered_image.png')

            if dataset.accurate_mask:
                img_tv_loss = tv_loss(image * inpaint_mask_tensor)
            else:
                img_tv_loss = tv_loss(image * (1-mask))

            image = image * mask
        Ll1 = l1_loss(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if opt.lambda_smooth != 0:
            loss += opt.lambda_smooth * img_tv_loss

        # # 1. Mask Loss
        # # Purpose: Penalize predicted opacity in regions where surgical tools are present (masked out in `mask`)
        # mask_gt = mask  # Ground truth mask where surgical tools are masked out
        # loss_mask = (alpha * (1 - mask_gt)).mean()
        # loss += 0.1 * loss_mask  # Adjust the weight as needed
        #
        # # 2. Visibility Mask
        # # Purpose: Identify regions where the model predicts the object to be visible
        # mask_vis = (alpha.detach() > 1e-5)
        #
        # 3. Surface Normal Consistency Loss
        # Purpose: Ensure predicted normals are consistent with normals derived from depth
        # Compute normals from depth
        #d2n = depth2normal(depth, mask_vis, viewpoint_cam)
        # Compute cosine similarity loss between predicted normals and depth-derived normals
        #loss_surface = cos_loss(norm, d2n)
        #loss += (0.01 + 0.1 * min(2 * iteration / opt.iterations, 1)) * loss_surface  # Adjust the weight as needed


        depth_loss = None
        if dataset.is_depth:
            gt_depth = viewpoint_cam.depth.unsqueeze(0).cuda()
            mask_depth = viewpoint_cam.mask_depth.unsqueeze(0).cuda()
            depth1 = depth * mask_depth
            depth_loss = l1_loss(depth1, gt_depth)

            #smoothness_loss_val = edge_aware_smoothness_loss(depth, gt_image)
            #loss += opt.lambda_smoothness * smoothness_loss_val

            loss += 0.001 * depth_loss


        # Perceptual loss
        perceptual_loss_val = perceptual_loss(image, gt_image)
        loss += opt.lambda_perceptual * perceptual_loss_val


        #normal and depth loss
        if mono is not None:
            loss += (0.04 - ((iteration / opt.iterations)) * 0.02) * loss_monoN
            # loss += 0.01 * loss_depth

        # deformation loss
        loss_pos, loss_cov = def_reg_loss(scene.gaussians, d_xyz, d_rotation, d_scaling)

        loss += opt.lambda_pos * loss_pos
        loss += opt.lambda_cov * loss_cov

        loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])


            if (iteration - 1) % 1000 == 0:
                normal_wrt = normal2rgb(normal1, mask_vis)
                depth_wrt = depth2rgb(depth, mask_vis)
                img_wrt = torch.cat([gt_image1, image1, normal_wrt * alpha, depth_wrt * alpha], 2)
                os.makedirs('test', exist_ok=True)
                save_image(img_wrt.cpu(), f'test/test.png')


            # Log and save
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly, depth_loss)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)


                optimize_path = os.path.join(args.model_path, "optimize/iteration_{}".format(iteration))

                render_path = os.path.join(optimize_path, "train/renders")
                depth_path = os.path.join(optimize_path, "train/depth")
                normal_path = os.path.join(optimize_path, "train/normal")
                rgb_depth_path = os.path.join(optimize_path, "train/rgb_depth")

                os.makedirs(render_path, exist_ok=True)
                os.makedirs(depth_path, exist_ok=True)
                os.makedirs(normal_path, exist_ok=True)
                os.makedirs(rgb_depth_path, exist_ok=True)

                train_view = scene.getTrainCameras().copy()
                for idx, view in enumerate(tqdm(train_view, desc="Rendering progress")):
                    fid = view.fid
                    name = view.colmap_id
                    xyz = gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                    render_pkg_re = render(view, gaussians, pipe, background, d_xyz, d_rotation, d_scaling)
                    rendering = render_pkg_re["render"]
                    depth_np = render_pkg_re["depth"]
                    alpha_np = render_pkg_re["alpha"]
                    mask_vis1 = (alpha_np.detach() > 1e-5)
                    depth2 = depth_np / (depth_np.max() + 1e-5)
                    rgb_depth_wrt = depth2rgb(depth_np, mask_vis1)

                    # --- New Code to Extract and Save Normals ---
                    normals = render_pkg_re["normal"]  # Extract normal maps
                    normals_visual = (normals + 1) / 2  # Normalize normals to [0, 1]
                    torchvision.utils.save_image(normals_visual, os.path.join(normal_path, '{0:05d}'.format(name) + ".png"))

                    torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(name) + ".png"))
                    torchvision.utils.save_image(depth2, os.path.join(depth_path, '{0:05d}'.format(name) + ".png"))
                    torchvision.utils.save_image(rgb_depth_wrt * alpha_np, os.path.join(rgb_depth_path, '{0:05d}'.format(name) + ".png"))

            # Densification
            if iteration < opt.densify_until_iter:  # < 15_000
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def read_config_params(args, config):
    params = ["OptimizationParams", "ModelParams", "PipelineParams"]
    for param in params:
        if param in config.keys():
            for key, value in config[param].items():
                if hasattr(args, key):
                    setattr(args, key, value)
    return args

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, depth_loss=None ):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)

        if depth_loss is not None:
            tb_writer.add_scalar('train_loss_patches/depth_loss', depth_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)


                    if viewpoint.mask is not None:
                        mask = viewpoint.mask.unsqueeze(0).cuda()
                        image = image * mask
                        gt_image = gt_image * mask
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[3000, 5000, 6000, 7_000] + list(range(10000, 60001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[20_000, 40000, 60000])
    
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    config = mmcv.Config.fromfile(args.config)
    args = read_config_params(args, config)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)

    # All done
    print("\nTraining complete.")
