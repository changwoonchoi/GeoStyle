import os
import pathlib
import pymeshlab
import torch
import torchvision
import logging
import yaml
import numpy as np
import nvdiffrast.torch as dr

from easydict import EasyDict
from NeuralJacobianFields import SourceMesh
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from geostyle.guidance.sdxl import StableDiffusionXLGuidance
from geostyle.utilities.video import Video
from geostyle.utilities.camera import FixedDistFOVCameraBatch, get_camera_params
from geostyle.utilities.approx_encoder import generate_img_latent_dataset, fit_approx_encoder
from geostyle.utilities.cage_loss import mesh2obbweights, obbparam2obbweights, get_label2vertices
from geostyle.utilities.symmetry_loss import detect_symmetry, symmetry_loss
from geostyle.utilities.aux_mesh import OBBParam, init_template_sphere, mesh2spheremesh, fps
from geostyle.utilities.helpers_nvdiffrec import create_scene
from nvdiffrec.src import obj
from nvdiffrec.src import util as nvdiffutil
from nvdiffrec.src import mesh
from nvdiffrec.src import render
from nvdiffrec.src import light
from nvdiffrec.src import material

def loop(cfg, train_approx_encoder=False):
    output_path = pathlib.Path(cfg['output_path'])
    os.makedirs(output_path, exist_ok=True)
    with open(output_path / 'config.yml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    cfg = EasyDict(cfg)

    print(f'[!] Output directory {cfg.output_path} created')

    device = torch.device(f'cuda:{cfg.gpu}')
    torch.cuda.set_device(device)

    if not train_approx_encoder:
        sdxl_version = cfg.sd_model.split("_")[-1]
        sds_guidance = StableDiffusionXLGuidance(
            model_type=sdxl_version,
            use_half_precision=bool(cfg.use_half_precision),
            device=device,
            lora_path=cfg.lora_model,
            lora_scale=cfg.lora_scale,
            grad_clamp_val=cfg.grad_clamp_val,
            approx_encoder=cfg.approx_encoder,
            approx_encoder_type=cfg.approx_encoder_type,
            approx_encoder_path=cfg.approx_encoder_path
        )
        print(f"[!] Loaded Stable Diffusion XL model {sdxl_version}")
        print(f"[!] Target text prompt is {cfg.text_prompt}")
    
    video = Video(cfg.output_path)
    glctx = dr.RasterizeGLContext()

    tmp_dir = os.path.join(output_path, 'tmp')
    if os.path.exists(tmp_dir):
        import shutil
        shutil.rmtree(tmp_dir)

    os.makedirs(tmp_dir, exist_ok=False)

    cfg.kd_min              = [ 0.0,  0.0,  0.0,  0.0] 
    cfg.kd_max              = [ 1.0,  1.0,  1.0,  1.0]
    cfg.ks_min              = [ 0.0, 0.08,  0.0]
    cfg.ks_max              = [ 1.0,  1.0,  1.0]
    cfg.nrm_min             = [-1.0, -1.0,  0.0] 
    cfg.nrm_max             = [ 1.0,  1.0,  1.0]

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(cfg.mesh)

    try:
        ms.meshing_remove_duplicate_vertices()
        ms.meshing_remove_unreferenced_vertices()
        p = pymeshlab.PercentageValue(80)
        ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=p)
    except:
        print("[!] Old pymeshlab version")
        pass

    if not ms.current_mesh().has_wedge_tex_coord():
        ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=10000)
    ms.save_current_mesh(str(output_path / 'tmp' / 'mesh.obj'))

    load_mesh = obj.load_obj(str(output_path / 'tmp' / 'mesh.obj'))
    load_mesh = mesh.unit_size(load_mesh)

    if cfg.log_vertex_color:
        vertex_color = (load_mesh.v_pos + 1) / 2.

    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=load_mesh.v_pos.cpu().numpy(), face_matrix=load_mesh.t_pos_idx.cpu().numpy()))
    ms.save_current_mesh(str(output_path / 'tmp' / 'mesh.obj'), save_vertex_color=False)

    if cfg.mtl is not None:
        _mat = material.load_mtl(cfg.mtl)[0]
    else:
        logging.warn("[!] Please specify the material file in the config file, using default material")
        _mat = load_mesh.material

    mat = nvdiffutil.prepare_trainable_material(cfg, mlp=False, init_mat=_mat, optimize_mat=cfg.optimize_textures)
    if cfg.envlight is not None:
        if cfg.random_light_rotation:
            lgts = []
            for _ in range(cfg.num_envlight_rotations):
                rand_rot = np.random.uniform(0, 1)
                lgts.append(light.load_env(cfg.envlight, train_light=False, rotate=rand_rot))
            fixed_lgt = light.load_env(cfg.envlight, train_light=False)
        else:
            fixed_lgt = light.load_env(cfg.envlight, train_light=False)
    else:
        fixed_lgt = light.create_trainable_env_rnd(512, scale=0., bias=0.5)

    jacobian_source = SourceMesh.SourceMesh(0, str(output_path / 'tmp' / 'mesh.obj'), {}, 1, ttype=torch.float)
    if len(list((output_path / 'tmp').glob('*.npz'))) > 0:
        logging.warn(f'[!] Using existing Jacobian .npz files in {str(output_path)}/tmp/ ! Please check if this is intentional.')
    jacobian_source.load()
    jacobian_source.to(device)

    with torch.no_grad():
        gt_jacobians = jacobian_source.jacobians_from_vertices(load_mesh.v_pos.unsqueeze(0))

    if train_approx_encoder:
        n_vert = jacobian_source.vertices_from_jacobians(gt_jacobians).squeeze()
        latent_img_dataset_path = os.path.join(output_path, 'latent_img_dataset')
        generate_img_latent_dataset(cfg, base_mesh=load_mesh, n_vert=n_vert, lgt=lgts if cfg.random_light_rotation_latent_img_dataset else fixed_lgt, mat=mat, output_path=latent_img_dataset_path)
        approx_encoder_path = fit_approx_encoder(cfg, img_latent_path=latent_img_dataset_path, output_path=latent_img_dataset_path)
        print("="*80)
        print("[!] Finished training approximated encoder")
        print("="*80)
        return approx_encoder_path
    
    dist = (cfg.dist_min + cfg.dist_max) / 2
    fov = (cfg.fov_min + cfg.fov_max) / 2
    cams_data = FixedDistFOVCameraBatch(
        cfg.train_res,
        dist, 
        cfg.elev_min,
        cfg.elev_max,
        cfg.azim_min,
        cfg.azim_max,
        fov,
        cfg.aug_loc,
        cfg.aug_light,
        cfg.aug_bkg,
        cfg.batch_size,
        rand_solid=True,
        bkg_color=cfg.bkg_color,
    )

    if cfg.use_cage_loss:
        dist = (cfg.dist_min + cfg.dist_max) / 2
        fov = (cfg.fov_min + cfg.fov_max) / 2
        aux_cams_data = FixedDistFOVCameraBatch(
            cfg.train_res,
            dist, 
            cfg.elev_min,
            cfg.elev_max,
            cfg.azim_min,
            cfg.azim_max,
            fov,
            cfg.aug_loc,
            cfg.aug_light,
            cfg.aug_bkg,
            cfg.batch_size,
            rand_solid=True,
            bkg_color=cfg.bkg_color,
        )
    
    cams = torch.utils.data.DataLoader(cams_data, cfg.batch_size, num_workers=0, pin_memory=True)
    if cfg.use_cage_loss:
        aux_cams = torch.utils.data.DataLoader(aux_cams_data, cfg.batch_size, num_workers=0, pin_memory=True)

    os.makedirs(output_path / 'mesh_final', exist_ok=True)
    os.makedirs(output_path / 'mesh_images', exist_ok=True)
    logger = SummaryWriter(str(output_path / 'logs'))

    t_loop = tqdm(range(cfg.epochs))

    if cfg.use_cage_loss:
        assert cfg.segmented_label_path is not None
        
        label = np.load(cfg.segmented_label_path)
        assert len(load_mesh.t_pos_idx) == len(label)
        faces2label = {}
        for i in range(len(load_mesh.t_pos_idx)):
            assert len(load_mesh.t_pos_idx[i]) == 3
            faces2label[i] = label[i]

        label2vertices = get_label2vertices(faces=load_mesh.t_pos_idx, faces2labels=faces2label)

        assert sum([len(v_lst) for v_lst in label2vertices.values()]) == load_mesh.v_pos.shape[0]
        all_indices = torch.cat([torch.tensor(v) for v in label2vertices.values()])
        assert len(all_indices) == len(torch.unique(all_indices))

        print("="*80)
        print("[!] Loaded faces2label from cached file")
        print("="*80)
    
    if cfg.symmetry_loss_weight > 0:
        with torch.no_grad():
            detected_symmetry = detect_symmetry(mesh_vertices=load_mesh.v_pos)
            if len(detected_symmetry) == 0:
                raise ValueError("No symmetry detected in the mesh. Please check the mesh, adjust threshold, or disable symmetry loss.")
        print("="*80)
        print(f"[!] Detected {len(detected_symmetry)} symmetries in the mesh")
        print("="*80)

    with torch.no_grad():
        gt_jacobians = jacobian_source.jacobians_from_vertices(load_mesh.v_pos.unsqueeze(0))
    gt_jacobians.requires_grad_(True)

    optimizer = torch.optim.Adam([gt_jacobians], lr=cfg.lr)
    w_base = None

    for it in t_loop:
        n_vert = jacobian_source.vertices_from_jacobians(gt_jacobians).squeeze()
        cage_loss_weight = (cfg.cage_loss_weight - cfg.cage_loss_weight / 100) * (1.0 - it / (cfg.cage_loss_epoch - 1)) + cfg.cage_loss_weight / 100 if it < cfg.cage_loss_epoch and cfg.use_cage_loss else 0.0 
        use_cage_loss = True if it < cfg.cage_loss_epoch and cfg.use_cage_loss else False

        if use_cage_loss and w_base is None:
            aux_sampling_method = cfg.aux_sampling_method
            
            _, ref_label2obb = mesh2obbweights(
                vertices=n_vert.detach().clone(), 
                label2vertices=label2vertices, 
                ref_label2obb=None,
                visualize=False
            )
        
            label2obbparam = {
                label: OBBParam(init_corner=obb['corners'].detach(), center=obb['center'].detach(), size=obb['size'].detach(), axes=obb['axes'].detach()) \
                    for label, obb in ref_label2obb.items()
            }
            obb_params = torch.nn.ModuleList(label2obbparam.values())

            os.makedirs(os.path.join(output_path, 'aux_mesh_images'), exist_ok=True)
            os.makedirs(os.path.join(output_path, 'aux_meshes'), exist_ok=True)

            sphere_verts, sphere_faces = init_template_sphere(dtype=n_vert.dtype, device=n_vert.device, subdivisions=0)
            n_V_sample = int(n_vert.shape[0] * cfg.aux_sampling_ratio)
            if aux_sampling_method == "random":
                sampled_vertices_idx = torch.randperm(n_vert.shape[0])[:n_V_sample]
            elif aux_sampling_method == "fps":
                sampled_vertices_idx = fps(n_vert, n_V_sample)
            else:
                raise NotImplementedError()

            if cfg.aux_symmetry_loss_weight > 0:
                for sym in detected_symmetry:
                    all_pairs = [sym['pairs']]
                    all_pairs = torch.cat(all_pairs, dim=0).to(n_vert.device)

                    symmetry_dict = dict()
                    for a1, a2 in all_pairs.cpu().tolist():
                        symmetry_dict[a2] = a1; symmetry_dict[a1] = a2; 
                    
                    unique_idxs = torch.unique(sampled_vertices_idx)
                    sym_map = torch.tensor(
                        [symmetry_dict.get(int(v), -1) for v in unique_idxs]
                    ).to(n_vert.device)

                    valid_sym = sym_map != -1
                    sym_idxs = sym_map[valid_sym]
                    sampled_vertices_idx = torch.cat([sym_idxs, unique_idxs]).unique()

            sampled_vertices = n_vert[sampled_vertices_idx].detach().clone()
            aux_mesh_vertices, aux_mesh_faces = mesh2spheremesh(sampled_vertices, sphere_verts, sphere_faces, cfg.aux_sphere_radius)

            aux_ms = pymeshlab.MeshSet()
            aux_ms.add_mesh(pymeshlab.Mesh(vertex_matrix=aux_mesh_vertices.detach().cpu().numpy(), face_matrix=aux_mesh_faces.detach().cpu().numpy()))
            if not aux_ms.current_mesh().has_wedge_tex_coord():
                aux_ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=10000)
            aux_ms.save_current_mesh(str(output_path / 'tmp' / 'aux_mesh.obj'), save_vertex_color=False)

            aux_mesh = obj.load_obj(str(output_path / 'tmp' / 'aux_mesh.obj'))
            aux_mesh = mesh.unit_size(aux_mesh)

            print("="*80)
            print('[!] Auxiliary Mesh Initialized')
            print("="*80)

            if cfg.aux_symmetry_loss_weight > 0:
                with torch.no_grad():
                    aux_detected_symmetry = detect_symmetry(mesh_vertices=sampled_vertices)
                    if len(aux_detected_symmetry) == 0:
                        raise ValueError("No symmetry detected in the mesh. Please check the mesh, adjust threshold, or disable symmetry loss.")
                print("="*80)
                print(f"[!] Detected {len(aux_detected_symmetry)} symmetries in the auxiliary mesh")
                print("="*80)

        if use_cage_loss:
            w_base, ref_label2obb = mesh2obbweights(
                vertices=n_vert.detach().clone(), 
                label2vertices=label2vertices, 
                ref_label2obb=ref_label2obb,
                visualize=False
            )
            w_base = w_base.detach().clone()

            label2obbparam = {
                label: OBBParam(init_corner=obb['corners'].detach(), center=obb['center'].detach(), size=obb['size'].detach(), axes=obb['axes'].detach()) \
                    for label, obb in ref_label2obb.items()
            }
            obb_params = torch.nn.ModuleList(label2obbparam.values())
            obb_optimizer = torch.optim.Adam(obb_params.parameters(), lr=cfg.aux_lr)

            interpolated = torch.zeros_like(n_vert)
            for label, v_indices in label2vertices.items():
                obb = label2obbparam[label]
                obb_corners = obb.transform_obb()
                p_hat = w_base[v_indices] @ obb_corners
                interpolated[v_indices] = p_hat
            
            sampled_vertices = interpolated[sampled_vertices_idx]  
            pred_verts, pred_faces = mesh2spheremesh(sampled_vertices, sphere_verts, sphere_faces, cfg.aux_sphere_radius)
            
            aux_m = mesh.Mesh(
                pred_verts,
                pred_faces,
                material=mat,
                base=aux_mesh
            )
            _sz = 1024

            render_aux_mesh = create_scene([aux_m.eval()], sz=_sz)
            render_aux_mesh = mesh.auto_normals(render_aux_mesh)
            render_aux_mesh = mesh.compute_tangents(render_aux_mesh)

            if cfg.adapt_dist and it > 0:
                with torch.no_grad():
                    v_pos = aux_m.v_pos.clone()
                    vmin = v_pos.amin(dim=0)
                    vmax = v_pos.amax(dim=0)
                    v_pos -= (vmin + vmax) / 2
                    mult = torch.cat([v_pos.amin(dim=0), v_pos.amax(dim=0)]).abs().amax().cpu()
                    aux_cams.dataset.dist_min = cfg.dist_min * mult
                    aux_cams.dataset.dist_max = cfg.dist_max * mult

            aux_cams_iterator = iter(aux_cams)
            aux_params_camera = next(aux_cams_iterator)
            for key in aux_params_camera:
                aux_params_camera[key] = aux_params_camera[key].to(device)

            final_aux_mesh = render_aux_mesh.eval(aux_params_camera)

            if cfg.random_light_rotation:
                light_idx = np.random.randint(0, len(lgts))
                lgt = lgts[light_idx]
            aux_train_out_buffer = render.render_mesh(
                glctx,
                final_aux_mesh,
                aux_params_camera['mvp'],
                aux_params_camera['campos'],
                lgt if cfg.random_light_rotation else fixed_lgt,
                [cfg.train_res, cfg.train_res],
                spp=1,
                num_layers=1,
                msaa=False,                
                background=aux_params_camera['bkgs'],
                depth_background=None,
            )
            aux_train_render = aux_train_out_buffer['shaded']
            aux_train_render = nvdiffutil.rgb_to_srgb(aux_train_render)
            aux_train_render = aux_train_render[..., :3]
            aux_train_render = aux_train_render.permute(0, 3, 1, 2)
            aux_train_render = aux_train_render.clamp(0, 1)

            if it % cfg.log_interval_im == 0:
                log_idx = torch.randperm(cfg.batch_size) #[:5]
                s_log = aux_train_render[log_idx, :, :, :]
                s_log = torchvision.utils.make_grid(s_log)
                ndarr = s_log.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                im = Image.fromarray(ndarr)
                im.save(str(output_path / 'aux_mesh_images' / f'epoch_{it}.png'))

                obj.write_obj(
                    str(output_path / 'aux_meshes'),
                    aux_m.eval(),
                    fname=f'mesh_{it}.obj'
                )
    
            obb_optimizer.zero_grad()
            aux_sds_loss = sds_guidance(
                prompt=cfg.text_prompt,
                image=aux_train_render,
                cfg_scale=cfg.cfg_scale
            )

            if cfg.aux_symmetry_loss_weight > 0:
                aux_loss_symmetry = 0.0
                for symmetry in aux_detected_symmetry:
                    aux_loss_symmetry += symmetry_loss(sampled_vertices, symmetry)
            else:
                aux_loss_symmetry = 0.0

            aux_loss = cfg.aux_sds_weight * aux_sds_loss + cfg.aux_symmetry_loss_weight * aux_loss_symmetry
            aux_loss.backward()
            obb_optimizer.step()
            
        m = mesh.Mesh(
            n_vert,
            load_mesh.t_pos_idx,
            material=mat,
            base=load_mesh
        )
        _sz = 1024

        render_mesh = create_scene([m.eval()], sz=_sz)
        if it == 0:
            base_mesh = render_mesh.clone()
            base_mesh = mesh.auto_normals(base_mesh)
            base_mesh = mesh.compute_tangents(base_mesh)
        render_mesh = mesh.auto_normals(render_mesh)
        render_mesh = mesh.compute_tangents(render_mesh)

        if it % cfg.log_interval == 0:
            with torch.no_grad():
                params = get_camera_params(
                    cfg.log_elev,
                    cfg.log_rot,
                    cfg.log_dist,
                    cfg.log_res,
                    cfg.log_fov,
                )
                log_mesh = mesh.unit_size(render_mesh.eval(params))
                if cfg.bkg_color == "white":
                    log_background = torch.ones(1, cfg.log_res, cfg.log_res, 3).to(device)
                elif cfg.bkg_color == "black":
                    log_background = torch.zeros(1, cfg.log_res, cfg.log_res, 3).to(device)
                elif cfg.bkg_color == "gray":
                    log_background = 0.7 * torch.ones(1, cfg.log_res, cfg.log_res, 3).to(device)
                else:
                    raise ValueError

                if cfg.log_vertex_color:
                    log_image = render.render_mesh_with_vertex_color(
                        glctx,
                        log_mesh,
                        params['mvp'],
                        params['campos'],
                        fixed_lgt,
                        [cfg.log_res, cfg.log_res],
                        1,
                        background=log_background,
                        vertex_color=vertex_color
                    )
                    log_image = log_image[..., :3]
                    log_image = log_image.clamp(0, 1)
                else:
                    log_out_buffer = render.render_mesh(
                        glctx,
                        log_mesh,
                        params['mvp'],
                        params['campos'],
                        fixed_lgt,
                        [cfg.log_res, cfg.log_res],
                        1,
                        background=log_background,
                        depth_background=None
                    )
                    log_image = log_out_buffer['shaded']
                    log_image = nvdiffutil.rgb_to_srgb(log_image)
                    log_image = log_image[..., :3]
                    log_image = log_image.clamp(0, 1)

                log_image = video.ready_image(log_image)
                logger.add_mesh('predicted_mesh', vertices=log_mesh.v_pos.unsqueeze(0), faces=log_mesh.t_pos_idx.unsqueeze(0), global_step=it)
        
        if cfg.adapt_dist and it > 0:
            with torch.no_grad():
                v_pos = m.v_pos.clone()
                vmin = v_pos.amin(dim=0)
                vmax = v_pos.amax(dim=0)
                v_pos -= (vmin + vmax) / 2
                mult = torch.cat([v_pos.amin(dim=0), v_pos.amax(dim=0)]).abs().amax().cpu()
                cams.dataset.dist_min = cfg.dist_min * mult
                cams.dataset.dist_max = cfg.dist_max * mult

        cams_iterator = iter(cams)
        params_camera = next(cams_iterator)

        for key in params_camera:
            params_camera[key] = params_camera[key].to(device)
        
        final_mesh = render_mesh.eval(params_camera)

        if cfg.random_light_rotation:
            light_idx = np.random.randint(0, len(lgts))
            lgt = lgts[light_idx]
        train_out_buffer = render.render_mesh(
            glctx,
            final_mesh,
            params_camera['mvp'],
            params_camera['campos'],
            lgt if cfg.random_light_rotation else fixed_lgt,
            [cfg.train_res, cfg.train_res],
            spp=1,
            num_layers=1,
            msaa=False,                
            background=params_camera['bkgs'],
            depth_background=None,
        )
        train_render = train_out_buffer['shaded']
        train_render = nvdiffutil.rgb_to_srgb(train_render)
        train_render = train_render[..., :3] 
        train_render = train_render.permute(0, 3, 1, 2)
        train_render = train_render.clamp(0, 1)

        if it == 0:
            params_camera = next(iter(cams))
            for key in params_camera:
                params_camera[key] = params_camera[key].to(device)

        if it % cfg.log_interval_im == 0:
            log_idx = torch.randperm(cfg.batch_size)
            s_log = train_render[log_idx, :, :, :]
            s_log = torchvision.utils.make_grid(s_log)
            ndarr = s_log.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            im.save(str(output_path / 'mesh_images' / f'epoch_{it}.png'))

            obj.write_obj(
                str(output_path / 'mesh_final'),
                m.eval(),
                fname=f'mesh_{it}.obj'
            )
        
        optimizer.zero_grad()

        sds_loss = sds_guidance(
            prompt=cfg.text_prompt,
            image=train_render,
            cfg_scale=cfg.cfg_scale
        )
        logger.add_scalar('sds_loss', sds_loss, global_step=it)

        if use_cage_loss:
            w = obbparam2obbweights(
                vertices=n_vert, 
                label2vertices=label2vertices, 
                label2obbparam=label2obbparam,
                mesh=m.eval(), 
                visualize=False, 
                output_path=cfg.output_path, 
                it=it
            )
            _cage_loss = (w - w_base)**2
            cage_loss = torch.tensor(0.0, device=n_vert.device)
            for label, vertices_ind in label2vertices.items():
                cage_loss = cage_loss + _cage_loss[vertices_ind].sum() / len(vertices_ind)
            cage_loss = cage_loss / len(label2vertices.keys())
        else:
            cage_loss = 0.0
        
        if cfg.symmetry_loss_weight > 0:
            loss_symmetry = 0.0
            for symmetry in detected_symmetry:
                loss_symmetry += symmetry_loss(n_vert, symmetry)
        else:
            loss_symmetry = 0.0
        
        r_loss = (((gt_jacobians) - torch.eye(3, 3, device=device)) ** 2).mean()
        logger.add_scalar('jacobian_regularization', r_loss, global_step=it)

        total_loss = cfg.sds_weight * sds_loss + cfg.regularize_jacobians_weight * r_loss + cage_loss_weight * cage_loss + cfg.symmetry_loss_weight * loss_symmetry
        logger.add_scalar('total_loss', total_loss, global_step=it)

        total_loss.backward()
        optimizer.step()

        description = f"SDS = {sds_loss * cfg.sds_weight:.4f}, "
        if cfg.regularize_jacobians_weight > 0.0:
            description += f"Reg. = {r_loss * cfg.regularize_jacobians_weight:.4f}, "
        if cage_loss_weight > 0.0:
            description += f"Aux_SDS = {cfg.aux_sds_weight * aux_sds_loss:.4f}, Cage = {cage_loss * cage_loss_weight:.4f}, "
        if cfg.symmetry_loss_weight > 0.0:
            description += f"Sym. = {loss_symmetry * cfg.symmetry_loss_weight:.4f}, "
        if cage_loss_weight > 0.0 and cfg.aux_symmetry_loss_weight > 0.0:
            description += f"Aux_Sym. = {cfg.aux_symmetry_loss_weight * aux_loss_symmetry:.4f}, "
        description += f"Total = {total_loss:.4f}"
        t_loop.set_description(description)
 
    video.close()
    obj.write_obj(
        str(output_path / 'mesh_final'),
        m.eval()
    )
    
    return