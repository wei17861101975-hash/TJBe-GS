# Description: Load the EmerWaymo dataset for training and testing
# adapted from the PVG datareader for the data from EmerNeRF

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from scene.scene_utils import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, storePly
from utils.graphics_utils import BasicPointCloud, focal2fov
from utils.general_utils import save_ply, load_ply
from utils.waymo_utils import get_obj_pose_tracking, build_pointcloud, build_bbox_mask
from utils.general_utils_drivex import quaternion_to_matrix_numpy
from collections import defaultdict
import cv2
from glob import glob
import re

image_filename_to_cam = lambda x: int(x.split('.')[0][-1])
image_filename_to_frame = lambda x: int(x.split('.')[0][:6])

def _read_index_set(path):
    # -------- 读取 indicies.txt，提取需要保留的帧索引（数字），不存在则返回 None --------
    if not os.path.isfile(path):
        return None
    s = set()
    with open(path, 'r') as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            for m in re.findall(r'\d+', t):
                s.add(m)
    return s
    # --------

def load_camera_info(datadir, args=None):
    # '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    do_center_pose = True
    if args is not None:
        do_center_pose = bool(getattr(args, 'center_ego_pose', True))
    # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # --- 原逻辑：从 Waymo 风格的 ego_pose 目录读取位姿 ---
    ego_pose_dir = os.path.join(datadir, 'ego_pose')
    center_point = np.zeros(3)  # Initialize center_point
    if os.path.isdir(ego_pose_dir):
        ego_frame_poses = []
        ego_cam_poses = [[] for i in range(5)]
        ego_pose_paths = sorted(os.listdir(ego_pose_dir))
        for ego_pose_path in ego_pose_paths:
            # frame pose
            if '_' not in ego_pose_path:
                ego_frame_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
                ego_frame_poses.append(ego_frame_pose)
            else:
                cam = image_filename_to_cam(ego_pose_path)
                ego_cam_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
                ego_cam_poses[cam].append(ego_cam_pose)
        # center ego pose
        ego_frame_poses = np.array(ego_frame_poses)
# ‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’
        
        if do_center_pose:
            center_point = np.mean(ego_frame_poses[:, :3, 3], axis=0)
            
            ego_frame_poses[:, :3, 3] -= center_point  # [num_frames, 4, 4], ego -> world
# ‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’
        ego_cam_poses = [np.array(ego_cam_poses[i]) for i in range(5)]  # ego -> world
        ego_cam_poses = np.array(ego_cam_poses)
# ‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘
        if do_center_pose:
            ego_cam_poses[:, :, :3, 3] -= center_point  # [5, num_frames, 4, 4]
        return ego_frame_poses, ego_cam_poses, center_point
    # --- TUM 兼容逻辑：无 ego_pose 时，使用 groundtruth.txt + extrinsics(单位矩阵) 生成位姿 ---
    # 从 args 获取 camera_list 与 groundtruth_path
    camera_list = getattr(args, 'camera_list', [0])
    image_folder = os.path.join(datadir, "images")
    frame_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png") or f.endswith(".jpg")])
    # 以 camera_list 数量分割帧数
    num_frames = len(frame_files) // max(1, len(camera_list))
    # groundtruth 路径优先使用 args.groundtruth_path，其次尝试 datadir/groundtruth.txt
    gt_path = getattr(args, 'groundtruth_path', None)
    if gt_path is None:
        gt_path = os.path.join(datadir, 'groundtruth.txt')
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"Groundtruth not found: {gt_path}. 请在配置/命令行提供 groundtruth_path。")
    # 读取 TUM groundtruth: 每行 timestamp tx ty tz qx qy qz qw，构造 4x4 变换矩阵(ego->world)
    gt_poses = []
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith('#'):
                continue
            toks = line.split()
            if len(toks) < 8:
                continue
            tx, ty, tz = map(float, toks[1:4])
            qx, qy, qz, qw = map(float, toks[4:8])
            R = quaternion_to_matrix_numpy(np.array([qw, qx, qy, qz]))  # 输入为 [w, x, y, z]
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
            gt_poses.append(T)
    if len(gt_poses) == 0:
        raise RuntimeError("Groundtruth 文件解析失败，未读取到有效位姿。")
    # 截断或扩展以匹配图像帧数
    if len(gt_poses) >= num_frames:
        ego_frame_poses = np.stack(gt_poses[:num_frames], axis=0)
    else:
        # 若 groundtruth 少于帧数，进行重复填充
        reps = int(np.ceil(num_frames / len(gt_poses)))
        ego_frame_poses = np.stack((gt_poses * reps)[:num_frames], axis=0)
    # 中心化位姿
    if do_center_pose:
        center_point = np.mean(ego_frame_poses[:, :3, 3], axis=0)
        ego_frame_poses[:, :3, 3] -= center_point
    # 相机位姿：TUM 单相机，extrinsics 为单位矩阵 => cam 与 ego 重合，所有相机共享 ego 位姿
    ego_cam_poses = []
    for _ in range(max(1, max(camera_list) + 1)):
        ego_cam_poses.append(ego_frame_poses.copy())
    ego_cam_poses = np.array(ego_cam_poses)  # [n_cam, num_frames, 4, 4]
    return ego_frame_poses, ego_cam_poses, center_point
    # ego_frame_poses是每一帧减去所有帧的x,y,z均值后的中心化位姿，ego_cam_poses是相机位姿，
    # 由于TUM数据集只有一个相机，所以ego_frame_poses和ego_cam_poses是相同的


def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def transform_poses_pca(poses, fix_radius=0):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
    
    From https://github.com/SuLvXiangXin/zipnerf-pytorch/blob/af86ea6340b9be6b90ea40f66c0c02484dfc7302/internal/camera_utils.py#L161
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    if fix_radius > 0:
        scale_factor = 1. / fix_radius
    else:
        scale_factor = 1. / (np.max(np.abs(poses_recentered[:, :3, 3])) + 1e-5)
        scale_factor = min(1 / 10, scale_factor)

    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    return poses_recentered, transform, scale_factor


def readDriveXWaymoInfo(args):
    # -------- 修改后：显式读取是否加载天空掩码，TUM数据默认关闭 --------
    load_dynamic_mask = args.load_dynamic_mask
    load_bbox_mask = args.load_bbox_mask
    load_sky_mask = getattr(args, 'load_sky_mask', False)
    # --------
    neg_fov = args.neg_fov  # true
    start_time = args.start_time
    end_time = args.end_time

    # -----------------------------  # 渲染分辨率控制：与原模型输出大小一致或按配置放大
    ORIGINAL_SIZE = [[1280, 1920], [1280, 1920], [1280, 1920], [884, 1920], [884, 1920]]  # 默认原始分辨率（作为后备）
    load_size = [640, 960]  # 旧默认值（可能与数据不一致，将在下方覆盖）
    # -----------------------------

    cam_infos = []
    points = []
    points_time = []

    data_root = args.source_path
    image_folder = os.path.join(data_root, "images")
    # --- 修改：按 camera_list 数量计算帧数，兼容单相机 TUM 数据 ---
    num_seqs = len(os.listdir(image_folder)) / max(1, len(args.camera_list))
    if end_time == -1:
        end_time = int(num_seqs)
    else:
        end_time += 1

    # -------- 修改前：直接使用连续帧范围 start_time..end_time 作为训练帧
    # frame_num = end_time - start_time
    # --------
    # -------- 修改后：读取 indicies.txt，对帧进行筛选，仅保留需要的帧；支持20帧调试文件 indicies_20.txt
    idx_debug = getattr(args, 'use_debug_20frames', False)  # 是否使用20帧调试
    # idx_path = os.path.join(data_root, "indicies_20.txt") if idx_debug else os.path.join(data_root, "indicies.txt")  # 选择索引文件
    # idx_path = os.path.join(data_root, "indicies_56-86.txt") if idx_debug else os.path.join(data_root, "indicies.txt")
    # idx_path = os.path.join(data_root, "indicies.txt")
    # idx_path = os.path.join(data_root, "indicies_0-76.txt")if idx_debug else os.path.join(data_root, "indicies.txt")
    # idx_path = os.path.join(data_root, "indicies_382-462.txt") if idx_debug else os.path.join(data_root, "indicies.txt")
    # idx_path = os.path.join(data_root, "indicies_119-183.txt")if idx_debug else os.path.join(data_root, "indicies.txt")
    # idx_path = os.path.join(data_root, "indicies_31-107.txt")if idx_debug else os.path.join(data_root, "indicies.txt")
    # idx_path = os.path.join(data_root, "indicies_163-234.txt")if idx_debug else os.path.join(data_root, "indicies.txt")
    # idx_path = os.path.join(data_root, "indicies_241-320.txt")if idx_debug else os.path.join(data_root, "indicies.txt")

    # idx_path = os.path.join(data_root, "indicies_0-50.txt")if idx_debug else os.path.join(data_root, "indicies.txt")
    idx_path = getattr(args, 'indicies_path', None)
    # print(f"indicies文件{os.path.exists(idx_path)}存在")
    idx_set = _read_index_set(idx_path)  # 读取索引集合
    # print("idx_set:", idx_set)
    all_frames = list(range(start_time, end_time))
    print("all_frames:", all_frames)
    if idx_set is not None and len(idx_set) > 0:
        idx_int = set(int(x) for x in idx_set if x.isdigit())
        # print("idx_int:", idx_int)
        frames_selected = [f for f in all_frames if (f in idx_int or str(f) in idx_set)]
        print("frames_selected:", frames_selected)
    else:
        frames_selected = all_frames
    # if idx_debug:  # 若为20帧调试模式
    #     frames_selected = [f for f in frames_selected if f >= 0][:21]  # 强制仅保留前20帧
    frame_num = len(frames_selected)
    # print("frames_selected:", frames_selected)
    # --------
    # assert frame_num == 50, "frame_num should be 50"
    time_duration = args.time_duration  # 时间范围 [start_time, end_time)
    time_interval = (time_duration[1] - time_duration[0]) / (end_time - start_time)   # 每个帧的时间间隔

    camera_list = args.camera_list
    # -----------------------------  # 新逻辑：优先读取真实原始分辨率，其次支持目标渲染尺寸/放大因子
    load_size_cfg = getattr(args, 'load_size', None)  # 显式指定渲染尺寸 [H, W]
    target_render_size = getattr(args, 'target_render_size', None)  # 目标渲染尺寸 [H, W]
    render_scale = getattr(args, 'render_scale', None)  # 渲染放大因子（整数/浮点），基于原图尺寸

    # 读取所选相机的首帧图像尺寸，更新 ORIGINAL_SIZE
    if len(frames_selected) > 0 and len(camera_list) > 0:
        sample_frame = frames_selected[0]
        for cam_id in camera_list:
            sample_path = os.path.join(args.source_path, "images", f"{sample_frame:06d}_{cam_id}.png")
            if os.path.isfile(sample_path):
                im = Image.open(sample_path)
                W, H = im.size
                ORIGINAL_SIZE[cam_id] = [H, W]  # 以真实图像尺寸覆盖默认值

    # 计算最终渲染尺寸：优先级为 load_size_cfg > target_render_size > render_scale > 原图尺寸
    if load_size_cfg is not None:
        load_size = list(load_size_cfg)
    elif target_render_size is not None:
        load_size = [int(target_render_size[0]), int(target_render_size[1])]
    elif render_scale is not None:
        base_h, base_w = ORIGINAL_SIZE[camera_list[0]]
        load_size = [int(round(base_h * float(render_scale))), int(round(base_w * float(render_scale)))]
    else:
        # 默认使用原始图像尺寸（例如 TUM 的 480×640），避免渲染过小
        load_size = ORIGINAL_SIZE[camera_list[0]]
    # -----------------------------
    # truncated_min_range, truncated_max_range = 2, 80  # 截断范围，单位：米
    truncated_min_range, truncated_max_range = 0, 80  # 截断范围，单位：米
    # ---------------------------------------------
    # load poses: intrinsic, c2w, l2w per camera
    # ---------------------------------------------
    _intrinsics = []
    _distortions = []
    cam_to_egos = []
    for i in camera_list:
        # load intrinsics
        intrinsic = np.loadtxt(os.path.join(data_root, "intrinsics", f"{i}.txt"))  # 加载相机内参文件
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]  
        k1, k2, p1, p2, k3 = intrinsic[4], intrinsic[5], intrinsic[6], intrinsic[7], intrinsic[8]
        # scale intrinsics w.r.t. load size
        fx, fy = (  # 缩放内参，保持与加载尺寸比例一致
            fx * load_size[1] / ORIGINAL_SIZE[i][1],
            fy * load_size[0] / ORIGINAL_SIZE[i][0],
        )
        cx, cy = (
            cx * load_size[1] / ORIGINAL_SIZE[i][1],
            cy * load_size[0] / ORIGINAL_SIZE[i][0],
        )
        _intrinsics.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))  # 构建相机内参矩阵
        _distortions.append(np.array([k1, k2, p1, p2, k3]))  # 相机畸变参数
        # load extrinsics
        cam_to_ego = np.loadtxt(os.path.join(data_root, "extrinsics", f"{i}.txt"))  # 加载相机到ego的外参矩阵
        cam_to_egos.append(cam_to_ego)  # opencv_cam -> waymo_cam -> waymo_ego  # 相机到ego的变换矩阵 这里加载的是单位矩阵

    # --- 修改：传入 args 以在无 ego_pose 时使用 groundtruth 生成位姿 ---
    ego_frame_poses, ego_cam_poses, center_point = load_camera_info(data_root, args) 
    print("ego_frame_poses:", ego_frame_poses.shape) 
    # ego_frame_poses是从groundtruth.txt从导入真实外参计算的每一帧位姿，减去所有帧的x,y,z均值后的中心化位姿，ego_cam_poses是相机位姿，
    # 由于TUM数据集只有一个相机，所以ego_frame_poses和ego_cam_poses是相同的

    '''
    cam_to_egos: 从外参文件extrinsics加载的相机到ego的变换矩阵，是单位矩阵，用于将相机坐标转换到ego坐标

    ego_frame_poses: 从groundtruth.txt从导入真实外参（tx ty yz ...）计算的每一帧位姿，减去所有帧的x,y,z均值后的中心化位姿
    ego_cam_poses: 相机位姿，由于TUM数据集只有一个相机，所以ego_frame_poses和ego_cam_poses是相同的

    ego_to_world_start： ego_frame_poses的第一帧的位姿，作为参考原点，用于将ego坐标转换到世界坐标

    cam_to_worlds: 相机到世界的变换矩阵，是将ego_cam_poses第一帧为参考原点，归一化当前帧ego位姿，乘以单位矩阵
    ego_to_worlds: 以ego_to_world_start为参考原点，归一化当前帧ego位姿，当前帧的车体位姿相对于参考原点的世界坐标
    、
    lidar_to_worlds：等于ego_to_worlds

    '''

    # ---------------------------------------------
    # get c2w and w2c transformation per frame and camera 
    # ---------------------------------------------
    
    # compute per-image poses and intrinsics  # 计算每张图像的位姿与相机内参
    cam_to_worlds, ego_to_worlds = [], []  # 初始化：存储每图像的相机到世界、ego到世界的变换矩阵
    lidar_to_worlds = []  # 初始化：存储每帧LiDAR到世界的变换（Waymo设定下LiDAR与ego同位姿）

    # -------- 修改前：以 start_time 帧为参考原点，遍历连续帧  # 旧逻辑：直接用起始时间帧作为参考
    # ego_to_world_start = ego_frame_poses[start_time]  # 旧：参考原点为start_time对应ego位姿
    # for t in range(start_time, end_time):  # 旧：遍历连续帧范围
    # --------
    # -------- 修改后：以筛选后的第一帧为参考原点，遍历筛选帧列表  # 新逻辑：更稳健地使用筛选后的首帧作为参考
    print("ego_frame_poses[frames_selected[0]]:", len(ego_frame_poses))
    print("frames_selected[0]:", frames_selected[0])
    # print("ego_frame_poses[frames_selected[0]]:", len(ego_frame_poses))
    ego_to_world_start = ego_frame_poses[frames_selected[0]]  # 参考原点：筛选列表中的第一帧ego位姿
    # 用 inv(首帧) 进行一次归一化，这是为了把世界坐标系平移到参考帧并保持尺度稳定，属于常见的居中处理，并非第二次“求位姿
    for t in frames_selected:  # 遍历经indices.txt筛选后的帧索引
        # ego to world transformation: cur_ego -> world -> start_ego(world)  # 当前ego到参考世界坐标的变换
        ego_to_world = np.linalg.inv(ego_to_world_start) @ ego_frame_poses[t]  # 以参考原点归一化当前帧ego位姿，当前帧的车体位姿相对于参考原点的世界坐标
        ego_to_worlds.append(ego_to_world)  # 记录该帧的ego到世界变换
        for cam_id in camera_list:  # 遍历本帧启用的相机列表
            # transformation:  # 变换链说明
            # opencv_cam -> waymo_cam -> waymo_cur_ego -> world -> start_ego(world)  # 相机到参考世界的完整链路
            cam_ego_to_world = np.linalg.inv(ego_to_world_start) @ ego_cam_poses[cam_id, t]  # 当前相机ego位姿相对参考原点（第一帧位姿）的变换
            cam2world = cam_ego_to_world @ cam_to_egos[cam_id]  # 叠加相机外参（cam到ego），得到相机到世界，cam_to_egos是单位矩阵
            cam_to_worlds.append(cam2world)  # 记录该相机的c2w变换
        # lidar to world : lidar = ego in waymo  # Waymo约定LiDAR与ego一致
        lidar_to_worlds.append(ego_to_world)  # same as ego_to_worlds  # 记录LiDAR到世界（复用ego_to_world）
    # convert to numpy arrays  # 将累积的列表转换为数组以便后续索引与广播
    cam_to_worlds = np.stack(cam_to_worlds, axis=0)  # [num_images, 4, 4] 相机到世界矩阵  cam_to_worlds是相机到世界的变换矩阵，每个元素是一个4x4的变换矩阵
    lidar_to_worlds = np.stack(lidar_to_worlds, axis=0)  # [num_frames, 4, 4] LiDAR到世界矩阵

    # ---------------------------------------------
    # get image, sky_mask, lidar per frame and camera 
    # ---------------------------------------------
    # 选择可用的点云npz（优先使用20帧调试文件）
    pointcloud_path = None  # 点云与投影数据的压缩包路径
    fnames = [getattr(args, 'pointcloud_path', None)]
    # for fname in ['pointcloud_382-462.npz','pointcloud_56-86.npz', 'pointcloud_20.npz', 'pointcloud.npz']:
    # for fname in ['pointcloud_56-86.npz', 'pointcloud_20.npz', 'pointcloud.npz']:  # 遍历候选npz文件名（调试/完整）
    # for fname in ['pointcloud_20.npz', 'pointcloud.npz']:  # 遍历候选npz文件名（调试/完整）
    # for fname in ['pointcloud.npz','pointcloud_56-86.npz', 'pointcloud_20.npz']: 
    # for fname in ['pointcloud_0-76.npz', 'pointcloud.npz']:
    # for fname in ['pointcloud_119-183.npz', 'pointcloud.npz']:
    # for fname in ['pointcloud_31-107.npz', 'pointcloud.npz']:
    # for fname in ['pointcloud_163-234.npz', 'pointcloud.npz']:
    # for fname in ['pointcloud_241-320.npz', 'pointcloud.npz']:
    # for fname in ['pointcloud_0-50.npz', 'pointcloud.npz']:
    for fname in fnames:
        cand = os.path.join(data_root, fname)  # 拼接候选文件完整路径
        if os.path.isfile(cand):  # 若该文件存在
            print(f"npz文件存在,{cand}")
            pointcloud_path = cand  # 选定该npz路径
            break  # 找到后终止搜索
    if pointcloud_path is None:  # 两个候选均不存在则报错
        raise FileNotFoundError(os.path.join(data_root, 'pointcloud.npz'))
    npz_pc = np.load(pointcloud_path, allow_pickle=True)  # 加载npz文件为dict（允许对象类型）
    pts3d_dict = None  # 加载每帧LiDAR 3D点（dict：frame->Nx3）
    for k in ['pointcloud', 'pointcloud_20', 'points3d', 'lidar_points']:
        if k in npz_pc:
            pts3d_dict = npz_pc[k].item()
            break
    if pts3d_dict is None:
        raise KeyError(f'No pointcloud dict found in {pointcloud_path}, available keys: {list(npz_pc.keys())}')

    # ---------------------------------------------
    # Apply scene center to lidar points if needed (only for world-space points)
    # ---------------------------------------------
    npz_points_are_world = bool(getattr(args, 'npz_points_are_world', False))
    if center_point is not None and npz_points_are_world:
         print(f"Applying scene center {center_point} to loaded lidar points")
         for k in pts3d_dict.keys():
             if isinstance(pts3d_dict[k], np.ndarray) and pts3d_dict[k].shape[1] >= 3:
                 pts3d_dict[k][:, :3] -= center_point

    # pts2d_dict = np.load(pointcloud_path, allow_pickle=True)['camera_projection'].item()  # 可选：每点的相机投影信息（未用）

    # object_tracklets_vehicle.shape = [num_frames, num_objects, 8], 8: id(1) + position(3) + quaternion(4)
    # len(object_info) = num_objects, track_id, class, class_label(refer to waymo_utils.py), height, width, length, deformable, start_frame, end_frame(closed interval, contain start_frame and end_frame)
    # 解析track_info，使用点云npz中的帧键范围（若为20帧调试文件，将严格对齐到这20帧）
    frames_from_npz = []  # 从npz键提取帧范围（适配整数或字符串键）
    for k in pts3d_dict.keys():
        if isinstance(k, (int, np.integer)):
            frames_from_npz.append(int(k))
        else:
            s = str(k)
            if s.isdigit():
                frames_from_npz.append(int(s))
    if len(frames_from_npz) > 0:  # 若npz包含帧键，按其范围对齐
        selected_range = [min(frames_from_npz), max(frames_from_npz)]  # 以npz最小/最大帧为区间
    else:  # 否则回退到已有的frames_selected
        selected_range = [frames_selected[0], frames_selected[-1]]  # 使用传入的起止帧
        # object_tracklets_vehicle指的是各帧的自车位姿
    _, object_tracklets_vehicle, object_info = get_obj_pose_tracking(args,
        data_root,
        selected_range,
        ego_frame_poses,
        camera_list,
        use_box_world_center=getattr(args, 'use_box_world_center', False),
        dynamic_std_thresh=getattr(args, 'dynamic_std_thresh', 0.5),
        dynamic_distance_thresh=getattr(args, 'dynamic_distance_thresh', 2.0),
        scene_center=center_point,
    )
    ply_path = os.path.join(args.model_path, "input_ply")  # 生成对象/背景PLY的输出目录

    build_pointcloud(args, data_root, object_tracklets_vehicle, object_info,  # 构建对象/背景PLY
                         selected_range, ego_frame_poses, camera_list, scene_center=center_point)  # 传入帧范围与位姿等

    os.makedirs(os.path.join(args.source_path, "dynamic_mask_select"), exist_ok=True)  # 准备保存二维bbox融合掩码的目录
    dynamic_bbox_info, bbox_mask_list = build_bbox_mask(args, object_tracklets_vehicle, object_info,
                        selected_range, _intrinsics, cam_to_egos, camera_list)

    os.makedirs(os.path.join(args.source_path, "refined_bbox"), exist_ok=True)  # 准备保存掩码细化结果的目录
    # -------- 修改前：遍历连续帧
    # for idx, t in enumerate(tqdm(range(start_time, end_time), desc="Loading data", bar_format='{l_bar}{bar:50}{r_bar}')):
    # --------
    # -------- 修改后：遍历经 indicies.txt 筛选后的帧列表
    for idx, t in enumerate(tqdm(frames_selected, desc="Loading data", bar_format='{l_bar}{bar:50}{r_bar}')):  # 遍历经 indicies.txt 筛选后的帧列表
    # --------
        images = []
        image_paths = []
        HWs = []
        sky_masks = []
        dynamic_masks = []
        bbox_masks = []
        # print("idx:",idx)

        for cam_idx in camera_list:  # 遍历启用相机
            image_path = os.path.join(args.source_path, "images", f"{t:06d}_{cam_idx}.png")  # 当前帧相机图像路径
            im_data = Image.open(image_path)  # 打开图像
            im_data = im_data.resize((load_size[1], load_size[0]), Image.BILINEAR)  # PIL resize: (W, H)
            W, H = im_data.size  # 获取尺寸（W,H）
            image = np.array(im_data) / 255.  # 归一化为[0,1]浮点
            HWs.append((H, W))  # 记录分辨率
            images.append(image)  # 记录图像
            image_paths.append(image_path)  # 记录路径

            # -------- 修改前：无条件读取天空掩码文件，TUM数据无此文件会报错
            # sky_path = os.path.join(args.source_path, "sky_mask", f"{t:03d}_{cam_idx}.png")
            # sky_data = Image.open(sky_path)
            # sky_data = sky_data.resize((load_size[1], load_size[0]), Image.NEAREST)  # PIL resize: (W, H)
            # sky_mask = np.array(sky_data) > 0
            # sky_masks.append(sky_mask.astype(np.float32))
            # --------
            # -------- 修改后：按配置加载天空掩码，缺失或未启用时生成全零掩码
            if load_sky_mask:  # 按配置加载天空掩码
                sky_path = os.path.join(args.source_path, "sky_mask", f"{t:03d}_{cam_idx}.png")
                if os.path.isfile(sky_path):
                    sky_data = Image.open(sky_path)  # 读取掩码
                    sky_data = sky_data.resize((load_size[1], load_size[0]), Image.NEAREST)  # PIL resize: (W, H)
                    sky_mask = (np.array(sky_data) > 0).astype(np.float32)  # 二值化为0/1
                else:
                    sky_mask = np.zeros((load_size[0], load_size[1]), dtype=np.float32)  # 缺失则给全零
            else:
                sky_mask = np.zeros((load_size[0], load_size[1]), dtype=np.float32)  # 未启用则全零
            sky_masks.append(sky_mask)  # 记录当前相机的天空掩码
            # --------

            if load_bbox_mask:  # 按配置加载bbox融合掩码
                view_idx = t * 10 + cam_idx  # 视图索引（帧*10+相机）
                bbox_mask = bbox_mask_list[view_idx]  # 取该视图的融合掩码
                bbox_masks.append(bbox_mask.astype(np.float32))  # 记录掩码

            if load_dynamic_mask:  # 按配置加载/生成动态掩码
                # Try merging multiple dynamic_mask images: {frame}_{obj}_{cam}.png
                dm_dir = os.path.join(args.source_path, "dynamic_mask")
                pattern = os.path.join(dm_dir, f"{t:06d}_*_{cam_idx}.png")
                mask_files = sorted(glob(pattern))  # 查找该视图下的对象掩码列表
                if len(mask_files) > 0:
                    cur_view_mask = None  # 当前视图的融合掩码初始化
                    for mf in mask_files:
                        m = Image.open(mf)  # 读取对象掩码
                        m = m.resize((load_size[1], load_size[0]), Image.NEAREST)  # 对齐分辨率
                        m_np = (np.array(m) > 0)  # 二值化
                        cur_view_mask = m_np if cur_view_mask is None else np.logical_or(cur_view_mask, m_np)  # 融合多对象掩码
                    dynamic_masks.append(cur_view_mask.astype(np.float32))  # 记录融合后的掩码
                else:
                    # fallback to bbox+seg logic
                    view_idx = t * 10 + cam_idx  # 视图索引
                    dynamic_bbox_list = dynamic_bbox_info[view_idx]  # 该视图的动态bbox列表
                    seg_path = os.path.join(args.source_path, "seg_npy", f"{t:06d}_{cam_idx}.npy")  # 语义分割npy路径
                    seg_data = np.load(seg_path)  # 读取分割结果
                    seg_data = cv2.resize(seg_data, (load_size[1], load_size[0]), interpolation=cv2.INTER_NEAREST)  # 对齐分辨率
                    max_seg_value = seg_data.max()  # 最大类别ID
                    bbox_dict = defaultdict(list)  # bbox->候选分割掩码列表
                    for seg_value in range(1, max_seg_value + 1):
                        seg_mask = (seg_data == seg_value)  # 当前类别的像素掩码
                        if seg_mask.sum() == 0:  # 空类别跳过
                            continue
                        ideal_bbox = -1  # 最优匹配的bbox索引
                        max_overlap = 0  # 最大重叠像素计数
                        for bbox_idx, dynamic_bbox in enumerate(dynamic_bbox_list):
                            overlap = dynamic_bbox[seg_mask].sum()  # bbox内与该类别重叠像素
                            if overlap > 0.5 * seg_mask.sum() and overlap > max_overlap:  # 过半重叠且更优
                                max_overlap = overlap  # 更新最大重叠
                                ideal_bbox = bbox_idx  # 记录该bbox为最佳匹配
                        if ideal_bbox != -1:  # 找到最佳bbox
                            bbox_dict[ideal_bbox].append(seg_mask)  # 将该类别掩码归入该bbox候选
                    cur_view_mask = np.zeros_like(seg_data)  # 当前视图融合掩码初始化
                    for bbox_idx, seg_masks in bbox_dict.items():  # 遍历有候选的bbox
                        cur_bbox_mask = max(seg_masks, key=lambda x: dynamic_bbox_list[bbox_idx][x].sum() / np.logical_or(dynamic_bbox_list[bbox_idx], x).sum())  # 选择与bbox IoU最高的分割掩码
                        cur_view_mask = np.logical_or(cur_view_mask, cur_bbox_mask)  # 融合该bbox的最佳分割掩码
                    for bbox_idx in range(len(dynamic_bbox_list)):
                        if bbox_idx not in bbox_dict:  # 对于没有分割匹配的bbox
                            cur_view_mask = np.logical_or(cur_view_mask, dynamic_bbox_list[bbox_idx])  # 直接并入该bbox掩码
                    refined_bbox_path = os.path.join(args.source_path, "refined_bbox", f"{t:06d}_{cam_idx}.png")  # 保存融合结果路径
                    Image.fromarray(cur_view_mask.astype(np.uint8) * 255).save(refined_bbox_path)  # 写出融合掩码
                    dynamic_masks.append(cur_view_mask.astype(np.float32))  # 记录融合掩码

        timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * idx / (frame_num - 1)

        # ---- 新代码：健壮读取点云，兼容整数和字符串键；若缺失直接跳过当前帧点云，避免报错 ----
        lidar_points = pts3d_dict.get(t, None)  # 按整数键读取LiDAR点
        if lidar_points is None:
            lidar_points = pts3d_dict.get(str(t), None)  # 键类型不一致时按字符串键读取
        if lidar_points is None:
            continue  # 两种键均缺失则跳过该帧
        # ----
        # ---- 原代码：直接按整数键读取点云，若缺失会报错
        # lidar_points = pts3d_dict[t]
        # ----
        # select lidar points based on a truncated ego-forward-directional range
        # make sure most of lidar points are within the range of the camera 确保大部分点云在相机前方范围
        mode = getattr(args, 'truncation_mode', 'ego_x')
        npz_points_are_world = bool(getattr(args, 'npz_points_are_world', False))
        if npz_points_are_world:
            w_to_norm = np.linalg.inv(ego_to_world_start)
            lidar_points_h = np.concatenate([lidar_points, np.ones_like(lidar_points[..., :1])], axis=-1)
            lidar_points_world_norm = (lidar_points_h @ w_to_norm.T)[:, :3]
            w2ego = np.linalg.inv(lidar_to_worlds[idx])
            lidar_points_ego = (w2ego[:3, :3] @ lidar_points_world_norm.T + w2ego[:3, 3:4]).T
        else:
            lidar_points_world_norm = None
            lidar_points_ego = lidar_points

        if mode == 'ego_x':
            pts_ego = lidar_points_ego

            # 适配TUM：可选按“相机z轴前向”进行投影截断，避免坐标系差异造成的误删
            proj_trunc_mode = getattr(args, 'projection_truncation', 'none')  # 可选：'cam_z' 或 'none'
            proj_z_min = float(getattr(args, 'projection_z_min', 0.0))
            proj_z_max = float(getattr(args, 'projection_z_max', 1e9))
            if proj_trunc_mode == 'cam_z':
                # z = point_camera[:, 2]
                # keep = (z > proj_z_min) & (z < proj_z_max)
                # point_camera = point_camera[keep]
                valid_mask = (pts_ego[:, 2] > proj_z_min) & (pts_ego[:, 2] < proj_z_max)
            else:
                valid_mask = (pts_ego[:, 0] > truncated_min_range) & (pts_ego[:, 0] < truncated_max_range)

        elif mode == 'norm':
            r = np.linalg.norm(lidar_points_ego, axis=1)
            valid_mask = (r > truncated_min_range) & (r < truncated_max_range)
        elif mode == 'none':
            valid_mask = np.ones(lidar_points_ego.shape[0], dtype=bool)
        else:
            valid_mask = (lidar_points_ego[:, 0] > truncated_min_range) & (lidar_points_ego[:, 0] < truncated_max_range)

        if npz_points_are_world:
            lidar_points = lidar_points_world_norm[valid_mask]
        else:
            lidar_points = lidar_points_ego[valid_mask]
        # transform lidar points to world coordinate system
        if not npz_points_are_world:
            lidar_points = (  # 将LiDAR点从车辆/ego系转换到世界系
                    lidar_to_worlds[idx][:3, :3] @ lidar_points.T
                    + lidar_to_worlds[idx][:3, 3:4]
            ).T  # point_xyz_world

        # ---- 新代码：按帧均匀配额采样，降低内存峰值，最终总点数≈num_pts ----
        point_time = np.full_like(lidar_points[:, :1], timestamp)  # 为每点赋当前时间戳
        quota = max(1, int(args.num_pts / max(1, frame_num)))  # 每帧采样配额（均匀分配）
        if lidar_points.shape[0] > quota:
            choice = np.random.choice(lidar_points.shape[0], quota, replace=False)  # 随机采样配额数量
            sampled = lidar_points[choice]  # 采样点
            sampled_time = point_time[choice]  # 对应时间
        else:
            sampled = lidar_points  # 点数不足配额则全取
            sampled_time = point_time  # 时间全取
        
        points.append(sampled)  # 累积采样点
        
        points_time.append(sampled_time)  # 累积时间
        # ----
        # ---- 原代码：直接累积每帧全部点，后续一次性拼接导致内存峰值
        # points.append(lidar_points)
        # point_time = np.full_like(lidar_points[:, :1], timestamp)
        # points_time.append(point_time)
        # ----
        # print("camera_list:",len(camera_list))
        # print("bbox_masks:",len(bbox_masks))

        for cam_idx in camera_list:  # 遍历每个相机视图
            # world-lidar-pts --> camera-pts : w2c
            c2w = cam_to_worlds[int(len(camera_list)) * idx + cam_idx]  # 当前视图的相机->世界
            w2c = np.linalg.inv(c2w)  # 求世界->相机
            point_camera = (
                    w2c[:3, :3] @ lidar_points.T
                    + w2c[:3, 3:4]
            ).T
            # ---- 适配TUM：可选按“相机z轴前向”进行投影截断，避免坐标系差异造成的误删 ----

            # ----------------------------------------------------------------------
            # print("%"*20)
            # print("idx:",idx)
            # print("point_camera.shape:",point_camera.shape)
            # if idx == 1:
            #     print("point_camera:",point_camera)  # 打印第18帧相机点云

            R = np.transpose(w2c[:3, :3])  # 旋转矩阵按CUDA glm约定存转置
            T = w2c[:3, 3]  # 平移向量
            K = _intrinsics[cam_idx]  # 相机内参矩阵
            fx = float(K[0, 0])  # 焦距x
            fy = float(K[1, 1])  # 焦距y
            cx = float(K[0, 2])  # 光心x
            cy = float(K[1, 2])  # 光心y
            height, width = HWs[cam_idx]  # 图像高宽
            if neg_fov:
                FovY = -1.0  # 使用负FOV占位
                FovX = -1.0  # 使用负FOV占位
            else:
                FovY = focal2fov(fy, height)  # 由焦距计算垂直视场角
                FovX = focal2fov(fx, width)  # 由焦距计算水平视场角
            
            
            # print("FovY, FovX:", FovY, FovX)
            # print("K:", K)
            if args.undistort:
                # ---- 新代码：仅在掩码存在且启用时进行去畸变，否则置为 None/全零（sky），避免越界 ----
                image = cv2.undistort(np.array(images[cam_idx]), _intrinsics[cam_idx], _distortions[cam_idx])
                sky_mask = cv2.undistort(np.array(sky_masks[cam_idx]), _intrinsics[cam_idx], _distortions[cam_idx])
                if load_dynamic_mask and cam_idx < len(dynamic_masks):
                    dynamic_mask = cv2.undistort(np.array(dynamic_masks[cam_idx]), _intrinsics[cam_idx], _distortions[cam_idx])
                else:
                    dynamic_mask = None
                if load_bbox_mask and cam_idx < len(bbox_masks):
                    bbox_mask = cv2.undistort(np.array(bbox_masks[cam_idx]), _intrinsics[cam_idx], _distortions[cam_idx])
                else:
                    bbox_mask = None
                # ----
                # ---- 原代码：无条件对所有掩码进行去畸变，列表为空时会索引越界
                # image = cv2.undistort(np.array(images[cam_idx]), _intrinsics[cam_idx], _distortions[cam_idx])
                # sky_mask = cv2.undistort(np.array(sky_masks[cam_idx]), _intrinsics[cam_idx], _distortions[cam_idx])
                # dynamic_mask = cv2.undistort(np.array(dynamic_masks[cam_idx]), _intrinsics[cam_idx], _distortions[cam_idx])
                # bbox_mask = cv2.undistort(np.array(bbox_masks[cam_idx]), _intrinsics[cam_idx], _distortions[cam_idx])
                # ----
            else:
                image = images[cam_idx]
                sky_mask = sky_masks[cam_idx]
                # ---- 新代码：安全索引，缺失时置为 None，避免列表为空越界 ----
                dynamic_mask = dynamic_masks[cam_idx] if (load_dynamic_mask and cam_idx < len(dynamic_masks)) else None
                bbox_mask = bbox_masks[cam_idx] if (load_bbox_mask and cam_idx < len(bbox_masks)) else None
                # ----
                # ---- 原代码：按标志直接索引列表，列表为空时会越界
                # dynamic_mask = dynamic_masks[cam_idx] if load_dynamic_mask else None
                # bbox_mask = bbox_masks[cam_idx] if load_bbox_mask else None
                # ----

            cam_infos.append(CameraInfo(uid=idx * 10 + cam_idx, R=R, T=T, FovY=FovY, FovX=FovX,  # 记录当前视图的相机信息
                                        image=image,
                                        image_path=image_paths[cam_idx], image_name=f"{t:03d}_{cam_idx}",
                                        width=width, height=height, timestamp=timestamp,
                                        pointcloud_camera=point_camera,
                                        fx=fx, fy=fy, cx=cx, cy=cy,
                                        sky_mask=sky_mask,
                                        dynamic_mask=dynamic_mask,
                                        bbox_mask=bbox_mask))  # 含掩码与投影点
            
    pointcloud = np.concatenate(points, axis=0)  # 拼接所有帧的采样点
    pointcloud_timestamp = np.concatenate(points_time, axis=0)  # 拼接时间戳
    indices = np.random.choice(pointcloud.shape[0], args.num_pts, replace=True)  # 最终随机采样到指定总点数
    pointcloud = pointcloud[indices]  # 采样点云
    pointcloud_timestamp = pointcloud_timestamp[indices]  # 采样时间

    w2cs = np.zeros((len(cam_infos), 4, 4))  # 构造世界->相机矩阵批
    Rs = np.stack([c.R for c in cam_infos], axis=0)  # 收集旋转
    Ts = np.stack([c.T for c in cam_infos], axis=0)  # 收集平移
    w2cs[:, :3, :3] = Rs.transpose((0, 2, 1))  # 恢复旋转为未转置形式
    w2cs[:, :3, 3] = Ts  # 写入平移
    w2cs[:, 3, 3] = 1  # 齐次最后一行
    c2ws = unpad_poses(np.linalg.inv(w2cs))  # 求相机->世界并去掉齐次
    c2ws, transform_pca, scale_factor = transform_poses_pca(c2ws, fix_radius=args.fix_radius)  # PCA对齐并归一化尺度
    # if args.static_thresh > 0: # for PVG separation
    #     args.static_thresh = float(args.static_thresh * scale_factor)

    c2ws = pad_poses(c2ws)  # 重新加齐次维
    for idx, cam_info in enumerate(tqdm(cam_infos, desc="Transform data", bar_format='{l_bar}{bar:50}{r_bar}')):  # 遍历相机，应用归一化
        c2w = c2ws[idx]  # 当前相机的c2w
        w2c = np.linalg.inv(c2w)  # 求w2c
        cam_info.R[:] = np.transpose(w2c[:3, :3])  # 写回旋转（转置存储约定）
        cam_info.T[:] = w2c[:3, 3]  # 写回平移
        cam_info.pointcloud_camera[:] *= scale_factor  # 相机空间点云按尺度缩放
    pointcloud = (np.pad(pointcloud, ((0, 0), (0, 1)), constant_values=1) @ transform_pca.T)[:, :3]  # 点云应用PCA变换并去齐次
    if args.eval:  # 评估/划分训练测试
        # for dynamic scene
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num + 1) % args.testhold != 0]  # 训练集
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num + 1) % args.testhold == 0]  # 测试集

        # for emernerf comparison [testhold::testhold]
        if args.testhold == 10:  # EmerNeRF对比的特定划分策略
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num) % args.testhold != 0 or (idx // args.cam_num) == 0]  # 训练
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num) % args.testhold == 0 and (idx // args.cam_num) > 0]  # 测试
    else:
        train_cam_infos = cam_infos  # 全部用于训练
        test_cam_infos = []  # 测试为空

    nerf_normalization = getNerfppNorm(train_cam_infos)  # 计算NeRFpp标准化参数
    nerf_normalization['radius'] = 1 / nerf_normalization['radius']  # 半径取倒，统一缩放

    pcd = None

    # stage1: read point3d.ply, and initialize as static gaussians
    ply_path = os.path.join(args.model_path, "points3d.ply")  # 静态初始化PLY路径
    if not os.path.exists(ply_path):  # 若不存在则初始化写出
        rgbs = np.random.random((pointcloud.shape[0], 3))  # 随机颜色占位
        storePly(ply_path, pointcloud, rgbs, pointcloud_timestamp)  # 写出PLY
    try:
        pcd = fetchPly(ply_path)  # 读取PLY为BasicPointCloud
    except:
        pcd = BasicPointCloud(pointcloud, colors=np.zeros([pointcloud.shape[0], 3]), normals=None, time=pointcloud_timestamp)  # 读取失败则构造

    obj2world_dict = dict()  # 对象轨迹的object->world齐次矩阵列表
    obj_timestamp_list = dict()  # 对象轨迹对应的时间戳列表
    for track_id in object_info.keys():
        obj2world_dict[f'obj_{track_id:03d}'] = []  # 初始化轨迹列表
        obj_timestamp_list[f'obj_{track_id:03d}'] = []  # 初始化时间戳列表

    for i, frame in tqdm(enumerate(range(start_time, end_time))):  # 遍历时间范围（含起始，end_time为开区间）
        # if args.eval:
        #     if (i + 1) % args.testhold == 0:
        #         continue

        ego_pose = ego_frame_poses[frame]  # 当前帧 ego->world 
        # ego_frame_poses是从groundtruth.txt从导入真实外参计算的每一帧位姿，减去所有帧的x,y,z均值后的中心化位姿
        for tracklet in object_tracklets_vehicle[i]:  # 遍历该帧的对象轨迹
            track_id = int(tracklet[0])  # 轨迹对象ID
            if track_id >= 0:  # 有效对象
                obj_pose_vehicle = np.eye(4)  # 构造object->ego位姿
                obj_pose_vehicle[:3, :3] = quaternion_to_matrix_numpy(tracklet[4:8])  # 旋转
                obj_pose_vehicle[:3, 3] = tracklet[1:4]  # 平移
                obj_pose_world = ego_pose @ obj_pose_vehicle  # object->world
                obj2world_dict[f'obj_{track_id:03d}'].append(obj_pose_world)  # 记录位姿轨迹矩阵
                timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * i / (frame_num - 1)  # 插值当前时间戳
                obj_timestamp_list[f'obj_{track_id:03d}'].append(timestamp)  # 记录时间戳

    ply_dict = dict()  # 汇总PLY数据的字典
    ply_dict['bkgd'] = {'xyz_array': None, 'colors_array': None, 'start_frame': start_time, 'end_frame': end_time - 1}  # 背景条目
    for k, v in object_info.items():
        ply_dict[f'obj_{k:03d}'] = {'xyz_offset': None, 'trajectory': None, 'colors_array': None, 'start_frame': v['start_frame'], 'end_frame': v['end_frame'], 'timestamp_list': obj_timestamp_list[f'obj_{k:03d}']}  # 每对象条目

    for idx, item in enumerate(sorted(os.listdir(os.path.join(args.model_path, "input_ply")))):  # 遍历输入PLY目录
        # 0 is background
        if idx == 0:
            xyz, colors = load_ply(os.path.join(args.model_path, "input_ply", item))  # 读取背景PLY
            xyz = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1)  # 加齐次维
            transform = transform_pca @ np.linalg.inv(ego_to_world_start)  # 背景：世界->起始ego，再做PCA
            xyz = xyz @ transform.T  # 应用变换
            ply_dict['bkgd']['xyz_array'] = xyz[:, :3]  # 保存背景坐标
            ply_dict['bkgd']['colors_array'] = colors  # 保存背景颜色
            path11 = '/home/weiwei/work_content/BezierGS/eval_output/tum/smoke_test/input_ply/background.ply'  # 调试输出路径
            # storePly(path11, xyz[:, :3], colors)  # 可选：写出背景PLY以检查
            continue  # 背景处理完跳过后续对象流程

        xyz, colors = load_ply(os.path.join(args.model_path, "input_ply", item))  # 读取当前对象PLY
        obj_idx = int(item.split('_')[2].split('.')[0])  # 从文件名解析对象ID
        obj_key = f'obj_{obj_idx:03d}'  # 构造键
        if (obj_key not in obj2world_dict) or (obj_key not in obj_timestamp_list) or (obj_key not in ply_dict):
            print(f'skip object {obj_key} without tracklets or timestamps')
            continue

        cur_obj_offset_list = []
        cur_obj_trajectory = []
        xyz = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1)  # 加齐次维
        # print("-"*20, ply_dict)  # 调试：打印当前ply_dict结构
        ply_dict[obj_key]['colors_array'] = colors  # 记录对象颜色数组
        # ---- 新代码：对时间维度进行下采样（最多约50个时间点），显著降低内存占用 ----
        ts_list_full = obj_timestamp_list[obj_key]  # 对象完整时间戳列表
        ts_count = len(ts_list_full)  # 时间点总数
        t_step = max(1, ts_count // 50)  # 下采样步长（最多≈50个时间点）
        ts_indices = list(range(0, ts_count, t_step))  # 选择的时间索引
        # ******************  # 生成用于可视化对齐检查的对象世界坐标点（与bkgd一致的规范化世界坐标）
        first_xyz_world = None  # 记录第一个时间点的世界坐标，用于生成测试PLY
        # ******************
        for j, obj2world in enumerate([obj2world_dict[obj_key][k] for k in ts_indices]):  # 遍历下采样后的轨迹点
            # object -> cur ego -> world -> start ego(world)
            transform = np.linalg.inv(ego_to_world_start) @ obj2world  # world->起始ego（对齐参考）
            transform = transform_pca @ transform  # 再应用PCA对齐
            xyz_world = xyz @ transform.T  # 对象点应用轨迹变换
            xyz_world = xyz_world[:, :3]  # 去齐次维
            # ******************  # 仅取第一个时间点的世界坐标，用于输出 obj_XX_test.ply（与 bkgd 坐标系一致）
            if first_xyz_world is None:
                first_xyz_world = xyz_world.copy()
            # ******************
            # xyz offset(from trajectory)
            trajectory_pos = transform[:3, 3]  # 当前轨迹的世界平移
            xyz_offset = xyz_world - trajectory_pos  # 计算相对偏移
            cur_obj_offset_list.append(xyz_offset)
            cur_obj_trajectory.append(trajectory_pos)
        if len(cur_obj_offset_list) == 0 or len(cur_obj_trajectory) == 0:
            print(f'skip object {obj_key} due to empty trajectory/offset')
            continue
        ply_dict[obj_key]['xyz_offset'] = np.stack(cur_obj_offset_list, axis=1)  # [N, T, 3] 对象点相对偏移序列
        ply_dict[obj_key]['trajectory'] = np.stack(cur_obj_trajectory, axis=0)  # [T, 3] 轨迹位置序列
        ply_dict[obj_key]['timestamp_list'] = [ts_list_full[k] for k in ts_indices]  # 下采样后的时间戳列表
        # ******************  # 输出规范化世界坐标系下的对象测试点云，以便与真实图像对齐检查
        # 说明：obj_XX_test.ply 使用与 bkgd 相同的规范化世界坐标（起始ego对齐 + PCA），这里选取第一个时间点
        # 如需全时间点合并以检查动态拖影，可改为拼接 first_xyz_world 与更多时间点 xyz_world
        # if first_xyz_world is not None and first_xyz_world.shape[0] > 0:
        #     test_ply_path = os.path.join(args.model_path, "input_ply", f"points3D_{obj_key}_test.ply")
        #     try:
        #         storePly(test_ply_path, first_xyz_world, colors)
        #     except:
        #         print(f'failed to save test object ply for {obj_key}')
        # ******************
        # ----
        # ---- 原代码：使用全时序（T≈700）构建偏移与轨迹，内存占用过高易被系统杀死
        # for obj2world in obj2world_dict[f'obj_{obj_idx:03d}']:
        #     transform = np.linalg.inv(ego_to_world_start) @ obj2world
        #     transform = transform_pca @ transform
        #     xyz_world = xyz @ transform.T
        #     xyz_world = xyz_world[:, :3]
        #     trajectory_pos = transform[:3, 3]
        #     xyz_offset = xyz_world - trajectory_pos
        #     cur_obj_offset_list.append(xyz_offset)
        #     cur_obj_trajectory.append(trajectory_pos)
        # ply_dict[f'obj_{obj_idx:03d}']['xyz_offset'] = np.stack(cur_obj_offset_list, axis=1) # [N, T, 3]
        # ply_dict[f'obj_{obj_idx:03d}']['trajectory'] = np.stack(cur_obj_trajectory, axis=0) # [T, 3]
        # ply_dict[f'obj_{obj_idx:03d}']['timestamp_list'] = obj_timestamp_list[f'obj_{obj_idx:03d}']
        # ----

    scene_info = SceneInfo(point_cloud=pcd,  # 基础点云（静态初始化）
                           train_cameras=train_cam_infos,  # 训练相机集合
                           test_cameras=test_cam_infos,  # 测试相机集合
                           nerf_normalization=nerf_normalization,  # NeRFpp归一化参数
                           ply_path=ply_path,  # 静态PLY路径
                           ply_dict=ply_dict,  # 对象/背景PLY与轨迹数据
                           time_interval=time_interval,  # 每帧时间间隔
                           time_duration=time_duration,  # 序列起止时间
                           scale_factor=scale_factor,
                           frame_num=frame_num)  # 全局尺度因子

    return scene_info

# trans (4, 4)
# xyz: (4, N) 
# trans @ xyz = (4, N)

# xyz.T @ trans.T = (trans @ xyz).T = (N, 4) @ (4, 4) = (N, 4)

# start_ego: x front, y right, z down
# opencv: x right, y down, z front
# start_ego->opencv: y->x, x->z, z->y
# save_ply(np.concatenate([ply_82_pca[:,1:2], ply_82_pca[:, 2:3], ply_82_pca[:, 0:1]], axis=1), '082.ply')
