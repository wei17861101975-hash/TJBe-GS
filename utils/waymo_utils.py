import os
import numpy as np
import cv2
import torch
import json
import open3d as o3d
import math
from glob import glob
from tqdm import tqdm
from utils.box_utils import bbox_to_corner3d, inbbox_points, get_bound_2d_mask
from utils.general_utils_drivex import matrix_to_quaternion, quaternion_to_matrix_numpy
from plyfile import PlyData, PlyElement
from torchvision.utils import save_image
from collections import defaultdict

waymo_track2label = {"vehicle": 0, "pedestrian": 1, "cyclist": 2, "sign": 3, "misc": -1}

_camera2label = {
    'FRONT': 0,
    'FRONT_LEFT': 1,
    'FRONT_RIGHT': 2,
    'SIDE_LEFT': 3,
    'SIDE_RIGHT': 4,
}

_label2camera = {
    0: 'FRONT',
    1: 'FRONT_LEFT',
    2: 'FRONT_RIGHT',
    3: 'SIDE_LEFT',
    4: 'SIDE_RIGHT',
}
image_heights = [x//2 for x in [1280, 1280, 1280, 886, 886]]
image_widths = [x//2 for x in [1920, 1920, 1920, 1920, 1920]]
image_filename_to_cam = lambda x: int(x.split('.')[0][-1])
image_filename_to_frame = lambda x: int(x.split('.')[0][:6])


# load ego pose and camera calibration(extrinsic and intrinsic)
def load_camera_info(datadir, args=None):
# ‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’修改‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘
    do_center_pose = True
    if args is not None:
        do_center_pose = bool(getattr(args, 'center_ego_pose', True))
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''’‘’‘’‘’‘’‘

    ego_pose_dir = os.path.join(datadir, 'ego_pose')
    extrinsics_dir = os.path.join(datadir, 'extrinsics')
    intrinsics_dir = os.path.join(datadir, 'intrinsics')

    intrinsics = []
    extrinsics = []
    for i in range(5):
        intrinsic = np.loadtxt(os.path.join(intrinsics_dir, f"{i}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics.append(intrinsic)

    for i in range(5):
        cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir, f"{i}.txt"))
        extrinsics.append(cam_to_ego)

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
    if do_center_pose:
        center_point = np.mean(ego_frame_poses[:, :3, 3], axis=0)
        ego_frame_poses[:, :3, 3] -= center_point  # [num_frames, 4, 4]

    ego_cam_poses = [np.array(ego_cam_poses[i]) for i in range(5)]
    ego_cam_poses = np.array(ego_cam_poses)
# ’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’修改‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘
    if do_center_pose:
        ego_cam_poses[:, :, :3, 3] -= center_point  # [5, num_frames, 4, 4]
# ’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘
    return intrinsics, extrinsics, ego_frame_poses, ego_cam_poses


# calculate obj pose in world frame
# box_info: box_center_x box_center_y box_center_z box_heading
def make_obj_pose(ego_pose, box_info):  # 由车辆位姿与3D框信息计算对象在车辆/世界坐标系下的位姿表示 
    # box_info = tracklet[6:10]: [box_center_x box_center_y box_center_z box_heading] 
    tx, ty, tz, heading = box_info  # 从 box_info 解析对象中心位置（tx,ty,tz）与朝向角 heading（弧度，绕Z轴）
    c = math.cos(heading)  # 计算绕Z轴旋转的 cos 分量
    s = math.sin(heading)  # 计算绕Z轴旋转的 sin 分量
    rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])  # 绕Z轴的旋转矩阵 Rz(heading)

    obj_pose_vehicle = np.eye(4)  # 初始化对象在“车辆坐标系”下的齐次变换矩阵
    obj_pose_vehicle[:3, :3] = rotz_matrix  # 设置旋转部分为 Rz(heading)
    obj_pose_vehicle[:3, 3] = np.array([tx, ty, tz])  # 设置平移部分为对象中心（tx, ty, tz）
    obj_pose_world = np.matmul(ego_pose, obj_pose_vehicle)  # 左乘车辆 ego→world，得到对象在“世界坐标系”下的位姿

    obj_rotation_vehicle = torch.from_numpy(obj_pose_vehicle[:3, :3]).float().unsqueeze(0)  # 取车辆系下旋转矩阵并转为张量，形状 [1,3,3]
    obj_quaternion_vehicle = matrix_to_quaternion(obj_rotation_vehicle).squeeze(0).numpy()  # 将旋转矩阵转四元数（w,x,y,z）
    obj_quaternion_vehicle = obj_quaternion_vehicle / np.linalg.norm(obj_quaternion_vehicle)  # 归一化四元数，避免数值偏差
    obj_position_vehicle = obj_pose_vehicle[:3, 3]  # 取车辆系下的平移向量（对象中心）
    obj_pose_vehicle = np.concatenate([obj_position_vehicle, obj_quaternion_vehicle])  # 拼接为 [tx,ty,tz,qw,qx,qy,qz]

    obj_rotation_world = torch.from_numpy(obj_pose_world[:3, :3]).float().unsqueeze(0)  # 取世界系下旋转矩阵并转为张量，形状 [1,3,3]
    obj_quaternion_world = matrix_to_quaternion(obj_rotation_world).squeeze(0).numpy()  # 将旋转矩阵转四元数（w,x,y,z）
    obj_quaternion_world = obj_quaternion_world / np.linalg.norm(obj_quaternion_world)  # 归一化四元数，确保单位四元数
    obj_position_world = obj_pose_world[:3, 3]  # 取世界系下的平移向量（对象世界坐标中心）
    obj_pose_world = np.concatenate([obj_position_world, obj_quaternion_world])  # 拼接为 [Tx,Ty,Tz,Qw,Qx,Qy,Qz]

    return obj_pose_vehicle, obj_pose_world  # 返回车辆系与世界系下的位姿表示（位置+四元数）


def get_obj_pose_tracking(args, datadir, selected_frames, ego_poses, cameras=[0, 1, 2, 3, 4], use_box_world_center=False, 
                          dynamic_std_thresh=0, dynamic_distance_thresh=0, scene_center=None):  # 读取track信息并生成每帧可见对象的车辆/世界位姿
    tracklets_ls = []  # 暂存每条轨迹的字段数组
    objects_info = {}  # 记录每个对象的属性与尺寸等信息

    tracklet_path = os.path.join(datadir, 'track/track_info.txt')  # 轨迹信息文件路径
    tracklet_camera_vis_path = os.path.join(datadir, 'track/track_camera_vis.json')  # 每帧每对象的可见相机列表

    print(f'Loading from : {tracklet_path}')  # 打印正在加载的轨迹文件
    f = open(tracklet_path, 'r')  # 打开轨迹文本文件
    tracklets_str = f.read().splitlines()  # 逐行读取为字符串列表
    tracklets_str = tracklets_str[1:]  # 跳过第一行头信息

    f = open(tracklet_camera_vis_path, 'r')  # 打开可见相机JSON文件
    tracklet_camera_vis = json.load(f)  # 读取为字典：{track_id: {frame_id: [cam_ids]}}

    start_frame, end_frame = selected_frames[0], selected_frames[1]  # 选定帧范围的起止

    image_dir = os.path.join(datadir, 'images')  # 图像目录
    # n_cameras = 5  # 原Waymo多相机数量
    n_cameras = 1  # TUM单相机场景下设为1
    n_images = len(os.listdir(image_dir))  # 图像文件数量
    n_frames = n_images // n_cameras  # 序列帧数
    n_obj_in_frame = np.zeros(n_frames)  # 统计每帧对象数量的数组

    for tracklet in tracklets_str:  # 遍历每条轨迹的字符串行
    # tracklet：[frame_id track_id object_class alpha box_height box_width box_length box_center_x box_center_y box_center_z box_heading speed
        tracklet = tracklet.split()  # 按空格切分为字段列表
        frame_id = int(tracklet[0])  # 该条记录对应的帧编号
        track_id = int(tracklet[1])  # 对象的轨迹ID
        object_class = tracklet[2]  # 对象类别（vehicle/pedestrian/cyclist等）

        if object_class in ['sign', 'misc']:  # 过滤路牌与杂项
            continue  # 不参与对象建模

        cameras_vis_list = tracklet_camera_vis[str(track_id)][str(frame_id)]  # 该对象在此帧可见的相机列表
        join_cameras_list = list(set(cameras) & set(cameras_vis_list))  # 与期望相机集合求交
        if len(join_cameras_list) == 0:  # 若该对象在目标相机集合中不可见
            continue  # 跳过

        if track_id not in objects_info.keys():  # 首次出现该对象时初始化信息
            objects_info[track_id] = dict()  # 创建对象信息字典
            objects_info[track_id]['track_id'] = track_id  # 记录轨迹ID
            objects_info[track_id]['class'] = object_class  # 记录类别名称
            objects_info[track_id]['class_label'] = waymo_track2label[object_class]  # 映射到类别标签ID
            objects_info[track_id]['height'] = float(tracklet[4])  # 从轨迹字段读取高度
            objects_info[track_id]['width'] = float(tracklet[5])  # 从轨迹字段读取宽度
            objects_info[track_id]['length'] = float(tracklet[6])  # 从轨迹字段读取长度
        else:  # 对象已存在时，更新为该对象在多帧中的最大尺寸
            objects_info[track_id]['height'] = max(objects_info[track_id]['height'], float(tracklet[4]))  # 更新高度最大值
            objects_info[track_id]['width'] = max(objects_info[track_id]['width'], float(tracklet[5]))  # 更新宽度最大值
            objects_info[track_id]['length'] = max(objects_info[track_id]['length'], float(tracklet[6]))  # 更新长度最大值

        tr_array = np.concatenate(  # 拼接为统一的轨迹数组格式
            [np.array(tracklet[:2]).astype(np.float64), np.array([type]), np.array(tracklet[4:]).astype(np.float64)]  
            # [frame_id, track_id] + 占位(type) + [height..heading..]
        )
        tracklets_ls.append(tr_array)  # 加入轨迹数组列表
        n_obj_in_frame[frame_id] += 1  # 帧对象计数加一

    tracklets_array = np.array(tracklets_ls)  # 列表转为numpy数组，便于后续遍历
    max_obj_per_frame = int(n_obj_in_frame[start_frame:end_frame + 1].max())  # 选定范围内每帧最大对象数
    num_frames = end_frame - start_frame + 1  # 帧数（闭区间）
    visible_objects_ids = np.ones([num_frames, max_obj_per_frame]) * -1.0  # 初始化对象ID缓存为-1（不可见）
    visible_objects_pose_vehicle = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0  # 初始化车辆系位姿缓存
    visible_objects_pose_world = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0  # 初始化世界系位姿缓存

    # Iterate through the tracklets and process object data  # 遍历轨迹数组，填充每帧可见对象的位姿
    for tracklet in tracklets_array: 
         # tracklet: [frame_id, track_id, height, width, length, tx, ty, tz, heading, speed]  # 统一的轨迹字段格式
        frame_id = int(tracklet[0])  # 当前记录对应的帧号
        track_id = int(tracklet[1])  # 当前对象的轨迹ID
        if start_frame <= frame_id <= end_frame:  # 仅处理选定帧范围内的记录
            ego_pose = ego_poses[frame_id]  # 取该帧的车辆/平台到世界的位姿（4×4齐次矩阵）
            # ------ TUM分支：track_info 的 box_center 已为世界坐标；直接构建 world 位姿并回推 vehicle 位姿 ------
            if use_box_world_center:  # 使用世界坐标中心的 TUM 分支
                tx, ty, tz, heading = tracklet[6:10]  # 从轨迹字段取世界系中心和平移与航向角
                if scene_center is not None:
                    tx -= scene_center[0]
                    ty -= scene_center[1]
                    tz -= scene_center[2]
                c = math.cos(heading)  # 计算绕 Z 轴的旋转 cos 分量
                s = math.sin(heading)  # 计算绕 Z 轴的旋转 sin 分量
                rotz_matrix_world = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])  # 世界系下对象朝向对应的 Rz(heading)
                obj_pose_world_m = np.eye(4)  # 初始化对象的世界系齐次位姿矩阵
                obj_pose_world_m[:3, :3] = rotz_matrix_world  # 填充旋转部分，单位矩阵
                obj_pose_world_m[:3, 3] = np.array([tx, ty, tz])  # 填充世界系平移（box center）

                # --- START OF PROPOSED CHANGE ---
                # 检查是否为BONN数据集，并应用坐标系修正
                # 假设 args 中有一个 'dataset_type' 字段，值为 'bonn'
                # if getattr(args, 'dataset_type', '') == 'bonn':
                #     # 定义从BONN到TUM坐标系的旋转矩阵
                #     # X_tum = X_bonn
                #     # Y_tum = Z_bonn (BONN的深度变为TUM的前进方向)
                #     # Z_tum = -Y_bonn (BONN的负Y方向（头）变为TUM的正Z方向（上）)
                #     # 这对应于绕X轴旋转 -90 度
                #     R_bonn_to_tum = np.array([
                #         [1,  0,  0],
                #         [0,  0,  1],
                #         [0, -1,  0]
                #     ], dtype=np.float64)
                #     # 将修正旋转应用到 obj_pose_world_m 的旋转部分
                #     obj_pose_world_m[:3, :3] = R_bonn_to_tum @ obj_pose_world_m[:3, :3]
                # --- END OF PROPOSED CHANGE ---
                
                # obj_pose_vehicle_m车辆坐标系/相机坐标系的位姿矩阵
                obj_pose_vehicle_m = np.linalg.inv(ego_pose) @ obj_pose_world_m  # 回推到车辆系：object->vehicle = inv(ego->world) @ object->world
                # np.linalg.inv(ego_pose)代表的是世界坐标系到相机坐标系的变换矩阵
                
                obj_rotation_world = torch.from_numpy(obj_pose_world_m[:3, :3]).float().unsqueeze(0)  # 取世界系旋转并转张量
                obj_quaternion_world = matrix_to_quaternion(obj_rotation_world).squeeze(0).numpy()  # 旋转矩阵转四元数
                obj_quaternion_world = obj_quaternion_world / np.linalg.norm(obj_quaternion_world)  # 归一化四元数
                
                # obj_pose_world是物体世界坐标系下的位姿
                obj_pose_world = np.concatenate([np.array([tx, ty, tz]), obj_quaternion_world])  # 拼接世界位姿向量 [t, q]
                
                # obj_pose_vehicle是自车坐标系/相机坐标系的位姿
                obj_rotation_vehicle = torch.from_numpy(obj_pose_vehicle_m[:3, :3]).float().unsqueeze(0)  # 取车辆系旋转并转张量
                obj_quaternion_vehicle = matrix_to_quaternion(obj_rotation_vehicle).squeeze(0).numpy()  # 旋转矩阵转四元数
                obj_quaternion_vehicle = obj_quaternion_vehicle / np.linalg.norm(obj_quaternion_vehicle)  # 归一化四元数
                obj_pose_vehicle = np.concatenate([obj_pose_vehicle_m[:3, 3], obj_quaternion_vehicle])  # 拼接车辆位姿向量 [t, q]
            else:
                # ------ 原逻辑（Waymo）：对象位姿在车辆坐标系，乘以 ego→world 得到世界位姿 ------
                obj_pose_vehicle, obj_pose_world = make_obj_pose(ego_pose, tracklet[6:10])

            frame_idx = frame_id - start_frame  # 将绝对帧号映射到选定范围的相对索引
            obj_column = np.argwhere(visible_objects_ids[frame_idx, :] < 0).min()  # 找到该帧首个空槽位用于存放对象

            visible_objects_ids[frame_idx, obj_column] = track_id  # 记录对象ID到可见ID缓存
            visible_objects_pose_vehicle[frame_idx, obj_column] = obj_pose_vehicle  # 写入车辆坐标系下的位姿向量
            visible_objects_pose_world[frame_idx, obj_column] = obj_pose_world  # 写入世界坐标系下的位姿向量

    # Remove static objects  # 移除静态对象（仅保留运动对象）
    print("Removing static objects")  # 打印静态对象移除提示
    for key in objects_info.copy().keys():  # 遍历对象字典的副本以便安全删除
        all_obj_idx = np.where(visible_objects_ids == key)  # 找到该对象在可见矩阵中的所有位置索引
        if len(all_obj_idx[0]) > 0:  # 若该对象在选定范围内有出现
            obj_world_postions = visible_objects_pose_world[all_obj_idx][:, :3]  # 取其世界位置轨迹（N×3）
            distance = np.linalg.norm(obj_world_postions[0] - obj_world_postions[-1])  # 起止位置欧氏距离
            dynamic = np.any(np.std(obj_world_postions, axis=0) > dynamic_std_thresh) or distance > dynamic_distance_thresh  # 使用可配置阈值判定动态
            if not dynamic:  # 若判定为静态
                visible_objects_ids[all_obj_idx] = -1.  # 清空ID为不可见
                visible_objects_pose_vehicle[all_obj_idx] = -1.  # 清空车辆位姿
                visible_objects_pose_world[all_obj_idx] = -1.  # 清空世界位姿
                objects_info.pop(key)  # 从对象信息字典中移除该对象
        else:  # 该对象在选定范围内未出现
            objects_info.pop(key)  # 直接移除

    # Clip max_num_obj  # 截断每帧的最大对象数量以压缩缓存维度
    mask = visible_objects_ids >= 0  # 可见对象位置掩码
    max_obj_per_frame_new = np.sum(mask, axis=1).max()  # 统计选定范围内每帧的最大可见对象数
    print("Max obj per frame:", max_obj_per_frame_new)  # 输出最大对象数

    if max_obj_per_frame_new == 0:  # 若没有任何动态对象
        print("No moving obj in current sequence, make dummy visible objects")  # 输出提示并创建占位缓存
        visible_objects_ids = np.ones([num_frames, 1]) * -1.0  # 创建1列占位ID缓存
        visible_objects_pose_world = np.ones([num_frames, 1, 7]) * -1.0  # 创建1列占位世界位姿缓存
        visible_objects_pose_vehicle = np.ones([num_frames, 1, 7]) * -1.0  # 创建1列占位车辆位姿缓存
    elif max_obj_per_frame_new < max_obj_per_frame:  # 若新的最大对象数比原缓存维度更小，则压缩缓存维度
        visible_objects_ids_new = np.ones([num_frames, max_obj_per_frame_new]) * -1.0  # 新的ID缓存
        visible_objects_pose_vehicle_new = np.ones([num_frames, max_obj_per_frame_new, 7]) * -1.0  # 新的车辆位姿缓存
        visible_objects_pose_world_new = np.ones([num_frames, max_obj_per_frame_new, 7]) * -1.0  # 新的世界位姿缓存
        for frame_idx in range(num_frames):  # 遍历每帧
            for y in range(max_obj_per_frame):  # 遍历旧缓存的对象槽位
                obj_id = visible_objects_ids[frame_idx, y]  # 取旧槽位中的对象ID
                if obj_id >= 0:  # 若为有效对象
                    obj_column = np.argwhere(visible_objects_ids_new[frame_idx, :] < 0).min()  # 找到新缓存的空槽位
                    visible_objects_ids_new[frame_idx, obj_column] = obj_id  # 写入新ID缓存
                    visible_objects_pose_vehicle_new[frame_idx, obj_column] = visible_objects_pose_vehicle[frame_idx, y]  # 写入对应车辆位姿
                    visible_objects_pose_world_new[frame_idx, obj_column] = visible_objects_pose_world[frame_idx, y]  # 写入对应世界位姿

        visible_objects_ids = visible_objects_ids_new  # 用新缓存替换旧缓存
        visible_objects_pose_vehicle = visible_objects_pose_vehicle_new  # 替换车辆位姿缓存
        visible_objects_pose_world = visible_objects_pose_world_new  # 替换世界位姿缓存

    box_scale = 1  # 3D bbox 缩放因子（默认1）
    print('box scale: ', box_scale)  # 打印缩放因子

    frames = list(range(start_frame, end_frame + 1))  # 构造闭区间帧列表
    frames = np.array(frames).astype(np.int32)  # 转为整型数组便于索引

    # postprocess object_info  # 后处理对象信息（可变形与时间范围）
    for key in objects_info.keys():  # 遍历所有对象
        obj = objects_info[key]  # 取对象信息字典
        if obj['class'] == 'pedestrian':  # 行人视为可变形对象
            obj['deformable'] = True  # 标记可变形
        else:  # 其他类别视为非可变形
            obj['deformable'] = False  # 标记不可变形

        obj['width'] = obj['width'] * box_scale  # 应用宽度缩放
        obj['length'] = obj['length'] * box_scale  # 应用长度缩放

        obj_frame_idx = np.argwhere(visible_objects_ids == key)[:, 0]  # 该对象出现的帧索引列表
        obj_frame_idx = obj_frame_idx.astype(np.int32)  # 转为整型
        obj_frames = frames[obj_frame_idx]  # 映射到绝对帧号
        obj['start_frame'] = np.min(obj_frames)  # 对象开始出现的帧
        obj['end_frame'] = np.max(obj_frames)  # 对象最后出现的帧

        objects_info[key] = obj  # 写回更新后的对象信息

    # [num_frames, max_obj, track_id, x, y, z, qw, qx, qy, qz]  # 组合世界位姿轨迹（含对象ID）
    objects_tracklets_world = np.concatenate(  # 在最后一维拼接对象ID与位姿向量
        [visible_objects_ids[..., None], visible_objects_pose_world], axis=-1  # ID扩展为1维后与位姿拼接
    )

    objects_tracklets_vehicle = np.concatenate(  # 在最后一维拼接对象ID与车辆位姿向量
        [visible_objects_ids[..., None], visible_objects_pose_vehicle], axis=-1  # 形成车辆坐标系下位姿张量
    )

    return objects_tracklets_world, objects_tracklets_vehicle, objects_info  # 返回世界/车辆位姿信息字典


def padding_tracklets(tracklets, frame_timestamps, min_timestamp, max_timestamp):
    # tracklets: [num_frames, max_obj, ....]
    # frame_timestamps: [num_frames]

    # Clone instead of extrapolation
    if min_timestamp < frame_timestamps[0]:
        tracklets_first = tracklets[0]
        frame_timestamps = np.concatenate([[min_timestamp], frame_timestamps])
        tracklets = np.concatenate([tracklets_first[None], tracklets], axis=0)

    if max_timestamp > frame_timestamps[1]:
        tracklets_last = tracklets[-1]
        frame_timestamps = np.concatenate([frame_timestamps, [max_timestamp]])
        tracklets = np.concatenate([tracklets, tracklets_last[None]], axis=0)

    return tracklets, frame_timestamps


def storePly(path, xyz, rgb):
    # set rgb to 0 - 255
    if rgb.max() <= 1. and rgb.min() >= 0:
        rgb = np.clip(rgb * 255, 0., 255.)
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def split_points_pca_lateral(points: np.ndarray, max_lateral: float = 0.3):  # 基于PCA的横向分割的过滤
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
        return points, np.empty((0, 3), dtype=np.float32)
    if points.shape[0] == 0:
        return points, np.empty((0, 3), dtype=np.float32)
    c = points.mean(axis=0)
    X = points - c
    cov = np.cov(X.T)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, np.argmax(vals)]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    proj = (X @ axis[:, None]) * axis[None, :]
    lateral = np.linalg.norm(X - proj, axis=1)
    keep = lateral <= float(max_lateral)
    kept = points[keep]
    removed = points[~keep]
    return kept.astype(np.float32), removed.astype(np.float32)

def build_pointcloud(args, datadir, object_tracklets_vehicle, object_info, selected_frames, ego_frame_poses, camera_list, scene_center=None):  # 从压缩的点云/投影数据生成对象与背景点的PLY
    start_frame, end_frame = selected_frames[0], selected_frames[1]  # 解析帧范围（闭区间起止）

    print('build point cloud')  # 打印阶段提示
    pointcloud_dir = os.path.join(args.model_path, 'input_ply')  # 目标输出目录：每对象/背景的PLY
    os.makedirs(pointcloud_dir, exist_ok=True)  # 确保输出目录存在

    # ------ 新代码：对象点累积模式，默认使用bbox跨帧累积；可选'mask'按动态掩码逐对象筛点 ------
    OBJECT_ACCUM_MODE = getattr(args, 'object_accum_mode', 'bbox')  # 'bbox' | 'mask'
    # ------ 
    # ------ 原代码：无模式区分，先尝试掩码，再退回bbox（可能混入背景） ------
    # （原逻辑已在下方保留为注释以便对照）
    # ------

    # -------
    # 修改：采用分块累积与限容量采样，避免一次性累积全部帧导致内存被杀
    # 全局累积容器（受容量限制）
    CAP_BKGD = 800_000
    INITIAL_NUM_OBJ = 20_000
    bkgd_xyz_global = np.empty((0, 3), dtype=np.float32)
    bkgd_rgb_global = np.empty((0, 3), dtype=np.float32)
    obj_xyz_global = {f'obj_{tid:03d}': np.empty((0, 3), dtype=np.float32) for tid in object_info.keys()}
    obj_rgb_global = {f'obj_{tid:03d}': np.empty((0, 3), dtype=np.float32) for tid in object_info.keys()}
    align_obj_bkgd_coords = bool(getattr(args, 'align_obj_bkgd_coords', False))
    obj_world_xyz_global = {f'obj_{tid:03d}': np.empty((0, 3), dtype=np.float32) for tid in object_info.keys()} if align_obj_bkgd_coords else None
    obj_world_rgb_global = {f'obj_{tid:03d}': np.empty((0, 3), dtype=np.float32) for tid in object_info.keys()} if align_obj_bkgd_coords else None

    # 分块容器，用于周期性下采样与合并
    CHUNK_SIZE = 50
    chunk_bkgd_xyz = []
    chunk_bkgd_rgb = []
    chunk_obj_xyz = {f'obj_{tid:03d}': [] for tid in object_info.keys()}
    chunk_obj_rgb = {f'obj_{tid:03d}': [] for tid in object_info.keys()}
    chunk_obj_world_xyz = {f'obj_{tid:03d}': [] for tid in object_info.keys()} if align_obj_bkgd_coords else None
    chunk_obj_world_rgb = {f'obj_{tid:03d}': [] for tid in object_info.keys()} if align_obj_bkgd_coords else None
    # -------

    print('initialize from lidar pointcloud')  # 打印初始化点云的提示
    pointcloud_path = None  # 将被设置为选中的npz文件路径（pointcloud.npz 或 pointcloud_20.npz）
    prefer_20 = getattr(args, 'use_debug_20frames', False)  # 是否优先使用20帧调试文件
    ordered = [getattr(args, 'pointcloud_path', None)]
    # ordered = ['pointcloud_0-50.npz', 'pointcloud.npz'] if prefer_20 else ['pointcloud_20.npz']
    # ordered = ['pointcloud_241-320.npz', 'pointcloud.npz'] if prefer_20 else ['pointcloud_20.npz']
    # ordered = ['pointcloud_163-234.npz', 'pointcloud.npz'] if prefer_20 else ['pointcloud_20.npz']
    # ordered = ['pointcloud_31-107.npz', 'pointcloud.npz'] if prefer_20 else ['pointcloud_20.npz']
    # ordered = ['pointcloud_119-183.npz', 'pointcloud.npz'] if prefer_20 else ['pointcloud_20.npz']

    # ordered = ['pointcloud_0-76.npz']
    # ordered = ['pointcloud_56-86.npz','pointcloud_20.npz', 'pointcloud.npz'] if prefer_20 else ['pointcloud.npz', 'pointcloud_20.npz']  # 根据配置决定优先顺序
    # ordered = ['pointcloud_382-462.npz','pointcloud_56-86.npz','pointcloud_20.npz', 'pointcloud.npz']
    # ordered = ['pointcloud_20.npz', 'pointcloud.npz'] if prefer_20 else ['pointcloud.npz', 'pointcloud_20.npz']  # 根据配置决定优先顺序
    # ordered = ['pointcloud.npz']
    print(f"ordered: {ordered}")
    for fname in ordered:  # 遍历候选文件，按优先顺序选择存在的npz
        cand = os.path.join(datadir, fname)  # 拼接候选文件完整路径
        if os.path.isfile(cand):  # 如果该文件存在
            # print(f"cand: {cand}")
            pointcloud_path = cand  # 选中该文件作为数据源
            print(f"cand: {cand}")
            break  # 结束查找
    if pointcloud_path is None:  # 两个候选都不存在时抛错
        raise FileNotFoundError(os.path.join(datadir, 'pointcloud.npz'))  # 未找到点云npz则报错
    npz = np.load(pointcloud_path, allow_pickle=True)  # 读取npz内容（包含3D点与投影信息）
    pts3d_dict = None  # 3D点字典初始化（frame -> Nx3）
    for k in ['pointcloud', 'pointcloud_20', 'points3d', 'lidar_points']:  # 依次尝试常见键名
        if k in npz:  # 如果存在该键
            pts3d_dict = npz[k].item()  # 取出字典（frame->Nx3）
            break  # 结束查找
    if pts3d_dict is None:  # 未找到合适的3D点键名则抛错
        raise KeyError(f'No pointcloud dict found in {pointcloud_path}, available keys: {list(npz.keys())}')  # 未找到点云字典则报错
    
    # ------------------ 处理背景点云 ------------------
    # 如果有点云数据，先进行中心化处理（如果提供了 scene_center）
    # 这确保背景点云与中心化后的动态物体位姿对齐
    if scene_center is not None:
        print(f"Applying scene center {scene_center} to background pointcloud")
        for k in pts3d_dict.keys():
            if len(pts3d_dict[k]) > 0:
                pts3d_dict[k][:, :3] -= scene_center
    # ------------------------------------------------

    pts2d_dict = None  # 投影字典初始化（frame -> [..., cam, u, v]）
    for k in ['camera_projection', 'projection', 'uv']:  # 依次尝试常见投影键名
        if k in npz:  # 如果存在该键
            pts2d_dict = npz[k].item()  # 取出字典（frame->[..., cam, u, v]）
            break  # 结束查找
    if pts2d_dict is None:  # 未找到合适的投影键名则抛错
        raise KeyError(f'No camera projection dict found in {pointcloud_path}, available keys: {list(npz.keys())}')  # 未找到投影字典则报错

    frames_iter = []  # 准备遍历的帧列表（源于npz字典的键）
    for k in pts3d_dict.keys():  # 遍历点云字典的键
        if isinstance(k, (int, np.integer)):  # 若为整数
            frames_iter.append(int(k))  # 直接加入帧列表
        else:  # 若为字符串
            s = str(k)  # 转为字符串
            if s.isdigit():  # 若是数字字符串
                frames_iter.append(int(s))  # 转为整数加入
    frames_iter = sorted(frames_iter)  # 对帧号排序，保证时间顺序
    # if prefer_20 and len(frames_iter) > 20:  # 若配置为20帧调试且帧数量超过20
    #     frames_iter = frames_iter[:20]  # 仅取前20帧，缩短处理与验证周期
    for i, frame in tqdm(enumerate(frames_iter)):  # 遍历选定的帧列表，i为序号
        raw_3d = pts3d_dict.get(frame, None)  # 取该帧的3D点（车辆坐标系）
        if raw_3d is None:  # 键类型不一致时尝试字符串键
            raw_3d = pts3d_dict.get(str(frame), None)  # 取该帧的3D点（字符串键）
        raw_2d = pts2d_dict.get(frame, None)  # 取该帧的投影信息
        if raw_2d is None:  # 键类型不一致时尝试字符串键
            raw_2d = pts2d_dict.get(str(frame), None)  # 取该帧的投影信息（字符串键）
        if raw_3d is None or raw_2d is None:  # 若该帧缺失任一数据，跳过
            continue  # 跳过缺数据的帧

        # use the first projection camera
        points_camera_all = raw_2d[..., 0]  # 每点的观测相机ID
        points_projw_all = raw_2d[..., 1]  # 每点的像素列坐标（w）
        points_projh_all = raw_2d[..., 2]  # 每点的像素行坐标（h）

        # each point should be observed by at least one camera in camera lists
        mask = np.array([c in camera_list for c in points_camera_all]).astype(np.bool_)  # 仅保留被指定相机观测到的点

        # get filtered LiDAR pointcloud position and color  # 取出当前视图有效的3D点原始坐标


        # -----------------------------  # 修改分支：支持npz点云已为世界坐标的情况
        points_xyz_raw = raw_3d[mask]  # 原始3D点（来自npz），按相机列表过滤
        ego_pose = ego_frame_poses[frame]  # 该帧的ego->world（world为各帧位姿减去所有帧的均值位姿）位姿，用于坐标系互转
        if getattr(args, 'npz_points_are_world', False):  # 若npz中的点已是世界坐标
            points_xyz_world = np.concatenate([points_xyz_raw, np.ones_like(points_xyz_raw[..., :1])], axis=-1)  # 补齐齐次维，保持world坐标
            world2ego_identity = bool(getattr(args, 'world2ego_identity', False))
            world2ego = np.eye(4, dtype=ego_pose.dtype) if world2ego_identity else np.linalg.inv(ego_pose)
            points_xyz_vehicle = points_xyz_world @ world2ego.T  # 将世界坐标点右乘world->ego转置，得到车辆坐标
        else:  # 默认：npz中的点为车辆/ego坐标
            points_xyz_vehicle = np.concatenate([points_xyz_raw, np.ones_like(points_xyz_raw[..., :1])], axis=-1)  # 补齐齐次维，保持车辆坐标
            points_xyz_world = points_xyz_vehicle @ ego_pose.T  # 将车辆坐标点右乘ego->world转置，得到世界坐标
        # -----------------------------  # 修改分支结束

        points_rgb = np.ones_like(points_xyz_vehicle[:, :3])  # 初始化RGB为1（白色占位）
        points_camera = points_camera_all[mask]  # 过滤后的点对应的相机ID
        points_projw = points_projw_all[mask]  # 过滤后的点对应的像素列
        points_projh = points_projh_all[mask]  # 过滤后的点对应的像素行
         
         # 采集原图像上的颜色信息用于点云
        for cam in camera_list:  # 遍历每个启用相机
            image_filename = os.path.join(args.source_path, "images", f"{frame:06d}_{cam}.png")  # 当前帧该相机的图像路径
            # print("image_filename", image_filename)
            mask_cam = (points_camera == cam)  # 当前相机的点选择掩码
            image = cv2.imread(image_filename)[..., [2, 1, 0]] / 255.  # 读取图像并转为RGB和归一化到[0,1]

            mask_projw = points_projw[mask_cam]  # 当前相机点的像素列
            mask_projh = points_projh[mask_cam]  # 当前相机点的像素行
            mask_rgb = image[mask_projh, mask_projw]  # 从图像采样对应像素颜色
            points_rgb[mask_cam] = mask_rgb  # 写入RGB颜色

        # ------ 新代码：对象点累积（按模式选择） ------
        points_xyz_obj_mask = np.zeros(points_xyz_vehicle.shape[0], dtype=np.bool_)  # 标记被归入对象的点（其余为背景）
        dm_dir = os.path.join(args.source_path, "dynamic_mask")  # 动态掩码根目录
        used_mask = False  # 标记是否使用了掩码
        if OBJECT_ACCUM_MODE == 'mask' and os.path.isdir(dm_dir):  # 仅在掩码模式且目录存在时启用逐对象筛点
            for cam in camera_list:  # 遍历每个相机的掩码
                view_mask = (points_camera == cam)  # 选择该相机下的点
                proj_w = points_projw[view_mask]  # 对应像素列
                proj_h = points_projh[view_mask]  # 对应像素行
                # 每对象掩码：{frame}_{obj}_{cam}.png；若无，则尝试联合掩码 {frame}_{cam}.png
                pattern = os.path.join(dm_dir, f"{frame:06d}_*_{cam}.png")  # 每对象掩码通配符
                mask_files = sorted(glob(pattern))  # 查找该相机下的对象掩码
                if len(mask_files) == 0:  # 若无对象掩码，尝试联合动态掩码
                    union_path = os.path.join(dm_dir, f"{frame:06d}_{cam}.png")  # 联合动态掩码路径
                    if os.path.isfile(union_path):  # 存在则使用
                        mask_files = [union_path]
                for mf in mask_files:  # 遍历找到的掩码文件
                    m = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图（非零视为动态）
                    if m is None:  # 读取失败则跳过
                        continue  # 跳过该掩码
                    # ------ 新代码：按掩码分辨率与图像分辨率对齐缩放投影坐标，避免误匹配导致混入背景 ------
                    image = cv2.imread(os.path.join(args.source_path, "images", f"{frame:06d}_{cam}.png"))  # 读取对应图像用于分辨率对齐
                    img_h, img_w = image.shape[:2] if image is not None else (m.shape[0], m.shape[1])  # 图像分辨率
                    scale_w = m.shape[1] / float(img_w)  # 掩码相对于图像的宽度缩放
                    scale_h = m.shape[0] / float(img_h)  # 掩码相对于图像的高度缩放
                    proj_w_m = np.clip(np.round(proj_w * scale_w).astype(np.int32), 0, m.shape[1] - 1)  # 按掩码分辨率缩放并裁剪列坐标
                    proj_h_m = np.clip(np.round(proj_h * scale_h).astype(np.int32), 0, m.shape[0] - 1)  # 按掩码分辨率缩放并裁剪行坐标
                    valid = (proj_h_m >= 0) & (proj_h_m < m.shape[0]) & (proj_w_m >= 0) & (proj_w_m < m.shape[1])  # 有效像素掩码
                    idxs = np.where(view_mask)[0][valid]  # 当前相机的有效点索引
                    inside = m[proj_h_m[valid], proj_w_m[valid]] > 0  # 掩码命中的布尔向量
                    idxs_inside = idxs[inside]  # 掩码命中的点的全局索引
                    # ------ 
                    # ------ 原代码：未考虑掩码与图像分辨率差异，直接使用原投影坐标
                    # valid = (proj_h >= 0) & (proj_h < m.shape[0]) & (proj_w >= 0) & (proj_w < m.shape[1])
                    # idxs = np.where(view_mask)[0][valid]
                    # inside = m[proj_h[valid], proj_w[valid]] > 0
                    # idxs_inside = idxs[inside]
                    # ------ 
                    if idxs_inside.size == 0:  # 若该掩码没有选中任何点
                        continue  # 跳过该掩码
                    used_mask = True  # 标记为已使用掩码（后续不再回退到bbox）
                    # 解析对象ID：若文件名含“_{obj}_”，则用该ID；否则归并为 obj_000（联合动态）
                    base = os.path.basename(mf)  # 提取文件名
                    parts = base.split('_')  # 按下划线分割
                    obj_id = None  # 默认对象ID为空（联合动态）
                    if len(parts) >= 3 and parts[-1].endswith('.png'):  # 形如 frame_obj_cam.png
                        try:
                            obj_id = int(parts[-2])  # 解析对象ID
                        except:
                            obj_id = None  # 解析失败则视为联合动态
                    key = f"obj_{obj_id:03d}" if obj_id is not None else "obj_000"  # 构造对象键（联合动态为000）
                    if key not in chunk_obj_xyz:  # 若分块容器中尚无该对象
                        chunk_obj_xyz[key] = []  # 初始化对象分块点列表
                        chunk_obj_rgb[key] = []  # 初始化对象分块颜色列表
                    if key not in obj_xyz_global:  # 若全局容器中尚无该对象
                        obj_xyz_global[key] = np.empty((0, 3), dtype=np.float32)  # 初始化对象全局点数组
                        obj_rgb_global[key] = np.empty((0, 3), dtype=np.float32)  # 初始化对象全局颜色数组
                    # ------ 新代码：仅将掩码命中的点加入该对象（严格排除背景）；并在选择后将点转换到对象局部坐标系以避免跨帧姿态变化造成模糊 ------
                    # 根据tracklet查找对应对象的位姿（vehicle->local），将vehicle坐标转为local坐标
                    points_vehicle_inside = points_xyz_vehicle[idxs_inside, :3]  # 取掩码命中的点（车辆系坐标）
                    vehicle2local = None  # 车辆到对象局部坐标变换
                    if obj_id is not None:  # 若该掩码对应具体对象ID
                        frame_idx = int(frame - start_frame)  # 将绝对帧号映射到轨迹矩阵索引
                        if 0 <= frame_idx < len(object_tracklets_vehicle):  # 范围检查
                            for tracklet in object_tracklets_vehicle[frame_idx]:  # 遍历该帧的对象轨迹
                                if int(tracklet[0]) == obj_id:  # 找到对应对象ID
                                    obj_pose_vehicle = np.eye(4)  # 构造对象在车辆系的齐次位姿
                                    obj_pose_vehicle[:3, :3] = quaternion_to_matrix_numpy(tracklet[4:8])  # 填充旋转
                                    obj_pose_vehicle[:3, 3] = tracklet[1:4]  # 填充平移
                                    vehicle2local = np.linalg.inv(obj_pose_vehicle)  # 求局部变换（vehicle->local）
                                    break  # 找到即跳出
                    if vehicle2local is None:  # 若未找到对象位姿（联合动态）
                        # 若未找到对象位姿，回退为联合动态对象的local=vehicle
                        vehicle2local = np.eye(4, dtype=np.float32)  # 局部坐标与车辆系一致
                    points_local = (np.concatenate([points_vehicle_inside, np.ones_like(points_vehicle_inside[:, :1])], axis=1) @ vehicle2local.T)[:, :3]  # 车辆系点转对象局部

                    # ------------------------------------------------------------------------------------
                    # 使用PCA横向分割过滤掉距离主体过远的点（离群点过滤）
                    # 假设 points_local 已经转换到了 Canonical Frame，通常主体应该在原点附近
                    # 我们使用 split_points_pca_lateral 来保留主体部分
                    
                    # 执行过滤
                    kept_local, removed_local = split_points_pca_lateral(points_local, max_lateral=0.5) # 这里的阈值0.8可以根据实际物体大小调整，通常人体/车辆在0.5-1.0米左右

                    # 仅保留 kept_local 对应的索引
                    # 由于 split_points_pca_lateral 返回的是过滤后的点坐标，我们需要反推哪些点被保留了
                    # 这里更高效的方法是直接修改 split_points_pca_lateral 返回 mask，或者在这里重写逻辑
                    # 为保持一致性，我们直接复用上面的逻辑生成 keep_mask
                    
                    if len(points_local) > 0:
                         c = points_local.mean(axis=0)
                         X = points_local - c
                         cov = np.cov(X.T)
                         vals, vecs = np.linalg.eigh(cov)
                         axis = vecs[:, np.argmax(vals)]
                         axis = axis / (np.linalg.norm(axis) + 1e-12)
                         proj = (X @ axis[:, None]) * axis[None, :]
                         lateral = np.linalg.norm(X - proj, axis=1)
                         keep_mask = lateral <= 0.5 # 使用0.8作为横向距离阈值
                         
                         # 应用过滤
                         points_local = points_local[keep_mask]
                         idxs_inside = idxs_inside[keep_mask] # 同步更新原始索引，确保RGB和Mask同步
                         points_rgb_filtered = points_rgb[idxs_inside] # 这一步其实不需要，因为下面是直接用idxs_inside取
                    
                    # ------------------------------------------------------------------------------------

                    chunk_obj_xyz[key].append(points_local.astype(np.float32))  # 分块累积对象点（局部系）
                    chunk_obj_rgb[key].append(points_rgb[idxs_inside].astype(np.float32))  # 分块累积对象颜色
                    points_xyz_obj_mask[idxs_inside] = True  # 标记这些点已归入对象

                    # ------------------------------------------------------------------------------------
                    # 保存每帧动态物体点云用于调试
                    debug_dir = os.path.join(args.source_path, "debug_obj_ply")
                    os.makedirs(debug_dir, exist_ok=True)
                    if len(points_local) > 0:
                        try:
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(points_local.astype(np.float64))
                            pcd.colors = o3d.utility.Vector3dVector(points_rgb[idxs_inside].astype(np.float64))
                            save_path = os.path.join(debug_dir, f"frame{frame:06d}_cam{cam}_{key}_local.ply")
                            o3d.io.write_point_cloud(save_path, pcd)
                        except Exception as e:
                            print(f"Failed to save debug ply: {e}")
                    # ------------------------------------------------------------------------------------
                    if align_obj_bkgd_coords:
                        if key not in chunk_obj_world_xyz:
                            chunk_obj_world_xyz[key] = []
                            chunk_obj_world_rgb[key] = []
                        if key not in obj_world_xyz_global:
                            obj_world_xyz_global[key] = np.empty((0, 3), dtype=np.float32)
                            obj_world_rgb_global[key] = np.empty((0, 3), dtype=np.float32)
                        points_world_inside = points_xyz_world[idxs_inside, :3]
                        chunk_obj_world_xyz[key].append(points_world_inside.astype(np.float32))
                        chunk_obj_world_rgb[key].append(points_rgb[idxs_inside].astype(np.float32))
                    # ------ 
                    # ------ 原代码：未进行坐标对齐与局部转换，可能造成掩码误选或跨帧拖影
                    # valid = (proj_h >= 0) & (proj_h < m.shape[0]) & (proj_w >= 0) & (proj_w < m.shape[1])
                    # ...
                    # chunk_obj_xyz[key].append(points_xyz_world[idxs_inside, :3].astype(np.float32))
                    # ------ 
        if OBJECT_ACCUM_MODE == 'bbox' or not used_mask:  # 无掩码或指定为bbox模式时，使用3D bbox筛点
            # ------ 原代码：无掩码时，使用3D bbox筛点（Waymo原始逻辑），可能混入类别边界附近的背景点 ------
            frame_idx = int(frame - start_frame)  # 绝对帧号映射到轨迹矩阵索引
            if 0 <= frame_idx < len(object_tracklets_vehicle):  # 范围检查
                for tracklet in object_tracklets_vehicle[frame_idx]:  # 遍历当前帧所有对象轨迹
                    track_id = int(tracklet[0])  # 对象ID
                    if track_id >= 0:  # 有效对象
                        obj_pose_vehicle = np.eye(4)  # 构造车辆系对象位姿
                        obj_pose_vehicle[:3, :3] = quaternion_to_matrix_numpy(tracklet[4:8])  # 填充旋转
                        obj_pose_vehicle[:3, 3] = tracklet[1:4]  # 填充平移
                        vehicle2local = np.linalg.inv(obj_pose_vehicle)  # 车辆->局部
                        points_xyz_obj = points_xyz_vehicle @ vehicle2local.T  # 车辆系点转局部系
                        points_xyz_obj = points_xyz_obj[..., :3]  # 去掉齐次维
                        length = object_info[track_id]['length']  # 对象长度
                        width = object_info[track_id]['width']  # 对象宽度
                        height = object_info[track_id]['height']  # 对象高度
                        bbox = [[-length / 2, -width / 2, -height / 2], [length / 2, width / 2, height / 2]]  # 局部系轴对齐包围盒
                        obj_corners_3d_local = bbox_to_corner3d(bbox)  # 生成盒子8角点
                        points_xyz_inbbox = inbbox_points(points_xyz_obj, obj_corners_3d_local)  # 盒内点布尔掩码
                        points_xyz_obj_mask = np.logical_or(points_xyz_obj_mask, points_xyz_inbbox)  # 更新对象点总掩码
                        key = f'obj_{track_id:03d}'  # 对象键
                        if points_xyz_inbbox.any():  # 若盒内有点
                            chunk_obj_xyz[key].append(points_xyz_obj[points_xyz_inbbox].astype(np.float32))  # 分块累积对象点
                            chunk_obj_rgb[key].append(points_rgb[points_xyz_inbbox].astype(np.float32))  # 分块累积对象颜色
                            if align_obj_bkgd_coords:
                                if key not in chunk_obj_world_xyz:
                                    chunk_obj_world_xyz[key] = []
                                    chunk_obj_world_rgb[key] = []
                                if key not in obj_world_xyz_global:
                                    obj_world_xyz_global[key] = np.empty((0, 3), dtype=np.float32)
                                    obj_world_rgb_global[key] = np.empty((0, 3), dtype=np.float32)
                                points_world_inside = points_xyz_world[points_xyz_inbbox, :3]
                                chunk_obj_world_xyz[key].append(points_world_inside.astype(np.float32))
                                chunk_obj_world_rgb[key].append(points_rgb[points_xyz_inbbox].astype(np.float32))
            # ------ 

        # -----------------------------  # 可选：严格背景排除（使用3D bbox进一步剔除掩码漏检的动态点）
     
            
        '''是否启用严格剔除'''   
        strict_bkgd_exclusion = getattr(args, 'strict_bkgd_exclusion', False)  # 是否启用严格剔除
        bkgd_exclude_mask = np.zeros(points_xyz_raw.shape[0], dtype=np.bool_)
        if strict_bkgd_exclusion:
            frame_idx = int(frame - start_frame)
            if 0 <= frame_idx < len(object_tracklets_vehicle):
                for tracklet in object_tracklets_vehicle[frame_idx]:
                    track_id = int(tracklet[0])
                    if track_id < 0:
                        continue
                    # ===================== 原代码（已整体注释）：基于配置的世界/局部盒过滤 =====================
                    # ===================== 新代码：使用用户给定的世界坐标八点，轴对齐盒滤除 =====================
                    corners = np.array([
                        [-1.6, -0.9,  1.4],  # 上部四点
                        [-1.6, -0.9, 0],
                        [ 0.12, -0.9,  1.4],
                        [ 0.12, -0.9, 0],
                        [-1.6, -1.6,  1.4],  # 下部四点
                        [-1.6, -1.6, 0],
                        [ 0.12, -1.6,  1.4],
                        [ 0.12, -1.6, 0],
                    ], dtype=np.float32)
                    mins = corners.min(axis=0)
                    maxs = corners.max(axis=0)
                    pts = points_xyz_world[:, :3]
                    in_axis_aligned_bbox = (pts[:, 0] >= mins[0]) & (pts[:, 0] <= maxs[0]) & \
                                           (pts[:, 1] >= mins[1]) & (pts[:, 1] <= maxs[1]) & \
                                           (pts[:, 2] >= mins[2]) & (pts[:, 2] <= maxs[2])
                    bkgd_exclude_mask = np.logical_or(bkgd_exclude_mask, in_axis_aligned_bbox)
                    # ===================== 结束：新代码 =====================
        # 背景点选择：剔除已归入对象的点，以及（可选）所有落入对象bbox的点
        final_bg_mask = ~(np.logical_or(points_xyz_obj_mask, bkgd_exclude_mask))
        points_lidar_xyz = points_xyz_world[final_bg_mask][..., :3]  # 背景点（世界系）
        points_lidar_rgb = points_rgb[final_bg_mask]  # 背景颜色

        # -------
        # 修改：背景点累积分块，避免全量列表占用内存
        if points_lidar_xyz.shape[0] > 0:  # 若本帧有背景点
            chunk_bkgd_xyz.append(points_lidar_xyz.astype(np.float32))  # 加入背景点分块
            chunk_bkgd_rgb.append(points_lidar_rgb.astype(np.float32))  # 加入背景颜色分块
        # 分块下采样与合并（周期性触发）
        need_flush = ((i + 1) % CHUNK_SIZE == 0) or (i == len(frames_iter) - 1)  # 到达分块周期或最后一帧时触发合并
        if need_flush:  # 执行分块下采样与合并
            # 处理背景点分块
            if len(chunk_bkgd_xyz) > 0:  # 合并背景分块
                bkgd_xyz_chunk = np.concatenate(chunk_bkgd_xyz, axis=0)  # 拼接坐标
                bkgd_rgb_chunk = np.concatenate(chunk_bkgd_rgb, axis=0)  # 拼接颜色
                pcd = o3d.geometry.PointCloud()  # 构建临时点云对象
                pcd.points = o3d.utility.Vector3dVector(bkgd_xyz_chunk)  # 赋值点
                pcd.colors = o3d.utility.Vector3dVector(bkgd_rgb_chunk)  # 赋值颜色
                pcd = pcd.voxel_down_sample(voxel_size=0.15)  # 体素下采样，降低密度
                pcd, _ = pcd.remove_radius_outlier(nb_points=10, radius=0.5)  # 半径离群点移除，去噪
                bkgd_xyz_chunk = np.asarray(pcd.points).astype(np.float32)  # 提取下采样后的点
                bkgd_rgb_chunk = np.asarray(pcd.colors).astype(np.float32)  # 提取下采样后的颜色
                if bkgd_xyz_global.shape[0] == 0:  # 首次赋值
                    bkgd_xyz_global = bkgd_xyz_chunk  # 初始化全局背景点
                    bkgd_rgb_global = bkgd_rgb_chunk  # 初始化全局背景颜色
                else:  # 后续追加
                    bkgd_xyz_global = np.concatenate([bkgd_xyz_global, bkgd_xyz_chunk], axis=0)  # 追加背景点
                    bkgd_rgb_global = np.concatenate([bkgd_rgb_global, bkgd_rgb_chunk], axis=0)  # 追加背景颜色
                if bkgd_xyz_global.shape[0] > CAP_BKGD:  # 超容量时随机下采样至上限
                    idx = np.random.choice(bkgd_xyz_global.shape[0], CAP_BKGD, replace=False)  # 随机索引
                    bkgd_xyz_global = bkgd_xyz_global[idx]  # 下采样背景点
                    bkgd_rgb_global = bkgd_rgb_global[idx]  # 下采样背景颜色
                chunk_bkgd_xyz = []  # 清空分块缓存
                chunk_bkgd_rgb = []  # 清空分块缓存
            # 处理对象点分块
            for key in chunk_obj_xyz.keys():  # 合并对象分块
                if len(chunk_obj_xyz[key]) == 0:  # 该对象无分块则跳过
                    continue  # 跳过
                obj_xyz_chunk = np.concatenate(chunk_obj_xyz[key], axis=0).astype(np.float32)  # 拼接坐标
                obj_rgb_chunk = np.concatenate(chunk_obj_rgb[key], axis=0).astype(np.float32)  # 拼接颜色
                if obj_xyz_global[key].shape[0] == 0:  # 首次赋值
                    obj_xyz_global[key] = obj_xyz_chunk  # 初始化全局对象点
                    obj_rgb_global[key] = obj_rgb_chunk  # 初始化全局对象颜色
                else:  # 后续追加
                    obj_xyz_global[key] = np.concatenate([obj_xyz_global[key], obj_xyz_chunk], axis=0)  # 追加对象点
                    obj_rgb_global[key] = np.concatenate([obj_rgb_global[key], obj_rgb_chunk], axis=0)  # 追加对象颜色
                # if obj_xyz_global[key].shape[0] > INITIAL_NUM_OBJ:  # 超容量时随机下采样至上限
                #     idx = np.random.choice(obj_xyz_global[key].shape[0], INITIAL_NUM_OBJ, replace=False)  # 随机索引
                #     obj_xyz_global[key] = obj_xyz_global[key][idx]  # 下采样对象点
                #     obj_rgb_global[key] = obj_rgb_global[key][idx]  # 下采样对象颜色

                # ------ 新代码：使用体素下采样替代随机下采样，确保时空分布均匀 ------
                # 仅当点数过多时（例如 > 50000）才触发下采样，避免频繁计算
                if obj_xyz_global[key].shape[0] > 50000:
                     pcd_temp = o3d.geometry.PointCloud()
                     pcd_temp.points = o3d.utility.Vector3dVector(obj_xyz_global[key].astype(np.float64))
                     pcd_temp.colors = o3d.utility.Vector3dVector(obj_rgb_global[key].astype(np.float64))
                     # 使用较细的体素网格（例如 0.05m = 5cm）保留细节，同时合并重复点
                     pcd_temp = pcd_temp.voxel_down_sample(voxel_size=0.05)
                     obj_xyz_global[key] = np.asarray(pcd_temp.points).astype(np.float32)
                     obj_rgb_global[key] = np.asarray(pcd_temp.colors).astype(np.float32)
                # ------------------------------------------------------------------
                chunk_obj_xyz[key] = []  # 清空分块缓存
                chunk_obj_rgb[key] = []  # 清空分块缓存
            if align_obj_bkgd_coords:
                for key in list(chunk_obj_world_xyz.keys()):
                    if len(chunk_obj_world_xyz[key]) == 0:
                        continue
                    obj_world_xyz_chunk = np.concatenate(chunk_obj_world_xyz[key], axis=0)
                    obj_world_rgb_chunk = np.concatenate(chunk_obj_world_rgb[key], axis=0)
                    pcd_obj = o3d.geometry.PointCloud()
                    pcd_obj.points = o3d.utility.Vector3dVector(obj_world_xyz_chunk)
                    pcd_obj.colors = o3d.utility.Vector3dVector(obj_world_rgb_chunk)
                    # pcd_obj = pcd_obj.voxel_down_sample(voxel_size=0.15)
                    # pcd_obj, _ = pcd_obj.remove_radius_outlier(nb_points=10, radius=0.5)
                    obj_world_xyz_chunk = np.asarray(pcd_obj.points).astype(np.float32)
                    obj_world_rgb_chunk = np.asarray(pcd_obj.colors).astype(np.float32)
                    if obj_world_xyz_global[key].shape[0] == 0:
                        obj_world_xyz_global[key] = obj_world_xyz_chunk
                        obj_world_rgb_global[key] = obj_world_rgb_chunk
                    else:
                        obj_world_xyz_global[key] = np.concatenate([obj_world_xyz_global[key], obj_world_xyz_chunk], axis=0)
                        obj_world_rgb_global[key] = np.concatenate([obj_world_rgb_global[key], obj_world_rgb_chunk], axis=0)
                    # if obj_world_xyz_global[key].shape[0] > INITIAL_NUM_OBJ:
                    #     idx = np.random.choice(obj_world_xyz_global[key].shape[0], INITIAL_NUM_OBJ, replace=False)
                    #     obj_world_xyz_global[key] = obj_world_xyz_global[key][idx]
                    #     obj_world_rgb_global[key] = obj_world_rgb_global[key][idx]

                    # ------ 新代码：同上，使用体素下采样替代随机下采样 ------
                    if obj_world_xyz_global[key].shape[0] > 50000:
                         pcd_temp = o3d.geometry.PointCloud()
                         pcd_temp.points = o3d.utility.Vector3dVector(obj_world_xyz_global[key].astype(np.float64))
                         pcd_temp.colors = o3d.utility.Vector3dVector(obj_world_rgb_global[key].astype(np.float64))
                         # 世界坐标系下的体素下采样
                         pcd_temp = pcd_temp.voxel_down_sample(voxel_size=0.05)
                         obj_world_xyz_global[key] = np.asarray(pcd_temp.points).astype(np.float32)
                         obj_world_rgb_global[key] = np.asarray(pcd_temp.colors).astype(np.float32)
                    # --------------------------------------------------------
                    chunk_obj_world_xyz[key] = []
                    chunk_obj_world_rgb[key] = []
        # -------

    # -------
    # 修改：使用全局累积后的受限容量结果进行保存
    points_xyz_dict = {'bkgd': bkgd_xyz_global}  # 汇总坐标字典：背景
    points_rgb_dict = {'bkgd': bkgd_rgb_global}  # 汇总颜色字典：背景
    for k in obj_xyz_global.keys():  # 将各对象加入汇总
        points_xyz_dict[k] = obj_xyz_global[k]  # 对象坐标
        points_rgb_dict[k] = obj_rgb_global[k]  # 对象颜色
    # -------

    for k in points_xyz_dict.keys():  # 逐对象/背景写PLY
        points_xyz = points_xyz_dict[k]  # 当前键的坐标数组
        points_rgb = points_rgb_dict[k]  # 当前键的颜色数组
        ply_path = os.path.join(pointcloud_dir, f'points3D_{k}.ply')  # 目标PLY路径
        # ------ 新代码：跳过空对象，避免写空PLY失败；仅保存掩码筛选后的对象点 ------
        if points_xyz is None or points_xyz.shape[0] == 0:  # 空点云跳过
            print(f'skip empty pointcloud for {k}')  # 打印跳过提示
            continue  # 继续下一个键
        try:
            storePly(ply_path, points_xyz, points_rgb)  # 写PLY文件
            print(f'saving pointcloud for {k}, number of initial points is {points_xyz.shape}')  # 打印保存信息
        except:
            print(f'failed to save pointcloud for {k}')  # 写入失败提示
        # ------ 
    # if align_obj_bkgd_coords:
    #     pointcloud_world_dir = os.path.join(args.model_path, 'input_ply')
    #     os.makedirs(pointcloud_world_dir, exist_ok=True)
    #     ply_path_bkgd = os.path.join(pointcloud_world_dir, 'points3D_bkgd.ply')
    #     if bkgd_xyz_global is not None and bkgd_xyz_global.shape[0] > 0:
    #         try:
    #             # Ensure 3 channels for background color
    #             if bkgd_rgb_global.ndim == 2 and bkgd_rgb_global.shape[1] == 1:
    #                 bkgd_rgb_global = np.repeat(bkgd_rgb_global, 3, axis=1)
    #             elif bkgd_rgb_global.ndim == 1:
    #                 bkgd_rgb_global = np.stack([bkgd_rgb_global] * 3, axis=1)
    #             storePly(ply_path_bkgd, bkgd_xyz_global, bkgd_rgb_global)
    #         except:
    #             print('failed to save world pointcloud for bkgd')
    #     for k in obj_world_xyz_global.keys():
    #         points_xyz = obj_world_xyz_global[k]
    #         points_rgb = obj_world_rgb_global[k]
    #         if points_xyz is None or points_xyz.shape[0] == 0:
    #             continue
    #         ply_path_obj = os.path.join(pointcloud_world_dir, f'points3D_{k}.ply')
    #         try:
    #             # Ensure 3 channels for object color to avoid unwanted colormap (green)
    #             if points_rgb.ndim == 2 and points_rgb.shape[1] == 1:
    #                 points_rgb = np.repeat(points_rgb, 3, axis=1)
    #             elif points_rgb.ndim == 1:
    #                 points_rgb = np.stack([points_rgb] * 3, axis=1)
    #             storePly(ply_path_obj, points_xyz, points_rgb)
    #         except:
    #             print(f'failed to save world pointcloud for {k}')

def build_bbox_mask(args, object_tracklets_vehicle, object_info, selected_frames, intrinsics, cam_to_egos, camera_list):  # 根据3D盒与位姿生成每帧/相机的2D掩码
    start_frame, end_frame = selected_frames[0], selected_frames[1]  # 选定的起止帧索引
    save_dir = os.path.join(args.source_path, "dynamic_mask_select")  # 掩码可选保存目录（当前未启用）
    # [frame, cam] -> mask list  # 每个视角存放该帧所有对象的单独掩码列表
    dynamic_mask_info = defaultdict(list)  # 视角索引到对象掩码列表的映射
    bbox_masks = dict()  # 视角索引到融合后的总掩码的映射
    for i, frame in tqdm(enumerate(range(start_frame, end_frame + 1))):  # 遍历帧范围并记录在 object_tracklets_vehicle 中的相对索引 i
        for cam in camera_list:  # 遍历选定的相机视角
            view_idx = frame * 10 + cam  # 视角唯一索引（frame*10+cam）
            h, w = image_heights[cam], image_widths[cam]  # 当前相机的图像高宽
            obj_bound = np.zeros((h, w)).astype(np.uint8)  # 初始化融合掩码（所有对象的逻辑或）
            if i >= len(object_tracklets_vehicle):  # 若该帧在对象轨迹列表中不存在
                continue  # 跳过该帧
            obj_tracklets = object_tracklets_vehicle[i]  # 取该帧的所有对象轨迹（车辆坐标系位姿与尺寸）
            ixt, ext = intrinsics[cam], cam_to_egos[cam]  # 当前相机的内参K与外参cam->ego
            for obj_tracklet in obj_tracklets:  # 遍历该帧的每个对象
                track_id = int(obj_tracklet[0])  # 对象轨迹ID
                if track_id >= 0:  # 有效对象
                    obj_pose_vehicle = np.eye(4)  # 初始化对象在车辆坐标系的齐次位姿
                    obj_pose_vehicle[:3, :3] = quaternion_to_matrix_numpy(obj_tracklet[4:8])  # 由四元数设置旋转
                    obj_pose_vehicle[:3, 3] = obj_tracklet[1:4]  # 由平移向量设置位置
                    obj_length = object_info[track_id]['length']  # 对象长度
                    obj_width = object_info[track_id]['width']  # 对象宽度
                    obj_height = object_info[track_id]['height']  # 对象高度
                    bbox = np.array([[-obj_length, -obj_width, -obj_height],
                                     [obj_length, obj_width, obj_height]]) * 0.5  # 盒子最小/最大点（局部坐标半尺寸）
                    corners_local = bbox_to_corner3d(bbox)  # 将盒子转换为8个角点（局部坐标）
                    corners_local = np.concatenate([corners_local, np.ones_like(corners_local[..., :1])], axis=-1)  # 角点转齐次坐标
                    corners_vehicle = corners_local @ obj_pose_vehicle.T  # 将角点从对象局部变换到车辆坐标系
                    mask = get_bound_2d_mask(  # 将3D角点投影到像平面并生成2D掩码
                        corners_3d=corners_vehicle[..., :3],  # 车辆坐标系下的3D角点
                        K=ixt,  # 相机内参
                        pose=np.linalg.inv(ext),  # ego->cam 外参（cam->ego 的逆）
                        H=h, W=w  # 图像尺寸
                    )
                    obj_bound = np.logical_or(obj_bound, mask)  # 融合当前对象掩码
                    dynamic_mask_info[view_idx].append(mask)  # 记录该视角下的单对象掩码
            # save_image(torch.from_numpy(obj_bound).float(), os.path.join(save_dir, f'{frame:06d}_{cam}.png'))  # 可选：保存融合掩码到磁盘
            bbox_masks[view_idx] = obj_bound  # 写入该视角的融合掩码
    return dynamic_mask_info, bbox_masks  # 返回：每视角的对象掩码列表与融合掩码
