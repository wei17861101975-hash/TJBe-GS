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

import torch
import cv2
from scene.cameras import Camera
import numpy as np
from scene.scene_utils import CameraInfo
from tqdm import tqdm
from .graphics_utils import fov2focal

'''
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    sky_mask: np.array = None
    timestamp: float = 0.0
    FovY: float = None
    FovX: float = None
    fx: float = None
    fy: float = None
    cx: float = None
    cy: float = None
    pointcloud_camera: np.array = None
    dynamic_mask: np.array = None
    bbox_mask: np.array = None
'''

def loadCam(args, id, cam_info: CameraInfo, resolution_scale):  # 加载并构建单个相机对象，支持分辨率缩放与深度数据准备
    orig_w, orig_h = cam_info.width, cam_info.height  # 取原始图像宽高（来自相机信息）
    # print("="*20)  # 打印分隔线，便于调试阅读
    # print("id:",id)  # 打印当前相机的内部ID
    # print("cam_info:",cam_info.pointcloud_camera.shape)  # 打印该相机关联的点云（相机坐标系）形状

    if args.resolution in [1, 2, 3, 4, 8, 16, 32]:  # 若显式设置了常用的整数下采样因子
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(  # 计算最终训练分辨率的宽
            orig_h / (resolution_scale * args.resolution)  # 计算最终训练分辨率的高
        )
        scale = resolution_scale * args.resolution  # 记录总缩放倍数（多级缩放的乘积）
    else:  # 其它情况（例如指定目标宽度等），转为浮点缩放处理
        if args.resolution == -1:  # -1 表示不额外下采样
            global_down = 1  # 下采样因子为1（等比）
        else:  # 按目标宽度反算缩放比例
            global_down = orig_w / args.resolution  # 以原始宽度除以目标宽度得到缩放因子

        scale = float(global_down) * float(resolution_scale)  # 综合最终缩放倍数
        resolution = (int(orig_w / scale), int(orig_h / scale))  # 计算缩放后的训练分辨率 (W, H)

    if cam_info.cx:  # 若提供了显式相机内参（主点与焦距）
        cx = cam_info.cx / scale  # 主点cx按缩放倍数等比缩放
        cy = cam_info.cy / scale  # 主点cy按缩放倍数等比缩放
        fy = cam_info.fy / scale  # 焦距fy按缩放倍数等比缩放
        fx = cam_info.fx / scale  # 焦距fx按缩放倍数等比缩放
    else:  # 否则视为未提供显式内参
        cx = None  # 主点cx为空
        cy = None  # 主点cy为空
        fy = None  # 焦距fy为空
        fx = None  # 焦距fx为空
    
    if cam_info.image.shape[:2] != resolution[::-1]:  # 若原图尺寸与训练分辨率不一致
        image_rgb = cv2.resize(cam_info.image, resolution)  # 按训练分辨率缩放图像 (W, H)
    else:  # 尺寸一致则直接使用原图
        image_rgb = cam_info.image  # 保持原图
    image_rgb = torch.from_numpy(image_rgb).float().permute(2, 0, 1)  # 转为 [C,H,W] 的 float 张量
    gt_image = image_rgb[:3, ...]  # 取前三个通道作为RGB图（忽略可能存在的alpha）

    if cam_info.sky_mask is not None:  # 若有天空掩码
        # if cam_info.sky_mask.shape[:2] != resolution[::-1]:  # 尺寸不一致则缩放
        #     sky_mask = cv2.resize(cam_info.sky_mask, resolution)  # 缩放天空掩码到训练分辨率
        # else:  # 尺寸一致直接使用
        #     sky_mask = cam_info.sky_mask  # 保持原掩码
        # if len(sky_mask.shape) == 2:  # 单通道掩码则扩展为三维
        #     sky_mask = sky_mask[..., None]  # 在末维增加一维通道
        # sky_mask = torch.from_numpy(sky_mask).float().permute(2, 0, 1)  # 转为 [C,H,W] 的张量
        sky_mask = None
    else:  # 无天空掩码
        sky_mask = None  # 置空

    if cam_info.dynamic_mask is not None:  # 若有动态区域掩码
        if cam_info.dynamic_mask.shape[:2] != resolution[::-1]:  # 若尺寸不一致
            dynamic_mask = cv2.resize(cam_info.dynamic_mask, resolution)  # 缩放动态掩码
        else:  # 尺寸一致
            dynamic_mask = cam_info.dynamic_mask  # 保持原掩码
        if len(dynamic_mask.shape) == 2:  # 单通道时扩展维度
            dynamic_mask = dynamic_mask[..., None]  # 扩展为三维
        dynamic_mask = torch.from_numpy(dynamic_mask).float().permute(2, 0, 1)  # 转为 [C,H,W] 张量
    else:  # 无动态掩码
        dynamic_mask = None  # 置空

    if cam_info.bbox_mask is not None:  # 若有包围盒掩码
        if cam_info.bbox_mask.shape[:2] != resolution[::-1]:  # 尺寸不一致则缩放
            bbox_mask = cv2.resize(cam_info.bbox_mask, resolution)  # 缩放包围盒掩码
        else:  # 尺寸一致
            bbox_mask = cam_info.bbox_mask  # 保持原掩码
        if len(bbox_mask.shape) == 2:  # 单通道时扩展维度
            bbox_mask = bbox_mask[..., None]  # 扩展为三维
        bbox_mask = torch.from_numpy(bbox_mask).float().permute(2, 0, 1)  # 转为 [C,H,W] 张量
    else:  # 无包围盒掩码
        bbox_mask = None  # 置空

    use_gt_depth = getattr(args, "use_gt_depth", False)  # 是否使用数据集提供的真值深度
    # print("use_gt_depth:", use_gt_depth)  # 可选调试输出
    # print("cam_info.image_path:", cam_info.image_path)
    if use_gt_depth:  # 若启用真值深度（例如 TUM）
        import os  # 引入OS库以处理路径
        depth_path = None  # 初始化深度图路径
        img_path = cam_info.image_path  # 当前相机的图像路径
        if img_path and ("images" in img_path):  # 仅当路径包含 images 目录时推断深度路径
            root = img_path[:img_path.rfind("images")]  # images 上层目录作为根
            basename = os.path.basename(img_path)  # 提取文件名
            cand = os.path.join(root, "depth", basename)  # 组合对应的 depth 路径
            if os.path.isfile(cand):  # 若深度文件存在
                depth_path = cand  # 使用该路径
        if depth_path is not None:  # 找到了深度文件
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # 原始读取深度图（保留类型）
            # print("depth_raw:", depth_raw.shape, depth_raw[295,495])
            if depth_raw is not None:  # 读取成功
                h, w = gt_image.shape[1:]  # 取训练分辨率的高和宽（来自图像张量）
                if depth_raw.dtype == np.uint16:  # 16位整型需按比例转换为米或标准单位
                    scale = getattr(args, "tum_depth_scale", 1.0/5000.0)  # TUM 深度标定缩放（默认 1/5000）
                    depth = (depth_raw.astype(np.float32) * float(scale))  # 转为浮点并线性缩放
                    # print("depth:", depth.shape, depth[295,495])
                else:  # 其它类型直接转浮点
                    depth = depth_raw.astype(np.float32)  # 转为浮点深度
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)  # 按训练分辨率最近邻缩放
                pts_depth = torch.from_numpy(depth[None]).float()  # 转为 [1,H,W] 的深度张量
                # print("pts_depth:", pts_depth.shape, pts_depth[0,295,495])
            else:  # 读取失败
                pts_depth = None  # 置空
        else:  # 找不到深度文件
            pts_depth = None  # 置空
    # if (not use_gt_depth) and cam_info.pointcloud_camera is not None:  # 若不使用真值深度，且存在相机坐标点云
    #     h, w = gt_image.shape[1:]  # 使用训练分辨率的高和宽
    #     K = np.eye(3)  # 初始化 3×3 相机内参矩阵
    #     if (cam_info.cx is not None) and (cam_info.cy is not None) and (fx is not None) and (fy is not None):  # 优先使用显式内参
    #         K[0, 0] = float(fx)  # 设置 fx
    #         K[1, 1] = float(fy)  # 设置 fy
    #         K[0, 2] = float(cx)  # 设置 cx
    #         K[1, 2] = float(cy)  # 设置 cy
    #         # print("$"*20)  # 可选调试输出
    #         # print("K:", K)  # 打印内参矩阵
    #     else:  # 无显式内参时，依据视场角估算焦距，并用图像中心作为主点
    #         K[0, 0] = fov2focal(cam_info.FovX, w)  # 由水平FOV与宽度估算 fx
    #         K[1, 1] = fov2focal(cam_info.FovY, h)  # 由垂直FOV与高度估算 fy
    #         K[0, 2] = cam_info.width / 2  # 主点cx设为原始图像中心
    #         K[1, 2] = cam_info.height / 2  # 主点cy设为原始图像中心

    #     pts_depth = np.zeros([1, h, w])  # 初始化 [1,H,W] 深度图为零
    #     point_camera = cam_info.pointcloud_camera  # 取相机坐标系下的点云 (N, 3)
    #     # print("^"*20)  # 可选调试输出
    #     # # if id == 18 or id == 19:  # 针对特定ID调试
    #     # #     print("point_camera:",point_camera)  # 打印点云内容
    #     print("point_camera.shape:",cam_info.pointcloud_camera.shape)  # 打印点云形状
    #     # print("point_camera:",point_camera)  # 可选打印点云
    #     uvz = point_camera[point_camera[:, 2] > 0]  # 仅保留相机坐标系下 z>0 的前方点
    #     # print("^"*20)  # 可选分隔线
    #     # print("id",id)  # 打印相机ID
    #     print("uvz.shape:", uvz.shape)  # 打印有效点的数量与形状
    #     # print("uvz:", uvz)  # 可选打印有效点
    #     uvz = uvz @ K.T  # 将 3D 点乘以 K^T 得到未归一化的像素坐标与深度
    #     uvz[:, :2] /= uvz[:, 2:]  # 归一化：u,v = (x/z, y/z)
    #     uvz = uvz[uvz[:, 1] >= 0]  # 过滤 v>=0 的像素
    #     uvz = uvz[uvz[:, 1] < h]  # 过滤 v<h 的像素
    #     uvz = uvz[uvz[:, 0] >= 0]  # 过滤 u>=0 的像素
    #     uvz = uvz[uvz[:, 0] < w]  # 过滤 u<w 的像素
    #     uv = uvz[:, :2]  # 提取像素平面坐标 (u,v)
    #     uv = uv.astype(int)  # 像素坐标转为整数索引
    #     # TODO: may need to consider overlap  # 可能需要处理同一像素被多个点覆盖的情况
    #     pts_depth[0, uv[:, 1], uv[:, 0]] = uvz[:, 2]  # 将对应像素位置的深度写入图像
    #     pts_depth = torch.from_numpy(pts_depth).float()  # 转为 torch 张量
    #     print("pts_depth:", pts_depth.shape)  # 打印生成的深度图形状
    # else:  # 不生成深度图
    #     pts_depth = None  # 置空
    
    # print("---------------pts_depth:", pts_depth.shape)  # 打印生成的深度图形状
    return Camera(  # 构造并返回 Camera 对象（包含图像、掩码、深度等）
        colmap_id=cam_info.uid,  # COLMAP用的相机ID（或数据源ID）
        uid=id,  # 内部相机唯一ID
        R=cam_info.R,  # 相机旋转矩阵（世界->相机）
        T=cam_info.T,  # 相机平移向量（世界->相机）
        FoVx=cam_info.FovX,  # 水平视场角
        FoVy=cam_info.FovY,  # 垂直视场角
        cx=cx,  # 主点cx（按缩放后）
        cy=cy,  # 主点cy（按缩放后）
        fx=fx,  # 焦距fx（按缩放后）
        fy=fy,  # 焦距fy（按缩放后）
        image=gt_image,  # 训练用图像张量 [3,H,W]
        image_name=cam_info.image_name,  # 图像文件名
        data_device=args.data_device,  # 数据张量所在设备（cuda/cpu）
        timestamp=cam_info.timestamp,  # 时间戳（用于动态建模）
        resolution=resolution,  # 训练分辨率 (W,H)
        image_path=cam_info.image_path,  # 图像路径
        pts_depth=pts_depth,  # 深度图张量 [1,H,W] 或 None
        sky_mask=sky_mask,  # 天空掩码张量或 None
        dynamic_mask=dynamic_mask,  # 动态掩码张量或 None
        bbox_mask=bbox_mask  # 包围盒掩码张量或 None
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(tqdm(cam_infos)):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]

    if camera.cx is None:
        camera_entry = {
            "id": id,
            "img_name": camera.image_name,
            "width": camera.width,
            "height": camera.height,
            "position": pos.tolist(),
            "rotation": serializable_array_2d,
            "FoVx": camera.FovX,
            "FoVy": camera.FovY,
        }
    else:
        camera_entry = {
            "id": id,
            "img_name": camera.image_name,
            "width": camera.width,
            "height": camera.height,
            "position": pos.tolist(),
            "rotation": serializable_array_2d,
            "fx": camera.fx,
            "fy": camera.fy,
            "cx": camera.cx,
            "cy": camera.cy,
        }
    return camera_entry
