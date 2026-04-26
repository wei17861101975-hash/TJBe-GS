import os
import json
import numpy as np
import cv2
from typing import Dict, List, Tuple
from scipy.spatial.transform import Rotation as R

# 说明：
# 本模块用于将 TUM RGB-D 数据集转换为与 Waymo 预处理一致的 pointcloud.npz 文件结构，
# 其中包含：
# - 'pointcloud'：dict[int -> (N, 3)]，每帧的 3D 点（世界坐标）
# - 'camera_projection'：dict[int -> (N, 3)]，每帧每点对应的 [camera_id, u, v]
#
# 设计目标：
# - 兼容下游 BezierGS 的 build_pointcloud 流程（读取上述两个键）
# - 支持 TUM 单相机（camera_id 固定为 0）
# - 使用 association 和 indices 对齐 RGB、Depth、Pose 并选择需要的帧
# - 深度单位转换可通过 depth_scale 控制（例如 TUM 常见深度值需 /5000 转米）
#
# 使用示例（命令行或脚本）：
# build_pointcloud_npz_from_tum(
#     data_root="E:/TUM/rgbd_dataset",
#     image_dir=os.path.join("E:/TUM/rgbd_dataset", "rgb"),
#     depth_dir=os.path.join("E:/TUM/rgbd_dataset", "depth"),
#     intrinsics=(fx, fy, cx, cy),
#     groundtruth_path=os.path.join("E:/TUM/rgbd_dataset", "groundtruth.txt"),
#     association_path=os.path.join("E:/TUM/rgbd_dataset", "association.txt"),
#     indices_path=os.path.join("E:/TUM/rgbd_dataset", "indices.txt"),
#     out_npz_path=os.path.join("E:/TUM/rgbd_dataset", "pointcloud.npz"),
#     depth_scale=5000.0
# )


def _parse_groundtruth(groundtruth_path: str) -> Dict[float, Dict[str, float]]:
    """
    解析 TUM 的 groundtruth.txt，返回 time->pose 字典
    每行格式：timestamp tx ty tz qx qy qz qw
    """
    gt = {}
    with open(groundtruth_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                continue
            t = float(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8])
            gt[t] = {"tx": tx, "ty": ty, "tz": tz, "qx": qx, "qy": qy, "qz": qz, "qw": qw}
    return gt


def _list_sorted_images(dir_path: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    files = [f for f in os.listdir(dir_path) if f.lower().endswith(exts)]
    def _key(fname: str):
        base = os.path.splitext(os.path.basename(fname))[0]
        try:
            return float(base)
        except:
            return base
    files.sort(key=_key)
    return files


def _parse_groundtruth_list(groundtruth_path: str) -> List[Dict[str, float]]:
    """
    按文件行顺序解析 groundtruth.txt，跳过以 '#' 开头的注释行。
    返回列表，每项为 {'ts','tx','ty','tz','qx','qy','qz','qw'}，索引从 0 开始。
    """
    out = []
    with open(groundtruth_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                continue
            ts = float(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8])
            out.append({"ts": ts, "tx": tx, "ty": ty, "tz": tz, "qx": qx, "qy": qy, "qz": qz, "qw": qw})
    return out


def _parse_associations(association_path: str, image_dir: str = None, depth_dir: str = None, groundtruth_path: str = None) -> List[Dict]:
    """
    解析 association 文件，将 RGB、Depth 与时间戳/位姿索引对齐。
    支持两种格式：
      1) TUM 官方格式：<rgb_ts> <rgb_path> <depth_ts> <depth_path>
      2) 索引格式：首行为表头，后续行为 "(rgb_index, depth_index, time_index)"
         - rgb_index/depth_index 以 0 为起始，指向 image_dir/depth_dir 下排序后的文件列表
         - time_index 以 0 为起始，对应 groundtruth.txt 中去除注释后的第 N 行
    返回列表，每项包含至少：
      - 对于格式(1)：{"rgb_ts": float, "rgb_path": str, "depth_ts": float, "depth_path": str}
      - 对于格式(2)：{"rgb_ts": float, "rgb_path": str, "depth_ts": float, "depth_path": str, "pose_index": int}
    """
    pairs = []
    lines = []
    with open(association_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)

    if not lines:
        return pairs

    # 判断是否为索引格式（包含括号）
    is_index_format = any(("(" in ln and ")" in ln) for ln in lines)

    if not is_index_format:
        # 兼容原 TUM 官方格式
        for line in lines:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                rgb_ts = float(parts[0])
            except:
                # 有些文件首行可能是表头，跳过
                continue
            rgb_path = parts[1]
            depth_ts = float(parts[2])
            depth_path = parts[3]
            pairs.append(
                {"rgb_ts": rgb_ts, "rgb_path": rgb_path, "depth_ts": depth_ts, "depth_path": depth_path}
            )
        return pairs

    # 索引格式需要目录与 groundtruth
    if image_dir is None or depth_dir is None or groundtruth_path is None:
        raise ValueError("索引格式的 association 解析需要提供 image_dir, depth_dir, groundtruth_path")

    rgb_files = _list_sorted_images(image_dir)
    depth_files = _list_sorted_images(depth_dir)
    gt_list = _parse_groundtruth_list(groundtruth_path)
    assert len(rgb_files) > 0 and len(depth_files) > 0, "RGB/Depth 目录为空或未找到图像文件"
    assert len(gt_list) > 0, "groundtruth 为空或解析失败"

    # 首行若是表头，跳过
    start_idx = 0
    if not lines[0].startswith("("):
        start_idx = 1

    for i in range(start_idx, len(lines)):
        ln = lines[i]
        # 解析形如 "(a, b, c)" 的元组
        s = ln.strip()
        if s[0] == "(" and s[-1] == ")":
            s = s[1:-1]
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 3:
            continue
        try:
            ri = int(parts[0])
            di = int(parts[1])
            ti = int(parts[2])
        except:
            continue
        assert 0 <= ri < len(rgb_files), f"rgb_index 越界: {ri}"
        assert 0 <= di < len(depth_files), f"depth_index 越界: {di}"
        assert 0 <= ti < len(gt_list), f"time_index 越界: {ti}"

        rgb_path = os.path.join(image_dir, rgb_files[ri])
        depth_path = os.path.join(depth_dir, depth_files[di])
        def _file_ts(path: str):
            try:
                return float(os.path.splitext(os.path.basename(path))[0])
            except:
                return None
        rgb_ts = _file_ts(rgb_path)
        depth_ts = _file_ts(depth_path)
        gt = gt_list[ti]
        pairs.append(
            {
                "rgb_ts": rgb_ts if rgb_ts is not None else gt["ts"],
                "rgb_path": rgb_path,
                "depth_ts": depth_ts if depth_ts is not None else gt["ts"],
                "depth_path": depth_path,
                "pose_index": ti,
            }
        )
    return pairs


def _parse_indices(indices_path: str) -> List[int]:
    idxs = []
    if not os.path.exists(indices_path):
        alt = os.path.join(os.path.dirname(indices_path), "indicies.txt")
        if os.path.exists(alt):
            indices_path = alt
        else:
            raise FileNotFoundError(f"indices 文件不存在: {indices_path}")
    with open(indices_path, "r") as f:
        content = f.read().strip()
    if not content:
        return idxs
    if content.startswith("[") and content.endswith("]"):
        import ast
        try:
            arr = ast.literal_eval(content)
            for v in arr:
                try:
                    idxs.append(int(v))
                except:
                    pass
            return idxs
        except:
            pass
    for line in content.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        s = s.replace("[", " ").replace("]", " ").replace("(", " ").replace(")", " ").replace(",", " ").strip()
        for token in s.split():
            try:
                idxs.append(int(token))
            except:
                pass
    return idxs


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """
    四元数转旋转矩阵（3x3）
    """
    return R.from_quat([qx, qy, qz, qw]).as_matrix()


def _nearest_pose(gt: Dict[float, Dict[str, float]], ts: float) -> Dict[str, float]:
    """
    在 groundtruth 字典中查找与给定 ts 最近的位姿（简单最近邻）。
    """
    times = np.array(list(gt.keys()))
    i = int(np.argmin(np.abs(times - ts)))
    return gt[times[i]]


def _backproject_and_transform(
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    pose: Dict[str, float],
    depth_scale: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将深度图反投影到相机坐标系，再用位姿变换到世界坐标系。
    返回：
      Xw: (N, 3) 世界坐标
      uu: (N,) 像素 u（列）
      vv: (N,) 像素 v（行）
    """
    # 深度单位转换（例如 TUM 通常需要 /5000.0 得到米）
    depth_m = depth.astype(np.float32)
    if depth_scale and depth_scale > 0:
        depth_m = depth_m / depth_scale

    H, W = depth_m.shape
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)

    z = depth_m.reshape(-1)
    valid = z > 0
    uu = uu.reshape(-1)[valid]
    vv = vv.reshape(-1)[valid]
    z = z[valid]

    # 相机坐标系下点
    xc = (uu - cx) / fx * z
    yc = (vv - cy) / fy * z
    Xc = np.stack([xc, yc, z], axis=1)  # (N, 3)

    # 位姿：世界 <- 相机
    R_w_c = _quat_to_rot(pose["qx"], pose["qy"], pose["qz"], pose["qw"])
    T_w_c = np.array([pose["tx"], pose["ty"], pose["tz"]], dtype=np.float32)
    Xw = (R_w_c @ Xc.T).T + T_w_c
    return Xw.astype(np.float32), uu.astype(np.int32), vv.astype(np.int32)


def build_pointcloud_npz_from_tum(
    data_root: str,
    image_dir: str,
    depth_dir: str,
    intrinsics: Tuple[float, float, float, float],
    groundtruth_path: str,
    association_path: str,
    indices_path: str,
    out_npz_path: str,
    depth_scale: float = 5000.0,
    mask_dir: str = None,
    object_ids: List[int] = None
):
    """
    主入口：基于 TUM 输入生成 Waymo 兼容的 pointcloud.npz。
    参数：
      - data_root：数据集根目录（用于保持一致路径结构）
      - image_dir：RGB 图像目录（仅用于存在性校验）
      - depth_dir：深度图目录（仅用于存在性校验）
      - intrinsics：(fx, fy, cx, cy)
      - groundtruth_path：groundtruth.txt 文件路径
      - association_path：association.txt 文件路径（RGB-Depth 对齐）
      - indices_path：indices.txt 文件路径（选择帧）
      - out_npz_path：输出 npz 文件路径
      - depth_scale：深度单位转换比例（默认 5000.0）
    产出：
      - np.savez(out_npz_path, pointcloud=..., camera_projection=...)
    """
    fx, fy, cx, cy = intrinsics

    # 解析 groundtruth（时间戳->位姿）
    gt = _parse_groundtruth(groundtruth_path)
    assert len(gt) > 0, "groundtruth 为空或解析失败"

    # 解析 association（对齐 RGB/Depth 与时间戳/位姿索引）
    pairs = _parse_associations(association_path, image_dir=image_dir, depth_dir=depth_dir, groundtruth_path=groundtruth_path)
    assert len(pairs) > 0, "association 为空或解析失败"

    # 解析 indices（选择帧）
    idxs = _parse_indices(indices_path)
    assert len(idxs) > 0, "indices 为空或解析失败"

    pointcloud: Dict[int, np.ndarray] = {}
    camera_projection: Dict[int, np.ndarray] = {}
    point_labels: Dict[int, np.ndarray] = {}
    if object_ids is None:
        object_ids = [0, 1]

    for frame_idx in idxs:
        assert 0 <= frame_idx < len(pairs), f"indices 包含非法索引 {frame_idx}"
        pair = pairs[frame_idx]

        # 相机 ID：TUM 单相机，固定为 0
        camera_id = 0

        # 解析并加载图像与深度路径（association 文件通常是相对路径）
        rgb_path = os.path.join(data_root, pair["rgb_path"]) if not os.path.isabs(pair["rgb_path"]) else pair["rgb_path"]
        depth_path = os.path.join(data_root, pair["depth_path"]) if not os.path.isabs(pair["depth_path"]) else pair["depth_path"]

        # 读取深度（不必读取 RGB，这里只需深度与像素坐标）
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        assert depth is not None, f"无法读取深度图：{depth_path}"

        # 位姿选择：若提供 pose_index（索引格式），则直接使用；否则按时间戳最近邻
        if "pose_index" in pair:
            gt_list = _parse_groundtruth_list(groundtruth_path)
            pose = gt_list[pair["pose_index"]]
        else:
            ts = pair["rgb_ts"]
            pose = _nearest_pose(gt, ts)

        # 反投影并转换到世界坐标
        Xw, uu, vv = _backproject_and_transform(depth, fx, fy, cx, cy, pose, depth_scale)

        # 生成 camera_projection
        proj = np.concatenate([np.full((Xw.shape[0], 1), camera_id, dtype=np.int32),
                               uu[:, None], vv[:, None]], axis=1).astype(np.int32)

        pointcloud[frame_idx] = Xw
        camera_projection[frame_idx] = proj
        if mask_dir is not None and len(object_ids) > 0:
            labels = np.full((Xw.shape[0],), -1, dtype=np.int32)
            assigned = np.zeros(Xw.shape[0], dtype=bool)
            for oid in object_ids:
                mask = _load_mask_image(mask_dir, frame_idx, oid)
                if mask is None:
                    continue
                hit = mask[vv, uu]
                if np.any(hit):
                    labels[hit] = int(oid)
                    assigned[hit] = True
            point_labels[frame_idx] = labels.astype(np.int32)

    # 保存为 npz（与 Waymo 的键一致）
    if len(point_labels) > 0:
        np.savez(out_npz_path, pointcloud=pointcloud, camera_projection=camera_projection, point_labels=point_labels)
    else:
        np.savez(out_npz_path, pointcloud=pointcloud, camera_projection=camera_projection)


if __name__ == "__main__":
    # 可选：示例运行（请按需修改路径与内参）
    # 由于环境差异，这里仅提供参考，不主动执行。
    example = False
    if example:
        build_pointcloud_npz_from_tum(
            data_root="E:/TUM/rgbd_dataset",
            image_dir=os.path.join("E:/TUM/rgbd_dataset", "rgb"),
            depth_dir=os.path.join("E:/TUM/rgbd_dataset", "depth"),
            intrinsics=(525.0, 525.0, 319.5, 239.5),  # 示例内参，需使用实际值
            groundtruth_path=os.path.join("E:/TUM/rgbd_dataset", "groundtruth.txt"),
            association_path=os.path.join("E:/TUM/rgbd_dataset", "association.txt"),
            indices_path=os.path.join("E:/TUM/rgbd_dataset", "indices.txt"),
            out_npz_path=os.path.join("E:/TUM/rgbd_dataset", "pointcloud.npz"),
            depth_scale=5000.0
        )


# -----------------------------
# 动态物体 mask 融合与 input_ply 生成
# -----------------------------
def _load_mask_image(mask_dir: str, frame_idx: int, obj_id: int) -> np.ndarray:
    """
    加载给定帧与对象的 mask 图（白为前景，黑为背景），命名：frame_{X}_{obj_id}.png
    返回布尔数组 (H, W)，True 表示属于该对象。
    """
    mask_path = os.path.join(mask_dir, f"frame_{frame_idx}_{obj_id}.png")
    if not os.path.exists(mask_path):
        return None
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return (img > 0)


def build_input_ply_from_tum_masks(
    data_root: str,
    npz_path: str,
    association_path: str,
    indices_path: str,
    mask_dir: str,
    out_model_path: str,
    object_ids: List[int] = [0, 1],
    initial_num_obj: int = 20000,
    bg_per_frame_max: int = 2000,
    bg_max_points: int = 300000,
    voxel_size_bg: float = 0.15,
    outlier_nb_points: int = 10,
    outlier_radius: float = 0.5,
    obj_ref_mode: str = "max",
    obj_ref_frames: Dict[int, int] = None
):
    """
    基于已生成的 pointcloud.npz 与动态物体 mask，将点按帧分配到各对象或背景，
    并写入 out_model_path/input_ply 目录下的 points3D_bkgd.ply、points3D_obj_XXX.ply。
    颜色来自对应帧的 RGB 像素（通过 camera_projection 的 [u,v] 采样）。
    """
    from .general_utils import save_ply  # 使用现有 PLY 写入工具

    os.makedirs(os.path.join(out_model_path, "input_ply"), exist_ok=True)

    # 加载 pointcloud 与投影
    data = np.load(npz_path, allow_pickle=True)
    pc_dict = data["pointcloud"].item()
    proj_dict = data["camera_projection"].item()

    # 加载帧到 RGB/Depth 路径与时间戳/位姿索引的映射（支持索引格式）
    pairs = _parse_associations(
        association_path,
        image_dir=os.path.join(data_root, "rgb"),
        depth_dir=os.path.join(data_root, "depth"),
        groundtruth_path=os.path.join(data_root, "groundtruth.txt"),
    )
    idxs = _parse_indices(indices_path)

    # 聚合容器
    points_xyz_dict: Dict[str, List[np.ndarray]] = {"bkgd": []}
    points_rgb_dict: Dict[str, List[np.ndarray]] = {"bkgd": []}
    for oid in object_ids:
        key = f"obj_{oid:03d}"
        points_xyz_dict[key] = []
        points_rgb_dict[key] = []
    obj_samples: Dict[int, List[tuple]] = {oid: [] for oid in object_ids}
    print("---------------------开始循环------------------------------")
    for frame_idx in idxs:
        assert frame_idx in pc_dict and frame_idx in proj_dict, f"npz 缺少帧 {frame_idx}"
        Xw = pc_dict[frame_idx]  # (N,3)
        proj = proj_dict[frame_idx]  # (N,3) = [cam_id, u, v]
        uu = proj[:, 1].astype(np.int32)
        vv = proj[:, 2].astype(np.int32)
        # print("---------------------采样颜色------------------------------")

        # 读取该帧的 RGB，以便采样颜色
        rgb_path = pairs[frame_idx]["rgb_path"]
        rgb_path = os.path.join(data_root, rgb_path) if not os.path.isabs(rgb_path) else rgb_path
        image = cv2.imread(rgb_path)[..., ::-1] / 255.0  # 转为 RGB，范围 [0,1]
        H, W, _ = image.shape

        # 像素边界过滤
        in_bounds = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        Xw = Xw[in_bounds]
        uu = uu[in_bounds]
        vv = vv[in_bounds]
        colors = image[vv, uu]

        # 加载各对象的 mask
        # 对于每个点，按 mask 归类到第一个命中的对象；否则归入背景
        assigned = np.zeros(Xw.shape[0], dtype=bool)
        for oid in object_ids:
            key = f"obj_{oid:03d}"
            # print("---------------------加载mask------------------------------")
            mask = _load_mask_image(mask_dir, frame_idx, oid)
            if mask is None:
                continue
            # 点级命中：mask[v, u] == True
            hit = mask[vv, uu]
            if np.any(hit):
                points_xyz_dict[key].append(Xw[hit])
                points_rgb_dict[key].append(colors[hit])
                assigned[hit] = True
                obj_samples[oid].append((frame_idx, Xw[hit], colors[hit]))

        # 未分配的进入背景
        bg_mask = ~assigned
        if np.any(bg_mask):
            cur_bg_xyz = Xw[bg_mask]
            cur_bg_rgb = colors[bg_mask]
            if cur_bg_xyz.shape[0] > int(bg_per_frame_max):
                sel = np.random.choice(cur_bg_xyz.shape[0], int(bg_per_frame_max), replace=False)
                cur_bg_xyz = cur_bg_xyz[sel]
                cur_bg_rgb = cur_bg_rgb[sel]
            points_xyz_dict["bkgd"].append(cur_bg_xyz)
            points_rgb_dict["bkgd"].append(cur_bg_rgb)

    # 合并并写入
    # 背景做体素降采样与半径滤波；对象做限量采样
    # 背景
    print("--------------背景------------------------")
    if len(points_xyz_dict["bkgd"]) > 0:
        bg_xyz = np.concatenate(points_xyz_dict["bkgd"], axis=0)
        bg_rgb = np.concatenate(points_rgb_dict["bkgd"], axis=0)
        if bg_xyz.shape[0] > int(bg_max_points):
            sel = np.random.choice(bg_xyz.shape[0], int(bg_max_points), replace=False)
            bg_xyz = bg_xyz[sel]
            bg_rgb = bg_rgb[sel]
        # open3d 体素与滤波（与 waymo_utils 中一致）
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        print("----------------------------生成ply_1-------------------------")
        pcd.points = o3d.utility.Vector3dVector(bg_xyz.astype(np.float32))
        pcd.colors = o3d.utility.Vector3dVector(bg_rgb.astype(np.float32))
        print("----------------------------生成ply_2-------------------------")
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size_bg)
        pcd, _ = pcd.remove_radius_outlier(nb_points=outlier_nb_points, radius=outlier_radius)
        bg_xyz_ds = np.asarray(pcd.points).astype(np.float32)
        bg_rgb_ds = np.asarray(pcd.colors).astype(np.float32)
        print("----------------------------生成ply_3-------------------------")
        save_ply(bg_xyz_ds, os.path.join(out_model_path, "input_ply", "points3D_bkgd.ply"), rgbs=bg_rgb_ds)

    # 动态对象
    for oid in object_ids:
        key = f"obj_{oid:03d}"
        if len(obj_samples[oid]) == 0:
            continue
        if obj_ref_frames is not None and oid in obj_ref_frames:
            target_frame = int(obj_ref_frames[oid])
            cand = [i for i, s in enumerate(obj_samples[oid]) if s[0] == target_frame]
            if len(cand) > 0:
                sel_idx = cand[0]
            else:
                if obj_ref_mode == "max":
                    sel_idx = int(np.argmax([s[1].shape[0] for s in obj_samples[oid]]))
                else:
                    sel_idx = int(np.argmin([s[0] for s in obj_samples[oid]]))
        else:
            if obj_ref_mode == "max":
                sel_idx = int(np.argmax([s[1].shape[0] for s in obj_samples[oid]]))
            else:
                sel_idx = int(np.argmin([s[0] for s in obj_samples[oid]]))
        ref_xyz = obj_samples[oid][sel_idx][1]
        ref_rgb = obj_samples[oid][sel_idx][2]
        if ref_xyz.shape[0] > initial_num_obj:
            sel = np.random.choice(ref_xyz.shape[0], initial_num_obj, replace=False)
            ref_xyz = ref_xyz[sel]
            ref_rgb = ref_rgb[sel]
        save_ply(ref_xyz.astype(np.float32), os.path.join(out_model_path, "input_ply", f"points3D_obj_{oid:03d}.ply"), rgbs=ref_rgb.astype(np.float32))


def build_ply_dict_from_tum_masks_and_centers(  # 入口：基于TUM数据生成BezierGS初始化用的ply_dict
    data_root: str,  # 数据集根目录
    npz_path: str,  # pointcloud.npz文件路径
    association_path: str,  # association.txt路径（RGB/Depth/时间对齐）
    indices_path: str,  # indices.txt路径（选择进入建模的帧索引）
    mask_dir: str,  # 动态物体二值mask目录，命名frame_{frame}_{obj}.png
    detection_json_path: str = None,  # 可选检测结果JSON（提供某些帧的中心world坐标）
    object_ids: List[int] = [0, 1],  # 需要处理的对象id列表
    initial_num_obj: int = 20000,  # 每个对象参考帧的最大采样点数
    min_mask_pixels: int = 1000,
    bg_per_frame_max: int = 2000,  # 每帧背景最大采样点数（防止内存暴涨）
    bg_max_points: int = 300000,   # 背景全局最大点数上限
    bg_voxel_size: float = None,   # 可选体素下采样（例如0.15），None表示不进行体素下采样
    input_ply_dir: str = None,     # 可选，若提供则优先使用该目录下的 points3D_obj_XXX.ply 作为参考形状
    obj_ref_frames: Dict[int, int] = None,  # 可选，指定各对象参考帧，用于计算参考中心
) -> Dict[str, dict]:  # 返回：ply_dict，包含背景与各对象的轨迹与偏移
    """
    直接生成 BezierGS 需要的 ply_dict：  # 说明：生成BezierGS初始化所需的数据结构
      - 'bkgd': {'xyz_array','colors_array','start_frame','end_frame'}  # 背景点与颜色及帧范围
      - 'obj_XXX': {'xyz_offset','trajectory','colors_array','start_frame','end_frame','timestamp_list'}  # 对象偏移、轨迹、颜色与时间列表
    轨迹（trajectory）与 timestamp_list：  # 轨迹与时间戳来源
      - 优先从 detection_results.json 读取（如包含世界坐标中心），否则用该帧对象点的世界坐标质心估计中心。  # 检测中心优先，其次质心
    相对偏移（xyz_offset）：  # 局部偏移生成方式
      - 选取每个对象的参考帧的点集，计算 offset_const = Xw_ref - center_world_ref，  # 参考帧点减中心为偏移常量
        并在所有帧重复（刚体假设）。形状 [N, T, 3]。  # 在所有帧复制，得到(N,T,3)
    """
    # 载入 pointcloud 与投影  # 从npz读取全局点云与相机投影
    data = np.load(npz_path, allow_pickle=True)  # 允许pickle，读取字典
    pc_dict = data["pointcloud"].item()  # 每帧的3D点字典
    proj_dict = data["camera_projection"].item()  # 每帧每点的[cam_id,u,v]
    print(f"--------------------获取ply_dict初始化，加载{len(pc_dict)}帧点云与{len(proj_dict)}帧投影---------------------------------")

    # 加载帧到 RGB/Depth 路径与时间戳/位姿索引（支持索引格式）  # 建立帧索引到路径与时间映射
    pairs = _parse_associations(  # 解析association，支持(ri,di,ti)格式
        association_path,  # 文件路径
        image_dir=os.path.join(data_root, "rgb"),  # RGB目录
        depth_dir=os.path.join(data_root, "depth"),  # 深度目录
        groundtruth_path=os.path.join(data_root, "groundtruth.txt"),  # 位姿文件
    )
    idxs = _parse_indices(indices_path)  # 解析indices，得到参与建模的帧索引列表

    # 不使用检测JSON作为中心来源，中心统一由掩码命中点云的均值计算

    # 聚合容器  # 用于收集背景点与对象数据
    bg_xyz_all: List[np.ndarray] = []  # 背景点坐标列表
    bg_rgb_all: List[np.ndarray] = []  # 背景点颜色列表

    obj_points_per_frame: Dict[int, Dict[int, np.ndarray]] = {oid: {} for oid in object_ids}   # obj_id -> {frame_idx: Xw_obj}  # 每帧对象点云
    obj_colors_per_frame: Dict[int, Dict[int, np.ndarray]] = {oid: {} for oid in object_ids}   # obj_id -> {frame_idx: RGB}  # 每帧对象点颜色
    obj_center_world: Dict[int, Dict[int, np.ndarray]] = {oid: {} for oid in object_ids}       # obj_id -> {frame_idx: center_world}  # 每帧对象中心
    obj_timestamps: Dict[int, List[float]] = {oid: [] for oid in object_ids}                   # obj_id -> list of timestamps aligned to frames  # 时间戳列表
    obj_frames: Dict[int, List[int]] = {oid: [] for oid in object_ids}                         # obj_id -> list of frame indices  # 帧索引列表
    print(f"--------------------获取ply_dict初始化，开始处理{len(idxs)}帧---------------------------------")
    
    for frame_idx in idxs:  # 遍历选中的帧索引
        if frame_idx not in pc_dict or frame_idx not in proj_dict:  # 若npz缺少该帧数据
            print(f"缺少帧{frame_idx}的点云或投影数据")
            continue  # 跳过
        Xw = pc_dict[frame_idx]  # 该帧的世界坐标点(N,3)
        proj = proj_dict[frame_idx]  # 该帧的投影信息(N,3)
        uu = proj[:, 1].astype(np.int32)  # 像素列坐标u
        vv = proj[:, 2].astype(np.int32)  # 像素行坐标v

        rgb_path = pairs[frame_idx]["rgb_path"]  # 该帧对应的RGB路径（可能相对）
        rgb_path = os.path.join(data_root, rgb_path) if not os.path.isabs(rgb_path) else rgb_path  # 规范为绝对路径
        image = cv2.imread(rgb_path)[..., ::-1] / 255.0  # 读取并转BGR->RGB，归一化到[0,1]
        H, W, _ = image.shape  # 图像尺寸

        in_bounds = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)  # 像素边界过滤掩码
        Xw = Xw[in_bounds]  # 只保留可见像素对应的点
        uu = uu[in_bounds]  # 过滤后的u
        vv = vv[in_bounds]  # 过滤后的v
        colors = image[vv, uu]  # 采样颜色

        # print(f"--------------------获取ply_dict，处理帧{frame_idx}，可见点{Xw.shape[0]}---------------------------------")
        assigned = np.zeros(Xw.shape[0], dtype=bool)  # 记录每点是否已分配给某对象
        for oid in object_ids:  # 遍历对象id
            mask = _load_mask_image(mask_dir, frame_idx, oid)  # 加载该帧该对象掩码
            hit = None  # 命中掩码布尔数组
            if mask is not None:  # 若掩码存在
                if int(mask.sum()) < int(min_mask_pixels):
                    continue
                hit = mask[vv, uu]  # 取每点的掩码命中
            if hit is not None and np.any(hit):  # 若有命中点
                obj_points_per_frame[oid][frame_idx] = Xw[hit]  # 保存对象点云
                obj_colors_per_frame[oid][frame_idx] = colors[hit]  # 保存对象点颜色
                assigned[hit] = True  # 标记已分配
                obj_frames[oid].append(frame_idx)  # 记录该帧
                obj_timestamps[oid].append(pairs[frame_idx]["rgb_ts"])  # 记录时间戳
                center_world = np.mean(Xw[hit], axis=0).astype(np.float32)  # 用掩码命中点云的均值作为中心
                obj_center_world[oid][frame_idx] = center_world  # 保存中心
        # print(f"--------------------分配bg_mask---------------------------------")
        bg_mask = ~assigned  # 未分配的点视为背景
        if np.any(bg_mask):  # 若存在背景点
            cur_bg_xyz = Xw[bg_mask]
            cur_bg_rgb = colors[bg_mask]
            if cur_bg_xyz.shape[0] > int(bg_per_frame_max):
                sel = np.random.choice(cur_bg_xyz.shape[0], int(bg_per_frame_max), replace=False)
                cur_bg_xyz = cur_bg_xyz[sel]
                cur_bg_rgb = cur_bg_rgb[sel]
            bg_xyz_all.append(cur_bg_xyz)  # 收集背景坐标（已限采）
            bg_rgb_all.append(cur_bg_rgb)  # 收集背景颜色（已限采）

    # 组装 ply_dict  # 初始化字典
    ply_dict: Dict[str, dict] = {}  # 最终返回的数据结构

    # 背景：若提供了 input_ply_dir 则优先使用该目录下的 points3D_bkgd.ply，否则合并并可选下采样
    print(f"--------------------获取ply_dict，处理背景{len(bg_xyz_all)}帧---------------------------------")
    use_input_bkgd = False
    if input_ply_dir is not None:
        bkgd_ply = os.path.join(input_ply_dir, "points3D_bkgd.ply")
        if os.path.isfile(bkgd_ply):
            try:
                from .general_utils import load_ply
            except Exception:
                from utils.general_utils import load_ply
            xyz_loaded, colors_loaded = load_ply(bkgd_ply)
            ply_dict["bkgd"] = {
                "xyz_array": xyz_loaded.astype(np.float32),
                "colors_array": colors_loaded.astype(np.float32),
                "start_frame": min(idxs) if len(idxs) > 0 else 0,
                "end_frame": max(idxs) if len(idxs) > 0 else 0,
            }
            use_input_bkgd = True
    if not use_input_bkgd:
        if len(bg_xyz_all) > 0:
            bg_xyz = np.concatenate(bg_xyz_all, axis=0).astype(np.float32)
            bg_rgb = np.concatenate(bg_rgb_all, axis=0).astype(np.float32)
            if bg_voxel_size is not None and bg_voxel_size > 0:
                try:
                    import open3d as o3d
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(bg_xyz.astype(np.float32))
                    pcd.colors = o3d.utility.Vector3dVector(bg_rgb.astype(np.float32))
                    pcd = pcd.voxel_down_sample(voxel_size=float(bg_voxel_size))
                    bg_xyz = np.asarray(pcd.points).astype(np.float32)
                    bg_rgb = np.asarray(pcd.colors).astype(np.float32)
                except Exception as e:
                    print(f"[WARN] open3d voxel downsample failed: {e}")
            if bg_xyz.shape[0] > int(bg_max_points):
                sel = np.random.choice(bg_xyz.shape[0], int(bg_max_points), replace=False)
                bg_xyz = bg_xyz[sel]
                bg_rgb = bg_rgb[sel]
            ply_dict["bkgd"] = {
                "xyz_array": bg_xyz,
                "colors_array": bg_rgb,
                "start_frame": min(idxs) if len(idxs) > 0 else 0,
                "end_frame": max(idxs) if len(idxs) > 0 else 0,
            }
        else:
            ply_dict["bkgd"] = {
                "xyz_array": np.zeros((0, 3), dtype=np.float32),
                "colors_array": np.zeros((0, 3), dtype=np.float32),
                "start_frame": min(idxs) if len(idxs) > 0 else 0,
                "end_frame": max(idxs) if len(idxs) > 0 else 0,
            }

    # 动态对象  # 为每个对象构建条目
    print(f"--------------------获取ply_dict，开始处理{len(object_ids)}个对象---------------------------------")
    for oid in object_ids:  # 遍历对象
        frames = sorted(set(obj_frames[oid]))  # 去重并排序对象存在的帧
        if len(frames) == 0:  # 若该对象无帧
            ply_dict[f"obj_{oid:03d}"] = {  # 写空条目
                "xyz_offset": None,  # 无偏移
                "trajectory": None,  # 无轨迹
                "colors_array": None,  # 无颜色
                "start_frame": None,  # 无起始
                "end_frame": None,  # 无结束
                "timestamp_list": [],  # 空时间
            }
            continue  # 下一个对象
        traj_list = []  # 轨迹点列表
        ts_list = []  # 时间戳列表
        for fi in frames:  # 遍历该对象的帧
            traj_list.append(obj_center_world[oid][fi])  # 加入中心world点
            ts_list.append(pairs[fi]["rgb_ts"])  # 加入对应时间戳
        trajectory = np.stack(traj_list, axis=0).astype(np.float32)  # 转为(T,3)轨迹数组
        timestamp_list = ts_list  # 保存时间戳列表

        ref_point_frames = sorted(list(obj_points_per_frame[oid].keys()))  # 有有效点云的帧集合
        if len(ref_point_frames) == 0:  # 若没有任何帧有对象点云
            ply_dict[f"obj_{oid:03d}"] = {  # 只提供轨迹而无偏移与颜色
                "xyz_offset": None,  # 无偏移
                "trajectory": trajectory,  # 轨迹点
                "colors_array": None,  # 无颜色
                "start_frame": frames[0],  # 该对象起始帧
                "end_frame": frames[-1],  # 该对象结束帧
                "timestamp_list": timestamp_list,  # 时间戳列表
            }
            continue  # 下一个对象
        # 参考帧优先使用配置的 obj_ref_frames，其次选择第一个有点云的帧
        ref_frame = ref_point_frames[0]
        if obj_ref_frames is not None and oid in obj_ref_frames:
            wanted = int(obj_ref_frames[oid])
            if wanted in obj_points_per_frame[oid]:
                ref_frame = wanted
        # 参考点与颜色：若提供了 input_ply_dir 则使用其点云作为参考形状；否则使用该参考帧的点云与颜色
        ref_points = None
        ref_colors = None
        use_input_obj = False
        if input_ply_dir is not None:
            obj_ply = os.path.join(input_ply_dir, f"points3D_obj_{oid:03d}.ply")
            if os.path.isfile(obj_ply):
                try:
                    from .general_utils import load_ply
                except Exception:
                    from utils.general_utils import load_ply
                pts_loaded, colors_loaded = load_ply(obj_ply)
                ref_points = pts_loaded.astype(np.float32)
                ref_colors = colors_loaded.astype(np.float32)
                use_input_obj = True
        if not use_input_obj:
            ref_points = obj_points_per_frame[oid][ref_frame]
            ref_colors = obj_colors_per_frame[oid][ref_frame]
        ref_center = obj_center_world[oid][ref_frame]  # 参考帧的对象中心(3,)

        if ref_points.shape[0] > initial_num_obj:  # 点数过多则下采样
            sel = np.random.choice(ref_points.shape[0], initial_num_obj, replace=False)  # 随机选择子集
            ref_points = ref_points[sel]  # 采样点云
            ref_colors = ref_colors[sel]  # 采样颜色

        offset_const = (ref_points - ref_center[None, :]).astype(np.float32)  # 计算参考帧的偏移常量(N,3)
        T = len(frames)  # 对象存在帧数
        xyz_offset = np.repeat(offset_const[:, None, :], T, axis=1)  # 复制到所有帧(N,T,3)

        ply_dict[f"obj_{oid:03d}"] = {  # 写入对象条目
            "xyz_offset": xyz_offset,  # 局部偏移序列
            "trajectory": trajectory,  # 轨迹序列
            "colors_array": ref_colors.astype(np.float32),  # 参考帧颜色
            "start_frame": frames[0],  # 起始帧
            "end_frame": frames[-1],  # 结束帧
            "timestamp_list": timestamp_list,  # 时间戳列表
        }

    return ply_dict  # 返回组装好的ply字典
