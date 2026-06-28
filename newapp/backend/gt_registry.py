"""
gt_registry.py
内置数据集 GT 注册表。
支持两种识别方式：
  1. 文件名关键词匹配（原有）
  2. HSI 尺寸特征匹配（新增，用于识别 split.mat）
"""

import os
import numpy as np
import scipy.io as sio
from pathlib import Path

GT_DATA_DIR = Path(__file__).parent / "gt_data"

# ── 数据集注册表 ───────────────────────────────────────────────
GT_REGISTRY = {
    "pavia": {
        "keywords":  ["paviau", "pavia_u", "pavia"],
        "file":      "PaviaU_gt.mat",
        "key":       "paviaU_gt",
        "sign":      "Pavia",
        "classes":   9,
        "spectral":  103,
        "shape":     (610, 340),   # (h, w)
    },
    "indian": {
        "keywords":  ["indian", "indian_pines"],
        "file":      "Indian_pines_gt.mat",
        "key":       "indian_pines_gt",
        "sign":      "Indian",
        "classes":   16,
        "spectral":  200,
        "shape":     (145, 145),
    },
    "salinas": {
        "keywords":  ["salinas"],
        "file":      "Salinas_gt.mat",
        "key":       "salinas_gt",
        "sign":      "Salinas",
        "classes":   16,
        "spectral":  204,
        "shape":     (512, 217),
    },
    "houston": {
        "keywords":  ["houston", "houston2013"],
        "file":      "Houston2013_gt.mat",
        "key":       "gt",
        "sign":      "Houston",
        "classes":   15,
        "spectral":  144,
        "shape":     (349, 1905),
    },
}


def match_dataset(filename: str) -> dict | None:
    """根据文件名关键词匹配数据集。"""
    name_lower = Path(filename).stem.lower()
    for dataset_id, info in GT_REGISTRY.items():
        for kw in info["keywords"]:
            if kw in name_lower:
                return {**info, "dataset_id": dataset_id}
    return None


def match_dataset_by_shape(h: int, w: int, bands: int) -> dict | None:
    """
    根据 HSI 尺寸推断数据集（用于文件名无法识别的 split.mat）。
    允许 ±5 像素的误差。
    """
    for dataset_id, info in GT_REGISTRY.items():
        dh, dw = info["shape"]
        db = info["spectral"]
        if bands == db and abs(h - dh) <= 5 and abs(w - dw) <= 5:
            print(f"[gt_registry] 尺寸匹配: ({h},{w},{bands}) → {info['sign']}")
            return {**info, "dataset_id": dataset_id}
    return None


def detect_dataset_from_mat(mat_path: str) -> dict | None:
    """
    从 mat 文件内容自动识别数据集。
    先用文件名匹配，失败则用尺寸匹配。
    支持原始 mat 和 split.mat 两种格式。
    """
    file_path = Path(mat_path)

    # 1. 先尝试文件名匹配
    info = match_dataset(file_path.name)
    if info:
        return info

    # 2. 文件名匹配失败，读取内容用尺寸匹配
    try:
        mat = sio.loadmat(mat_path)
    except Exception as e:
        print(f"[gt_registry] 读取 mat 失败: {e}")
        return None

    # 找 HSI 数组（三维）
    hsi = None
    for key in ['input', 'data', 'HSI']:
        if key in mat and isinstance(mat[key], np.ndarray) and mat[key].ndim == 3:
            hsi = mat[key]
            break
    if hsi is None:
        for k, v in mat.items():
            if not k.startswith('__') and isinstance(v, np.ndarray) and v.ndim == 3:
                hsi = v
                break

    if hsi is None:
        print("[gt_registry] 未找到三维 HSI 数组")
        return None

    h, w, bands = hsi.shape
    print(f"[gt_registry] HSI 尺寸: ({h}, {w}, {bands})")
    return match_dataset_by_shape(h, w, bands)


def find_gt_for_file(filename: str) -> tuple:
    """根据文件名自动找到对应的内置 GT 数组。"""
    info = match_dataset(filename)
    if info is None:
        return None, None

    gt_path = GT_DATA_DIR / info["file"]
    if not gt_path.exists():
        print(f"[gt_registry] GT 文件不存在: {gt_path}")
        return None, None

    try:
        mat = sio.loadmat(str(gt_path))
        gt = mat.get(info["key"])
        if gt is None:
            for k, v in mat.items():
                if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 2:
                    gt = v
                    break
        if gt is not None:
            gt = gt.astype(np.int32)
        return gt, info
    except Exception as e:
        print(f"[gt_registry] 加载 GT 失败: {e}")
        return None, None


def list_supported_datasets() -> list[dict]:
    """返回所有已支持的数据集列表。"""
    result = []
    for dataset_id, info in GT_REGISTRY.items():
        gt_path = GT_DATA_DIR / info["file"]
        result.append({
            "id":       dataset_id,
            "sign":     info["sign"],
            "classes":  info["classes"],
            "spectral": info["spectral"],
            "keywords": info["keywords"],
            "shape":    info["shape"],
            "gt_ready": gt_path.exists(),
        })
    return result
