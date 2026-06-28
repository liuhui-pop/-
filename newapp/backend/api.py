"""
api.py - FastAPI 后端入口
HSI 高光谱图像分类系统 API

启动方式:
    pip install fastapi uvicorn python-multipart
    uvicorn api:app --reload --port 8000

前端访问地址: http://localhost:8000
API 文档:     http://localhost:8000/docs
"""

import os
import io
import json
import uuid
import time
import base64
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")  # 非交互模式，服务器环境必须
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import asyncio
import queue
import threading
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# ── 导入你的核心模块 ──────────────────────────────────────────
from workflow import train_by_param
from utils import recorder
from gt_registry import (find_gt_for_file, list_supported_datasets,
                          match_dataset, detect_dataset_from_mat)
from trainers import MODEL_REGISTRY, register_user_model, USER_MODEL_DIR
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="HSI Classification API", version="1.0.0")

# 允许前端跨域访问（Next.js 默认跑在 3000 端口）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 目录配置 ──────────────────────────────────────────────────
UPLOAD_DIR = Path("./uploads")          # 上传的数据文件
RESULTS_DIR = Path("./res_base/api")    # 训练结果
MODEL_DIR = Path("./save_models")       # 保存的模型
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
_experiment_records: list = []

def load_experiment_log() -> list:
    return _experiment_records

def save_experiment_log(records: list):
    global _experiment_records
    _experiment_records = records

def append_experiment_record(record: dict):
    global _experiment_records
    _experiment_records.insert(0, record)


# ── 内存中的任务状态表 ────────────────────────────────────────
# { job_id: { "status": "running"|"done"|"failed", "result": {...}, "error": "...",
#             "stream": Queue } }
job_store: dict = {}

# 全局流数据注册表：job_id -> Queue
# BaseTrainer 通过 push_stream_event() 写入，SSE 接口读取
stream_registry: dict[str, queue.Queue] = {}

def push_stream_event(job_id: str, event: dict):
    """供 BaseTrainer 调用，把每个 epoch 数据推入队列。"""
    if job_id in stream_registry:
        stream_registry[job_id].put(event)


# ═════════════════════════════════════════════════════════════
# 数据模型
# ═════════════════════════════════════════════════════════════

class TrainRequest(BaseModel):
    config_name: Optional[str] = None   # 不传时自动根据 dataset+model 选择
    model_value: Optional[str] = None   # 模型 value，如 "simple_cnn"
    dataset_sign: Optional[str] = None  # 数据集，如 "Pavia"、"Indian"
    train_sign: str = "train"
    noise_type: str = "clean"
    uploaded_file_id: Optional[str] = None
    lr: Optional[float] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None


class SplitRequest(BaseModel):
    file_id: str
    split_mode: str = "per_class"   # "per_class" | "percent"
    split_value: int = 10           # 每类数量 or 训练百分比


class JobStatusResponse(BaseModel):
    job_id: str
    status: str           # "running" | "done" | "failed"
    message: str
    result: Optional[dict] = None
    error: Optional[str] = None


# ═════════════════════════════════════════════════════════════
# 接口 1：POST /upload  上传数据文件
# ═════════════════════════════════════════════════════════════

@app.post("/upload", summary="上传 .mat 或 .csv 数据文件")
async def upload_file(file: UploadFile = File(...)):
    """
    前端 Import 按钮触发。
    接收 .mat / .csv / .npy 文件，保存到 ./uploads/ 并返回文件 ID。
    """
    allowed_exts = {".mat", ".csv", ".npy", ".tif", ".tiff"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式 {suffix}，支持: {allowed_exts}"
        )

    file_id = str(uuid.uuid4())[:8]
    save_name = f"{file_id}_{file.filename}"
    save_path = UPLOAD_DIR / save_name

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_size_mb = round(save_path.stat().st_size / 1024 / 1024, 2)

    return {
        "file_id": file_id,
        "file_name": file.filename,
        "saved_as": save_name,
        "size_mb": file_size_mb,
        "message": "上传成功"
    }


# ═════════════════════════════════════════════════════════════
# 接口 2：POST /train  发起训练任务（异步后台运行）
# ═════════════════════════════════════════════════════════════

def _run_train(job_id: str, param: dict, noise_type: str):
    """后台线程中执行训练，结果写入 job_store，实时数据写入 stream_registry。"""
    try:
        job_store[job_id]["status"] = "running"

        # 创建该任务的流队列
        stream_registry[job_id] = queue.Queue()

        # 把 job_id 和推送函数注入 param，BaseTrainer 可以调用
        param["_job_id"] = job_id
        param["_push_event"] = push_stream_event

        # 覆盖超参数（前端传入时生效）
        if "lr" in param and param["lr"] is not None:
            param["train"]["lr"] = param["lr"]
        if "epochs" in param and param["epochs"] is not None:
            param["train"]["epochs"] = param["epochs"]
        if "batch_size" in param and param["batch_size"] is not None:
            param["data"]["batch_size"] = param["batch_size"]

        # 构造结果路径
        time_stamp = datetime.now().strftime("%m%d%H%M")
        uniq_name = param.get("uniq_name", "job")
        result_name = f"{uniq_name}_{job_id}_{time_stamp}"
        param["path_res"] = str(RESULTS_DIR / result_name)
        param["path_pic"] = str(RESULTS_DIR / f"{result_name}.png")
        param["path_model_save"]  = str(MODEL_DIR / f"{uniq_name}.pth")   # 勿删！模型保存路径
        param["data"]["noise_type"] = noise_type

        # 调用训练函数
        rec = train_by_param(param, aug=0)

        # 提取评估结果
        eval_data = rec.record_data.get("eval", {})
        result = {
            "oa": eval_data.get("oa"),
            "aa": eval_data.get("aa"),
            "kappa": eval_data.get("kappa"),
            "each_acc": eval_data.get("each_acc"),
            "eval_time": rec.record_data.get("eval_time"),
            "result_path": param["path_res"],
        }
        job_store[job_id].update({"status": "done", "result": result})
        oa    = round(float(eval_data.get("oa")    or 0), 4)
        aa    = round(float(eval_data.get("aa")    or 0), 4)
        kappa = round(float(eval_data.get("kappa") or 0), 4)
        # 推送训练完成事件（含最终指标）
        push_stream_event(job_id, {
            "type": "done", "oa": oa, "aa": aa, "kappa": kappa,
        })
        append_experiment_record({
            "job_id":     job_id,
            "time":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset":    param.get("data", {}).get("data_sign", "unknown"),
            "model":      param.get("net", {}).get("trainer", "unknown"),
            "train_sign": param.get("train_sign", "train"),
            "noise_type": param.get("data", {}).get("noise_type", "clean"),
            "epochs":     param.get("train", {}).get("epochs", 0),
            "oa":         oa,
            "aa":         aa,
            "kappa":      kappa,
            "eval_time":  round(float(rec.record_data.get("eval_time") or 0), 2),
            "config":     param.get("_config_name", ""),
        })

    except Exception as e:
        job_store[job_id].update({"status": "failed", "error": traceback.format_exc()})
        if job_id in stream_registry:
            push_stream_event(job_id, {"type": "error", "message": str(e)})


@app.post("/train", summary="发起模型训练（异步）")
async def train(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    前端 Train 按钮触发。
    立即返回 job_id，训练在后台进行。
    前端可轮询 GET /job/{job_id} 查看进度。
    """
    # 自动选择配置文件：优先用前端指定的，否则根据 dataset+model 自动选
    try:
        config_name = req.config_name or _resolve_config(req.dataset_sign, req.model_value or "")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    config_path = Path("./params_use") / config_name
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"配置文件不存在: {config_name}，请先创建对应配置文件")

    with open(config_path, "r") as f:
        param = json.load(f)

    param["train_sign"] = req.train_sign

    # 超参数覆盖
    if req.lr is not None:
        param["lr"] = req.lr
    if req.epochs is not None:
        param["epochs"] = req.epochs
    if req.batch_size is not None:
        param["batch_size"] = req.batch_size

    # 数据集信息覆盖（当自动识别到数据集时）
    if req.dataset_sign is not None:
        param["data"]["data_sign"] = req.dataset_sign
    if req.dataset_sign in ["Indian", "Salinas"]:
        from gt_registry import GT_REGISTRY
        ds_key = req.dataset_sign.lower()
        if ds_key in GT_REGISTRY:
            info = GT_REGISTRY[ds_key]
            param["data"]["num_classes"] = info["classes"]
            param["data"]["spectral_size"] = info["spectral"]

    # 若前端指定了上传文件，把路径注入 param
    if req.uploaded_file_id:
        matches = list(UPLOAD_DIR.glob(f"{req.uploaded_file_id}_*"))
        if matches:
            uploaded_path = str(matches[0])
            param["data"]["uploaded_file_path"] = uploaded_path

    job_id = str(uuid.uuid4())[:8]
    job_store[job_id] = {"status": "pending", "result": None, "error": None}
    stream_registry[job_id] = queue.Queue()  # 提前初始化，防止 SSE 连接比训练启动早

    background_tasks.add_task(_run_train, job_id, param, req.noise_type)

    return {
        "job_id": job_id,
        "status": "pending",
        "message": f"训练任务已提交，使用配置: {req.config_name}"
    }


# ═════════════════════════════════════════════════════════════
# 接口 3：GET /job/{job_id}  查询任务状态
# ═════════════════════════════════════════════════════════════

@app.get("/job/{job_id}", response_model=JobStatusResponse, summary="查询训练任务状态")
async def get_job(job_id: str):
    """
    前端轮询此接口（每 3 秒一次）判断训练是否完成。
    status: pending → running → done / failed
    """
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="任务不存在")

    job = job_store[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        message={
            "pending": "任务排队中...",
            "running": "训练进行中，请稍候...",
            "done": "训练完成！",
            "failed": "训练失败，请查看 error 字段",
        }.get(job["status"], ""),
        result=job.get("result"),
        error=job.get("error"),
    )


# ── 工具函数：从结果 JSON 读取完整训练数据 ───────────────────
def _load_result_json(result: dict) -> dict:
    result_path = result.get("result_path", "")
    # 兼容有无 .json 后缀
    json_path = result_path if result_path.endswith(".json") else result_path + ".json"
    if json_path and os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _parse_index_value(data: dict, key: str):
    """从 recorder 的 index_value 格式解析成 [{x, y}] 列表。"""
    obj = data.get(key, {})
    if not obj or obj.get("type") != "index_value":
        return []
    return [
        {"x": idx, "y": round(float(v), 6)}
        for idx, v in zip(obj["index"], obj["value"])
    ]


def _parse_each_acc(raw: str) -> list[float]:
    try:
        return [round(float(x), 4) for x in raw.strip("[]").split()]
    except Exception:
        return []


# ═════════════════════════════════════════════════════════════
# 接口 4：GET /evaluate/{job_id}  获取评估指标
# ═════════════════════════════════════════════════════════════

@app.get("/evaluate/{job_id}", summary="获取训练完成后的评估指标")
async def evaluate(job_id: str):
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="任务不存在")
    job = job_store[job_id]
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail=f"任务尚未完成: {job['status']}")

    saved = _load_result_json(job["result"])
    eval_data = saved.get("eval", {})
    each_acc = _parse_each_acc(eval_data.get("each_acc", "[]"))

    return {
        "job_id":           job_id,
        "oa":               round(float(eval_data.get("oa") or 0), 4),
        "aa":               round(float(eval_data.get("aa") or 0), 4),
        "kappa":            round(float(eval_data.get("kappa") or 0), 4),
        "each_acc":         each_acc,
        "eval_time_seconds": round(float(saved.get("eval_time") or 0), 3),
    }


# ═════════════════════════════════════════════════════════════
# 接口 5：GET /visualize/{job_id}  完整可视化数据
# ═════════════════════════════════════════════════════════════

@app.get("/visualize/{job_id}", summary="获取完整可视化数据（曲线+指标+分类图）")
async def visualize(job_id: str):
    """
    返回：
    - loss_curve:    epoch_loss 曲线 [{x, y}]
    - oa_curve:      训练过程 OA 曲线 [{x, y}]
    - metrics_bar:   最终 OA/AA/Kappa [{name, value}]
    - class_acc:     逐类精度 [{class, accuracy}]
    - pred_image:    分类结果图 base64（若有 _pred.npy）
    - confusion_matrix_text: 混淆矩阵文本
    """
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="任务不存在")
    job = job_store[job_id]
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail=f"任务尚未完成: {job['status']}")

    saved   = _load_result_json(job["result"])
    eval_data = saved.get("eval", {})
    each_acc  = _parse_each_acc(eval_data.get("each_acc", "[]"))

    # ── 曲线数据 ──────────────────────────────────────────────
    loss_curve = _parse_index_value(saved, "epoch_loss")
    oa_curve   = _parse_index_value(saved, "train_oa")

    # ── 指标柱状图 ────────────────────────────────────────────
    metrics_bar = [
        {"name": "OA",    "value": round(float(eval_data.get("oa")    or 0), 4)},
        {"name": "AA",    "value": round(float(eval_data.get("aa")    or 0), 4)},
        {"name": "Kappa", "value": round(float(eval_data.get("kappa") or 0), 4)},
    ]

    # ── 逐类精度 ──────────────────────────────────────────────
    class_names = saved.get("param", {}).get("data", {}).get("data_sign", "")
    pavia_names = ["Asphalt","Meadows","Gravel","Trees",
                   "Metal Sheets","Bare Soil","Bitumen","Bricks","Shadows"]
    names = pavia_names if "Pavia" in class_names else [f"Class {i+1}" for i in range(len(each_acc))]
    class_acc = [
        {"class": names[i] if i < len(names) else f"Class {i+1}", "accuracy": v}
        for i, v in enumerate(each_acc)
    ]

    # ── 分类结果图：优先读 jpg，否则从 npy 生成 ──────────────
    pred_image = None
    result_path = job["result"].get("result_path", "")
    from pathlib import Path as _Path

    _base = _Path(result_path)
    pred_jpg = str(_base.parent / (_base.name + "_pred.jpg"))
    pred_npy = str(_base.parent / (_base.name + "_pred.npy"))
    print(f"[visualize] result_path={result_path}")
    print(f"[visualize] pred_jpg={pred_jpg}, exists={os.path.exists(pred_jpg)}")
    if pred_jpg and os.path.exists(pred_jpg):
        # 直接读取已生成的彩色 jpg
        with open(pred_jpg, "rb") as f:
            pred_image = base64.b64encode(f.read()).decode("utf-8")
    elif pred_npy and os.path.exists(pred_npy):
        # 回退：从 npy 用 matplotlib 生成
        pred = np.load(pred_npy)
        n_classes = int(pred.max())
        cmap = plt.get_cmap("tab20", n_classes + 1)
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#1a1a1a")
        im = ax.imshow(pred, cmap=cmap, vmin=0, vmax=n_classes)
        ax.set_title("Prediction Map", color="white", fontsize=10)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(colors="#9ca3af", labelsize=7)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#1a1a1a", dpi=120)
        plt.close(fig)
        buf.seek(0)
        pred_image = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "job_id":                job_id,
        "loss_curve":            loss_curve,
        "oa_curve":              oa_curve,
        "metrics_bar":           metrics_bar,
        "class_acc":             class_acc,
        "pred_image":            pred_image,
        "confusion_matrix_text": eval_data.get("confusion", ""),
    }


# ═════════════════════════════════════════════════════════════
# 接口 6：GET /export/{job_id}  导出模型/结果
# ═════════════════════════════════════════════════════════════

@app.get("/export/{job_id}", summary="下载训练结果 JSON 文件")
async def export(job_id: str):
    """
    前端 Export 按钮触发。
    返回训练结果 JSON 文件供用户下载。
    后续可扩展为下载 .pth 模型权重文件。
    """
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="任务不存在")

    job = job_store[job_id]
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="任务尚未完成")

    result_path = job["result"].get("result_path")
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="结果文件不存在")

    return FileResponse(
        path=result_path,
        media_type="application/json",
        filename=f"result_{job_id}.json"
    )


# ═════════════════════════════════════════════════════════════
# 接口 7：GET /configs  获取可用配置文件列表
# ═════════════════════════════════════════════════════════════

@app.get("/configs", summary="获取 params_use 目录下所有可用配置")
async def list_configs():
    config_dir = Path("./params_use")
    if not config_dir.exists():
        return {"configs": []}
    configs = [f.name for f in config_dir.glob("*.json")]
    return {"configs": sorted(configs)}


# 数据集前缀映射
DATASET_PREFIX = {
    "Pavia":   "pavia",
    "Indian":  "indian",
    "Salinas": "salinas",
    "Houston": "houston",
}

# 模型配置文件名映射
MODEL_CONFIG_NAME = {
    "transformer_DELTA": "transformer_delta",
    "simple_cnn":        "simple_cnn",
    "cnn_attention":     "cnn_attention",
    "vit":               "vit",
}


@app.get("/detect_dataset/{file_id}", summary="根据文件名或内容自动识别数据集")
async def detect_dataset(file_id: str):
    """
    上传文件后调用，自动识别数据集类型。
    先用文件名匹配，失败则读取文件内容用尺寸匹配。
    支持原始 mat 和 split.mat 两种格式。
    """
    matches = list(UPLOAD_DIR.glob(f"{file_id}_*"))
    if not matches:
        raise HTTPException(status_code=404, detail="文件不存在")

    file_path = matches[0]

    # 优先用文件名匹配
    _, info = find_gt_for_file(file_path.name)

    # 文件名匹配失败，用内容尺寸匹配
    if info is None and file_path.suffix.lower() == ".mat":
        info = detect_dataset_from_mat(str(file_path))

    if info is None:
        return {
            "detected":     False,
            "file_id":      file_id,
            "file_name":    file_path.name,
            "dataset_sign": None,
            "num_classes":  None,
            "spectral":     None,
            "is_split":     False,
            "message":      "未能自动识别数据集，请确认文件名包含数据集关键词（如 PaviaU、Indian 等）",
        }

    # 判断是否为 split.mat（含 TR/TE 字段）
    is_split = False
    try:
        mat = sio.loadmat(str(file_path))
        is_split = "TR" in mat and "TE" in mat
    except Exception:
        pass

    return {
        "detected":     True,
        "file_id":      file_id,
        "file_name":    file_path.name,
        "dataset_sign": info["sign"],
        "num_classes":  info["classes"],
        "spectral":     info["spectral"],
        "is_split":     is_split,
        "message":      (
            f"识别为 {info['sign']}（{info['classes']} 类，{info['spectral']} 波段）"
            + ("，已包含 TR/TE 划分，可直接训练" if is_split else "，需要先执行数据划分")
        ),
    }


def _resolve_config(dataset_sign: str | None, model_value: str) -> str:
    """
    根据数据集和模型自动选择配置文件名。
    优先使用 {dataset}_{model}.json，不存在时报清晰错误。
    """
    config_dir     = Path("./params_use")
    dataset_prefix = DATASET_PREFIX.get(dataset_sign or "", "pavia")
    model_suffix   = MODEL_CONFIG_NAME.get(model_value, model_value.lower())
    config_name    = f"{dataset_prefix}_{model_suffix}.json"

    if (config_dir / config_name).exists():
        return config_name

    # 列出同模型的所有可用配置，给出有用的提示
    available = [
        f.name for f in config_dir.glob(f"*_{model_suffix}.json")
    ]
    hint = f"可用的同模型配置：{available}" if available else "params_use/ 目录下暂无该模型的配置文件"
    raise FileNotFoundError(
        f"配置文件 {config_name} 不存在。"
        f"数据集={dataset_sign}，模型={model_value}。"
        f"{hint}。请在 params_use/ 目录下创建对应配置文件。"
    )


# ═════════════════════════════════════════════════════════════
# 接口 8：GET /preview/{file_id}  mat 文件可视化
# ═════════════════════════════════════════════════════════════

def _mat_to_base64(fig: plt.Figure) -> str:
    """将 matplotlib Figure 转为 base64 字符串。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="#1a1a1a", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _find_hsi_array(mat: dict) -> np.ndarray | None:
    """从 mat 文件中找到最大的三维 HSI 数组。"""
    best = None
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 3:
            if best is None or v.size > best.size:
                best = v
    return best


def _find_gt_array(mat: dict) -> np.ndarray | None:
    """从 mat 文件中找到二维 GT 标签数组（必须是整数类型）。"""
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if (isinstance(v, np.ndarray)
                and v.ndim == 2
                and np.issubdtype(v.dtype, np.integer)):
            return v
    return None


def _render_rgb(data: np.ndarray,
                r_idx: int = -1,
                g_idx: int = -1,
                b_idx: int = -1) -> str:
    """HSI 伪彩色图，支持自定义波段，-1 表示自动选择。"""
    h, w, c = data.shape
    if r_idx < 0 or r_idx >= c:
        r_idx = min(int(c * 0.7), c - 1)
    if g_idx < 0 or g_idx >= c:
        g_idx = min(int(c * 0.4), c - 1)
    if b_idx < 0 or b_idx >= c:
        b_idx = min(int(c * 0.1), c - 1)

    rgb = np.stack([data[:, :, r_idx],
                    data[:, :, g_idx],
                    data[:, :, b_idx]], axis=-1).astype(np.float32)

    for i in range(3):
        p2, p98 = np.percentile(rgb[:, :, i], [2, 98])
        if p98 > p2:
            rgb[:, :, i] = np.clip((rgb[:, :, i] - p2) / (p98 - p2), 0, 1)

    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#1a1a1a")
    ax.imshow(rgb)
    ax.set_title(f"HSI 伪彩色图  R:{r_idx} G:{g_idx} B:{b_idx}", color="white", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    return _mat_to_base64(fig)


def _render_spectral_curve(data: np.ndarray) -> str:
    """
    光谱曲线图。
    随机采样 5 个像素，绘制其全波段响应曲线。
    """
    h, w, c = data.shape
    rng = np.random.default_rng(42)
    rows = rng.integers(0, h, 5)
    cols = rng.integers(0, w, 5)

    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#1a1a1a")
    ax.set_facecolor("#1a1a1a")
    colors = ["#3b82f6", "#22c55e", "#eab308", "#a855f7", "#ef4444"]
    for i, (r, c_) in enumerate(zip(rows, cols)):
        spectrum = data[r, c_, :].astype(np.float32)
        ax.plot(spectrum, color=colors[i], linewidth=1.5,
                label=f"像素 ({r},{c_})")

    ax.set_title("光谱曲线图", color="white", fontsize=10)
    ax.set_xlabel("波段编号", color="#9ca3af", fontsize=8)
    ax.set_ylabel("反射率", color="#9ca3af", fontsize=8)
    ax.tick_params(colors="#9ca3af", labelsize=7)
    ax.legend(fontsize=7, labelcolor="white",
              facecolor="#2a2a2a", edgecolor="#444")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    fig.tight_layout()
    return _mat_to_base64(fig)


def _render_gt(gt: np.ndarray) -> str:
    """
    GT 标签热力图（分类结果图）。
    """
    n_classes = int(gt.max())
    cmap = plt.get_cmap("tab20", n_classes + 1)

    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#1a1a1a")
    im = ax.imshow(gt, cmap=cmap, vmin=0, vmax=n_classes)
    ax.set_title("GT 标签图", color="white", fontsize=10)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors="#9ca3af", labelsize=7)
    fig.tight_layout()
    return _mat_to_base64(fig)


@app.get("/preview/{file_id}", summary="解析 mat 文件并返回三张可视化图片")
async def preview_mat(
    file_id: str,
    r_band: int = -1,
    g_band: int = -1,
    b_band: int = -1,
):
    """支持自定义 R/G/B 波段的 mat 文件可视化接口。"""
    matches = list(UPLOAD_DIR.glob(f"{file_id}_*"))
    if not matches:
        raise HTTPException(status_code=404, detail="文件不存在，请重新上传")

    file_path = matches[0]
    if file_path.suffix.lower() != ".mat":
        raise HTTPException(status_code=400, detail="当前仅支持 .mat 文件预览")

    try:
        mat = sio.loadmat(str(file_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"mat 文件读取失败: {e}")

    hsi = _find_hsi_array(mat)
    gt  = _find_gt_array(mat)

    if hsi is None:
        raise HTTPException(status_code=400, detail="未找到三维 HSI 数组")

    h, w, c = hsi.shape
    actual_r = r_band if 0 <= r_band < c else min(int(c * 0.7), c - 1)
    actual_g = g_band if 0 <= g_band < c else min(int(c * 0.4), c - 1)
    actual_b = b_band if 0 <= b_band < c else min(int(c * 0.1), c - 1)

    return {
        "file_id": file_id,
        "meta": {
            "height": h, "width": w, "bands": c,
            "n_classes": int(gt.max()) if gt is not None else None,
            "file_name": file_path.name,
            "r_band": actual_r, "g_band": actual_g, "b_band": actual_b,
        },
        "rgb_image":      _render_rgb(hsi, r_band, g_band, b_band),
        "spectral_image": _render_spectral_curve(hsi),
        "gt_image":       _render_gt(gt) if gt is not None else None,
    }


# ═════════════════════════════════════════════════════════════
# 接口 9：POST /split  数据划分
# ═════════════════════════════════════════════════════════════

@app.post("/split", summary="对原始 mat 文件进行训练/测试划分，生成 split.mat")
async def split_data(req: SplitRequest):
    """
    前端「拆分」按钮触发。
    自动从内置 GT 注册表匹配 GT 文件，无需用户单独上传。
    支持两种模式：
    - per_class: 每类随机选 N 个样本作为训练集
    - percent:   按百分比划分（训练集占 split_value%）
    """
    matches = list(UPLOAD_DIR.glob(f"{req.file_id}_*"))
    if not matches:
        raise HTTPException(status_code=404, detail="文件不存在，请重新上传")

    file_path = matches[0]
    if file_path.suffix.lower() != ".mat":
        raise HTTPException(status_code=400, detail="仅支持 .mat 文件划分")

    try:
        mat = sio.loadmat(str(file_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"mat 文件读取失败: {e}")

    # 如果已经是 split.mat（含 input/TR/TE），直接返回
    if "input" in mat and "TR" in mat and "TE" in mat:
        return {
            "split_file_id":   req.file_id,
            "split_file_name": file_path.name,
            "size_mb":         round(file_path.stat().st_size / 1024 / 1024, 2),
            "message":         "检测到已划分的 split.mat，直接使用",
        }

    # 读取 HSI 数据
    hsi = _find_hsi_array(mat)
    if hsi is None:
        raise HTTPException(status_code=400, detail="未找到三维 HSI 数组")

    # 从内置 GT 注册表自动匹配 GT
    gt, dataset_info = find_gt_for_file(file_path.name)
    if gt is None:
        supported = [info["keywords"][0] for info in list_supported_datasets() if info["gt_ready"]]
        raise HTTPException(
            status_code=400,
            detail=f"无法自动匹配 GT，请确认文件名包含数据集关键词。"
                   f"当前支持：{', '.join(supported)}。"
                   f"文件名示例：PaviaU.mat、Indian_pines_corrected.mat"
        )

    # 验证 HSI 和 GT 尺寸匹配
    h, w, c = hsi.shape
    if gt.shape != (h, w):
        raise HTTPException(
            status_code=400,
            detail=f"HSI 尺寸 {hsi.shape[:2]} 与 GT 尺寸 {gt.shape} 不匹配"
        )

    # 按类别收集像素坐标
    import random as _random
    class2pos: dict[int, list] = {}
    for i in range(h):
        for j in range(w):
            label = int(gt[i, j])
            if label > 0:
                class2pos.setdefault(label, []).append((i, j))

    TR = np.zeros_like(gt)
    TE = np.zeros_like(gt)

    for cls, pos_list in class2pos.items():
        n = len(pos_list)
        if req.split_mode == "per_class":
            train_n = min(req.split_value, int(n * 0.8))
        else:
            train_n = max(1, int(n * req.split_value / 100))

        selected = set(_random.sample(range(n), train_n))
        for idx, (i, j) in enumerate(pos_list):
            if idx in selected:
                TR[i, j] = cls
            else:
                TE[i, j] = cls

    # 保存 split.mat（格式与 data_provider 兼容）
    split_id   = str(uuid.uuid4())[:8]
    split_name = f"{split_id}_{file_path.stem}_split.mat"
    split_path = UPLOAD_DIR / split_name

    sio.savemat(str(split_path), {"input": hsi, "TR": TR, "TE": TE})

    # 同时保存一份到 data/{sign}/ 目录，供 data_provider 固定路径加载
 
    size_mb = round(split_path.stat().st_size / 1024 / 1024, 2)
    return {
        "split_file_id":   split_id,
        "split_file_name": split_name,
        "size_mb":         size_mb,
        "dataset":         dataset_info.get("sign", ""),
        "num_classes":     dataset_info.get("classes", 0),
        "train_samples":   int((TR > 0).sum()),
        "test_samples":    int((TE > 0).sum()),
        "message":         (
            f"划分完成 [{dataset_info.get('sign','')}]："
            f"训练 {int((TR>0).sum())} 样本，测试 {int((TE>0).sum())} 样本"
        ),
    }


@app.get("/datasets", summary="获取内置支持的数据集列表")
async def get_datasets():
    """前端可用此接口展示支持哪些数据集。"""
    return {"datasets": list_supported_datasets()}

# ═════════════════════════════════════════════════════════════
# 接口：GET /models  获取所有可用模型列表（含用户上传的）
# ═════════════════════════════════════════════════════════════
 
@app.get("/models", summary="获取所有可用模型列表")
async def list_models():
    """前端下拉框动态获取模型列表。"""
    models = []
    for trainer_type, info in MODEL_REGISTRY.items():
        models.append({
            "value": trainer_type,
            "label": info.get("label", trainer_type),
            "lr":    info.get("lr", 0.001),
            "optimizer": info.get("optimizer", "adam"),
            "is_user": info.get("is_user", False),
        })
    return {"models": models}
 
 
# ═════════════════════════════════════════════════════════════
# 接口：POST /upload_model  上传用户模型
# ═════════════════════════════════════════════════════════════
 
class ModelUploadMeta(BaseModel):
    trainer_type: str   # 注册名，如 "my_model"
    class_name:   str   # 模型类名，如 "MyModel"
    label:        str   # 显示名称，如 "我的模型"
    lr:           float = 0.001
    optimizer:    str   = "adam"
 
 
@app.post("/upload_model", summary="上传用户自定义模型")
async def upload_model(
    file:         UploadFile = File(...),
    trainer_type: str = "",
    class_name:   str = "",
    label:        str = "",
    lr:           float = 0.001,
    optimizer:    str = "adam",
):
    """
    用户上传 .py 模型文件，系统动态注册。
    模型类的 forward 方法可以返回任意格式，系统会自动适配。
    """
    if not file.filename.endswith(".py"):
        raise HTTPException(status_code=400, detail="只支持上传 .py 文件")
 
    if not trainer_type:
        trainer_type = Path(file.filename).stem.lower().replace(" ", "_")
    if not class_name:
        class_name = Path(file.filename).stem
    if not label:
        label = class_name
 
    # 保存 py 文件
    save_path = USER_MODEL_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
 
    # 保存配置 json（下次启动自动扫描用）
    import json as _json
    cfg_path = save_path.with_suffix(".json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        _json.dump({
            "trainer_type": trainer_type,
            "class_name":   class_name,
            "label":        label,
            "lr":           lr,
            "optimizer":    optimizer,
        }, f, ensure_ascii=False, indent=2)
 
    # 动态注册
    ok = register_user_model(str(save_path), trainer_type, class_name, lr, optimizer)
    if ok:
        MODEL_REGISTRY[trainer_type]["label"]    = label
        MODEL_REGISTRY[trainer_type]["is_user"]  = True
        return {
            "trainer_type": trainer_type,
            "class_name":   class_name,
            "label":        label,
            "message":      f"模型 {label} 注册成功，已加入下拉列表",
        }
    else:
        raise HTTPException(status_code=500,
                            detail=f"模型注册失败，请检查 {class_name} 类是否正确定义")



# ═════════════════════════════════════════════════════════════
# 接口：POST /convert  格式转换（tif → mat）
# ═════════════════════════════════════════════════════════════

class ConvertRequest(BaseModel):
    file_id: str
    format: str = "tif_to_mat"   # 目前只支持 tif_to_mat


@app.post("/convert", summary="格式转换（tif → mat）")
async def convert_file(req: ConvertRequest):
    """
    前端「格式转换」→「应用」触发。
    将 .tif/.tiff 文件转换为 .mat 格式，存入 uploads 目录。
    返回新文件的 file_id，前端加入侧边栏。
    """
    if req.format != "tif_to_mat":
        raise HTTPException(status_code=400, detail=f"暂不支持的转换格式: {req.format}")

    matches = list(UPLOAD_DIR.glob(f"{req.file_id}_*"))
    if not matches:
        raise HTTPException(status_code=404, detail="文件不存在，请重新上传")

    file_path = matches[0]
    if file_path.suffix.lower() not in (".tif", ".tiff"):
        raise HTTPException(status_code=400,
                            detail=f"当前仅支持 .tif/.tiff 转换，收到: {file_path.suffix}")

    try:
        import tifffile
        data = tifffile.imread(str(file_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"tif 文件读取失败: {e}")

    # tif 格式处理：
    # 如果是 (bands, H, W) → 转为 (H, W, bands)
    # 如果是 (H, W) → 单波段，扩展为 (H, W, 1)
    # 如果是 (H, W, bands) → 直接使用
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    elif data.ndim == 3:
        if data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
            # (bands, H, W) → (H, W, bands)
            data = np.transpose(data, (1, 2, 0))
    else:
        raise HTTPException(status_code=400,
                            detail=f"不支持的 tif 维度: {data.shape}，期望 2D 或 3D 数组")

    data = data.astype(np.float32)
    h, w, bands = data.shape

    # 保存为 mat
    new_id   = str(uuid.uuid4())[:8]
    new_name = f"{new_id}_{file_path.stem}.mat"
    new_path = UPLOAD_DIR / new_name

    sio.savemat(str(new_path), {"input": data})

    size_mb = round(new_path.stat().st_size / 1024 / 1024, 2)
    return {
        "file_id":   new_id,
        "file_name": new_name,
        "size_mb":   size_mb,
        "shape":     {"height": h, "width": w, "bands": bands},
        "message":   f"转换成功：{file_path.name} → {new_name}（{h}×{w}，{bands} 波段）",
    }


# ═════════════════════════════════════════════════════════════
# 接口：GET /download/{file_id}  下载上传目录中的任意文件
# ═════════════════════════════════════════════════════════════

@app.get("/download/{file_id}", summary="下载指定文件到本地")
async def download_file(file_id: str):
    """
    前端右键菜单「保存到本地」触发。
    支持下载 uploads/ 目录下的任意文件（mat、split.mat 等）。
    """
    matches = list(UPLOAD_DIR.glob(f"{file_id}_*"))
    if not matches:
        raise HTTPException(status_code=404, detail="文件不存在")

    file_path = matches[0]
    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream",
    )


# ═════════════════════════════════════════════════════════════
# 接口 10（原9）：GET /stream/{job_id}  SSE 实时训练流
# ═════════════════════════════════════════════════════════════

@app.get("/stream/{job_id}", summary="SSE 实时推送训练进度")
async def stream_train(job_id: str):
    """
    前端用 EventSource 连接此接口，实时接收每个 epoch 的数据。
    推送格式（JSON）：
      {"type":"epoch", "epoch":10, "loss":0.05, "oa":0.82}
      {"type":"done",  "oa":0.92, "aa":0.90, "kappa":0.89}
      {"type":"error", "message":"..."}
    """
    if job_id not in stream_registry:
        raise HTTPException(status_code=404, detail="任务不存在")

    async def event_generator():
        q = stream_registry[job_id]
        while True:
            try:
                # 非阻塞取，让 asyncio 可以切换
                event = q.get_nowait()
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                # 队列暂时为空，等待
                await asyncio.sleep(0.3)
                # 如果任务已失败且队列空，退出
                status = job_store.get(job_id, {}).get("status")
                if status == "failed":
                    yield f"data: {json.dumps({'type':'error','message':'训练失败'})}\n\n"
                    break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


# ═════════════════════════════════════════════════════════════
# 健康检查
# ═════════════════════════════════════════════════════════════
@app.get("/experiments", summary="获取所有实验记录")
async def get_experiments():
    return {"records": load_experiment_log()}


@app.delete("/experiments/{job_id}", summary="删除单条实验记录")
async def delete_experiment(job_id: str):
    records = load_experiment_log()
    records = [r for r in records if r.get("job_id") != job_id]
    save_experiment_log(records)
    return {"message": "已删除"}


@app.delete("/experiments", summary="清空所有实验记录")
async def clear_experiments():
    save_experiment_log([])
    return {"message": "已清空"}
@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.now().isoformat()}


# ═════════════════════════════════════════════════════════════
# 批量训练
# ═════════════════════════════════════════════════════════════

class BatchTask(BaseModel):
    model_value: Optional[str] = None
    dataset_sign: Optional[str] = None
    config_name: Optional[str] = None
    train_sign: str = "train"
    noise_type: str = "clean"
    uploaded_file_id: Optional[str] = None
    lr: Optional[float] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    repeat: int = 1   # 重复训练次数
 
 
class BatchTrainRequest(BaseModel):
    tasks: list[BatchTask]
 
 
# 批量训练状态
batch_store: dict = {}
 
 
def _run_batch(batch_id: str, tasks: list[BatchTask]):
    """后台串行执行批量训练任务。"""
    total = sum(t.repeat for t in tasks)
    current = 0
    results = []
 
    batch_store[batch_id] = {
        "status": "running",
        "total": total,
        "current": 0,
        "results": [],
        "message": "批量训练开始...",
    }
 
    for task_idx, task in enumerate(tasks):
        for repeat_idx in range(task.repeat):
            current += 1
            job_id = str(uuid.uuid4())[:8]
 
            # 选配置文件
            try:
                config_name = task.config_name or _resolve_config(
                    task.dataset_sign, task.model_value or ""
                )
            except FileNotFoundError as e:
                results.append({
                    "job_id": job_id, "status": "failed",
                    "error": str(e), "task_idx": task_idx, "repeat": repeat_idx + 1,
                })
                batch_store[batch_id]["current"] = current
                batch_store[batch_id]["results"] = results
                continue
 
            config_path = Path("./params_use") / config_name
            if not config_path.exists():
                results.append({
                    "job_id": job_id, "status": "failed",
                    "error": f"配置文件不存在: {config_name}",
                    "task_idx": task_idx, "repeat": repeat_idx + 1,
                })
                batch_store[batch_id]["current"] = current
                batch_store[batch_id]["results"] = results
                continue
 
            with open(config_path, "r") as f:
                param = json.load(f)
 
            param["_config_name"] = config_name
            param["train_sign"]   = task.train_sign
            param["_job_id"]      = job_id
            param["_push_event"]  = None  # 批量训练不推送 SSE
 
            # 覆盖超参数
            if task.lr is not None:        param["lr"]         = task.lr
            if task.epochs is not None:    param["epochs"]     = task.epochs
            if task.batch_size is not None: param["batch_size"] = task.batch_size
            if task.dataset_sign:          param["data"]["data_sign"] = task.dataset_sign
            if task.noise_type:            param["data"]["noise_type"] = task.noise_type
 
            # 构造结果路径
            time_stamp  = datetime.now().strftime("%m%d%H%M")
            uniq_name   = param.get("uniq_name", "batch")
            result_name = f"{uniq_name}_{job_id}_{time_stamp}"
            param["path_res"]        = str(RESULTS_DIR / result_name)
            param["path_pic"]        = str(RESULTS_DIR / f"{result_name}.png")
            param["path_model_save"] = str(MODEL_DIR / f"{uniq_name}_{repeat_idx}.pth")
 
            # 上传文件
            if task.uploaded_file_id:
                matches = list(UPLOAD_DIR.glob(f"{task.uploaded_file_id}_*"))
                if matches:
                    param["data"]["uploaded_file_path"] = str(matches[0])
 
            batch_store[batch_id]["message"] = (
                f"[{current}/{total}] {config_name} 第{repeat_idx+1}次..."
            )
 
            try:
                rec = train_by_param(param, aug=0)
                eval_data = rec.record_data.get("eval", {})
                oa    = round(float(eval_data.get("oa")    or 0), 4)
                aa    = round(float(eval_data.get("aa")    or 0), 4)
                kappa = round(float(eval_data.get("kappa") or 0), 4)
 
                result = {
                    "job_id":    job_id,
                    "status":    "done",
                    "task_idx":  task_idx,
                    "repeat":    repeat_idx + 1,
                    "config":    config_name,
                    "oa":        oa,
                    "aa":        aa,
                    "kappa":     kappa,
                    "eval_time": round(float(rec.record_data.get("eval_time") or 0), 2),
                }
                results.append(result)
 
                # 写入实验记录
                append_experiment_record({
                    "job_id":     job_id,
                    "time":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "dataset":    param.get("data", {}).get("data_sign", "unknown"),
                    "model":      param.get("net", {}).get("trainer", "unknown"),
                    "train_sign": task.train_sign,
                    "noise_type": task.noise_type,
                    "epochs":     param.get("train", {}).get("epochs", 0),
                    "oa": oa, "aa": aa, "kappa": kappa,
                    "eval_time":  result["eval_time"],
                    "config":     config_name,
                })
 
            except Exception as e:
                results.append({
                    "job_id": job_id, "status": "failed",
                    "error": traceback.format_exc(),
                    "task_idx": task_idx, "repeat": repeat_idx + 1,
                })
 
            batch_store[batch_id]["current"] = current
            batch_store[batch_id]["results"] = results
 
    batch_store[batch_id]["status"]  = "done"
    batch_store[batch_id]["message"] = f"批量训练完成，共 {total} 个任务"
 
 
@app.post("/batch_train", summary="批量训练")
async def batch_train(req: BatchTrainRequest, background_tasks: BackgroundTasks):
    batch_id = str(uuid.uuid4())[:8]
    batch_store[batch_id] = {"status": "pending", "total": 0, "current": 0,
                              "results": [], "message": "准备中..."}
    background_tasks.add_task(_run_batch, batch_id, req.tasks)
    return {"batch_id": batch_id, "message": "批量训练已提交"}
 
 
@app.get("/batch_train/{batch_id}", summary="查询批量训练进度")
async def get_batch_status(batch_id: str):
    if batch_id not in batch_store:
        raise HTTPException(status_code=404, detail="批量任务不存在")
    return batch_store[batch_id]
