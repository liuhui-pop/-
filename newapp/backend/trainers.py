"""
trainers.py
所有模型的 Trainer 统一在这里注册。

新增模型只需两步：
  1. 在 models/ 下写模型类（forward 返回 (feat, logits)）
  2. 在下面的 MODEL_REGISTRY 里加一行

不需要再单独创建 Trainer 文件。
"""

import torch.nn as nn
import torch.optim as optim

from BNTentTrainerDELTA import BaseTrainer


# ═════════════════════════════════════════════════════════════
# 通用 Trainer：所有走标准 train/test 流程的模型都用这个
# ═════════════════════════════════════════════════════════════

class GenericTrainer(BaseTrainer):
    """
    通用 Trainer，从 MODEL_REGISTRY 里查找模型类并实例化。
    支持 Adam / AdamW / SGD 优化器。
    """

    def real_init(self):
        trainer_type = self.params['net']['trainer']
        entry = MODEL_REGISTRY.get(trainer_type)
        if entry is None:
            raise ValueError(f"模型 '{trainer_type}' 未在 MODEL_REGISTRY 中注册")

        # 实例化模型
        model_cls = entry["model"]
        self.net = model_cls(self.params).to(self.device)

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 优化器
        lr = self.train_params.get('lr', entry.get("lr", 0.001))
        wd = self.train_params.get('weight_decay', entry.get("weight_decay", 5e-3))
        opt_type = entry.get("optimizer", "adam")

        if opt_type == "adamw":
            self.optimizer = optim.AdamW(self.net.parameters(), lr=lr, weight_decay=wd)
        elif opt_type == "sgd":
            self.optimizer = optim.SGD(self.net.parameters(), lr=lr,
                                       momentum=0.9, weight_decay=wd)
        else:
            self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=wd)

    def get_loss(self, outputs, target):
        return nn.CrossEntropyLoss()(outputs, target)


# ═════════════════════════════════════════════════════════════
# 模型注册表
# 新增模型在这里加一行即可，格式：
#
# "trainer_type": {
#     "model":        模型类,
#     "lr":           默认学习率（可选，默认 0.001）,
#     "weight_decay": 默认权重衰减（可选，默认 5e-3）,
#     "optimizer":    "adam" | "adamw" | "sgd"（可选，默认 "adam"）,
# }
# ═════════════════════════════════════════════════════════════

def _build_registry():
    """延迟导入，避免循环依赖。"""
    from models import CNN
    from models import ViT
    # from models.CNN import CNNNet
    from models.SQSFormer import SQSFormer

    class SQSFormerWrapper(SQSFormer):
        def forward(self, x):
            logits, _, _ = super().forward(x)
            return logits, logits

    registry = {
        "simple_cnn": {
            "model":     CNN,
            "lr":        0.001,
            "optimizer": "adam",
        },
        # "cnn_attention": {
        #     "model":     CNNNet,
        #     "lr":        0.001,
        #     "optimizer": "adam",
        # },
        "vit": {
            "model":        ViT,
            "lr":           0.0005,
            "weight_decay": 1e-4,
            "optimizer":    "adamw",
        },
        "sqsformer": {
            "model":     SQSFormerWrapper,
            "lr":        0.001,
            "optimizer": "adam",
        },
    }
    return registry


# 全局注册表（首次访问时构建）
_registry_cache = None

def get_registry():
    global _registry_cache
    if _registry_cache is None:
        _registry_cache = _build_registry()
    return _registry_cache

MODEL_REGISTRY = get_registry()


# ── 用户模型目录 ───────────────────────────────────────────────
import importlib.util
import sys
from pathlib import Path

USER_MODEL_DIR = Path(__file__).parent / "models" / "user_models"
USER_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def register_user_model(py_path: str, trainer_type: str, class_name: str,
                         lr: float = 0.001, optimizer: str = "adam") -> bool:
    """
    动态加载用户上传的模型文件并注册到 MODEL_REGISTRY。
    返回 True 表示成功，False 表示失败。
    """
    try:
        spec   = importlib.util.spec_from_file_location(trainer_type, py_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[trainer_type] = module
        spec.loader.exec_module(module)

        model_cls = getattr(module, class_name)

        # 包装：确保 forward 返回 (feat, logits)
        class WrappedModel(model_cls):
            def forward(self, x):
                out = super().forward(x)
                if isinstance(out, tuple):
                    # 已经是 tuple，取前两个
                    feat   = out[0] if len(out) > 1 else out[0]
                    logits = out[1] if len(out) > 1 else out[0]
                    return feat, logits
                # 单输出，feat 和 logits 都用它
                return out, out

        MODEL_REGISTRY[trainer_type] = {
            "model":     WrappedModel,
            "lr":        lr,
            "optimizer": optimizer,
        }
        print(f"[trainers] 用户模型已注册: {trainer_type} ({class_name})")
        return True
    except Exception as e:
        import traceback
        print(f"[trainers] 注册失败: {e}")
        traceback.print_exc()
        return False


def scan_user_models():
    """启动时扫描 user_models 目录，自动注册已有模型。"""
    import json
    for py_file in USER_MODEL_DIR.glob("*.py"):
        cfg_file = py_file.with_suffix(".json")
        if cfg_file.exists():
            try:
                with open(cfg_file, encoding="utf-8") as f:
                    cfg = json.load(f)
                register_user_model(
                    str(py_file),
                    cfg.get("trainer_type", py_file.stem),
                    cfg.get("class_name", py_file.stem),
                    cfg.get("lr", 0.001),
                    cfg.get("optimizer", "adam"),
                )
            except Exception as e:
                print(f"[trainers] 扫描 {py_file.name} 失败: {e}")


# 启动时自动扫描
scan_user_models()
