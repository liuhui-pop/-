"""
trainer.py
统一入口，get_trainer() 根据 trainer_type 返回对应 Trainer。

新增模型流程：
  1. models/ 下写模型类（forward 返回 (feat, logits)）
  2. trainers.py 的 _build_registry() 里加一行
  3. params_use/ 下加对应 json 配置文件
  4. lib/mock-data.ts 的 modelOptions 加一行
  
不需要再单独创建 Trainer 文件。
"""

from BNTentTrainerDELTA import TransformerTrainer_DELTA
from trainers import GenericTrainer


def get_trainer(params):
    trainer_type = params['net']['trainer']

    # ── 你的模型（有专属 Trainer）────────────────────────────
    if trainer_type == "transformer_DELTA":
        return TransformerTrainer_DELTA(params)

    # ── 所有其他模型走通用 Trainer ────────────────────────────
    return GenericTrainer(params)
