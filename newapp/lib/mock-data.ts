// ── HSI 数据集预览占位数据（后续由后端 /upload 接口替换）──────────────
// 每行代表一个像素样本，字段为部分光谱波段值 + 标签
export const mockDataset = [
  { id: 1, band_1: 0.312, band_2: 0.428, band_3: 0.519, band_4: 0.601, band_5: 0.487, label: "Asphalt",          class_id: 1 },
  { id: 2, band_1: 0.621, band_2: 0.703, band_3: 0.812, band_4: 0.744, band_5: 0.698, label: "Meadows",          class_id: 2 },
  { id: 3, band_1: 0.201, band_2: 0.334, band_3: 0.412, band_4: 0.389, band_5: 0.301, label: "Gravel",           class_id: 3 },
  { id: 4, band_1: 0.089, band_2: 0.142, band_3: 0.198, band_4: 0.221, band_5: 0.176, label: "Trees",            class_id: 4 },
  { id: 5, band_1: 0.731, band_2: 0.812, band_3: 0.891, band_4: 0.867, band_5: 0.802, label: "Metal Sheets",     class_id: 5 },
  { id: 6, band_1: 0.445, band_2: 0.512, band_3: 0.578, band_4: 0.534, band_5: 0.491, label: "Bare Soil",        class_id: 6 },
  { id: 7, band_1: 0.178, band_2: 0.245, band_3: 0.312, band_4: 0.289, band_5: 0.234, label: "Bitumen",          class_id: 7 },
  { id: 8, band_1: 0.534, band_2: 0.612, band_3: 0.689, band_4: 0.645, band_5: 0.598, label: "Bricks",           class_id: 8 },
  { id: 9, band_1: 0.023, band_2: 0.045, band_3: 0.067, band_4: 0.089, band_5: 0.056, label: "Shadows",          class_id: 9 },
  { id: 10, band_1: 0.398, band_2: 0.467, band_3: 0.534, band_4: 0.501, band_5: 0.445, label: "Asphalt",         class_id: 1 },
  { id: 11, band_1: 0.589, band_2: 0.667, band_3: 0.745, band_4: 0.712, band_5: 0.656, label: "Meadows",         class_id: 2 },
  { id: 12, band_1: 0.267, band_2: 0.334, band_3: 0.401, band_4: 0.378, band_5: 0.312, label: "Gravel",          class_id: 3 },
  { id: 13, band_1: 0.112, band_2: 0.178, band_3: 0.234, band_4: 0.256, band_5: 0.201, label: "Trees",           class_id: 4 },
  { id: 14, band_1: 0.698, band_2: 0.778, band_3: 0.856, band_4: 0.823, band_5: 0.767, label: "Metal Sheets",    class_id: 5 },
  { id: 15, band_1: 0.412, band_2: 0.489, band_3: 0.556, band_4: 0.512, band_5: 0.467, label: "Bare Soil",       class_id: 6 },
]
 
// ── 训练过程占位数据（后续由后端 recorder 的 append_index_value 替换）──
export const mockTrainingHistory = [
  { epoch: 1,  loss: 1.823, accuracy: 0.31, val_loss: 1.901, val_accuracy: 0.28 },
  { epoch: 5,  loss: 1.234, accuracy: 0.52, val_loss: 1.312, val_accuracy: 0.49 },
  { epoch: 10, loss: 0.876, accuracy: 0.67, val_loss: 0.934, val_accuracy: 0.63 },
  { epoch: 20, loss: 0.534, accuracy: 0.78, val_loss: 0.612, val_accuracy: 0.74 },
  { epoch: 30, loss: 0.312, accuracy: 0.86, val_loss: 0.398, val_accuracy: 0.82 },
  { epoch: 40, loss: 0.198, accuracy: 0.91, val_loss: 0.267, val_accuracy: 0.87 },
  { epoch: 50, loss: 0.134, accuracy: 0.94, val_loss: 0.212, val_accuracy: 0.90 },
  { epoch: 60, loss: 0.098, accuracy: 0.96, val_loss: 0.187, val_accuracy: 0.92 },
  { epoch: 70, loss: 0.076, accuracy: 0.97, val_loss: 0.176, val_accuracy: 0.93 },
  { epoch: 80, loss: 0.061, accuracy: 0.98, val_loss: 0.171, val_accuracy: 0.93 },
]
 
// ── 混淆矩阵占位（PaviaU 9类，后续由后端 evaluation.py 的 confusion 替换）──
export const mockConfusionMatrix = [
  [430,  12,   3,   0,   1,   8,   2,   4,   0],
  [  8, 1821,  15,   3,   0,  12,   4,   9,   1],
  [  2,   9, 198,   4,   1,   3,   8,   2,   0],
  [  0,   4,   3, 298,   2,   5,   1,   3,   0],
  [  1,   0,   2,   1, 131,   0,   2,   1,   0],
  [  5,  14,   4,   6,   0, 487,   3,   7,   1],
  [  3,   5,   7,   2,   3,   4, 320,   2,   0],
  [  2,   8,   3,   4,   1,   6,   2, 354,   1],
  [  0,   1,   0,   0,   0,   1,   0,   1,  93],
]
 
// ── HSI 评估指标占位（后续由后端 /evaluate 接口替换）──────────────────
export const mockMetrics = {
  oa: 0.9234,      // Overall Accuracy
  aa: 0.9012,      // Average Accuracy
  kappa: 0.9034,   // Kappa Coefficient
  // 逐类精度（PaviaU 9类）
  each_acc: [0.954, 0.982, 0.901, 0.945, 0.934, 0.967, 0.923, 0.956, 0.978],
}
 
// ── 类型定义（保持不变，供 file-sidebar 使用）─────────────────────────
export type FileItem = {
  id: string
  name: string
  type: "csv" | "json" | "xlsx" | "model" | "txt" | "mat"  // 新增 mat 类型
  size: string
  data?: Record<string, unknown>[]
}
 
// ── 侧边栏初始文件（换成 HSI 数据集示例）────────────────────────────
export const initialFiles: FileItem[] = [
  { id: "1", name: "PaviaU.mat",      type: "mat",   size: "13.2 MB" },
  { id: "2", name: "PaviaU_gt.mat",   type: "mat",   size: "0.3 MB"  },
  { id: "3", name: "transformer_delta.pth", type: "model", size: "48.6 MB" },
]
 
// ── 模型选项（对应 trainer.py 中的 trainer_type）────────────────────
export const modelOptions = [
  { value: "transformer_DELTA", label: "HyTAC", config: "pavia_delta_ctent.json", train_sign: "ctent" },
  { value: "simple_cnn",        label: "SimpleCNN",        config: "pavia_simplecnn.json",   train_sign: "train" },
  { value: "cnn_attention",     label: "CNN + Attention",  config: "pavia_cnn.json",         train_sign: "train" },
  { value: "vit",               label: "ViT",              config: "pavia_vit.json",          train_sign: "train" },
  { value: "SQSFormer", label: "SQS", config: "pavia_SQS.json", train_sign: "train" },
]
 
// ── 训练模式选项（对应 workflow.py 的 train_sign）────────────────────
export const trainSignOptions = [
  { value: "train", label: "训练 (Train)"  },
  { value: "test",  label: "测试 (Test)"   },
  { value: "tent",  label: "TENT 自适应"   },
  { value: "ctent", label: "C-TENT 自适应" },
]
 
// ── 噪声类型选项（对应 workflow.py 的 noise_type_list）──────────────
export const noiseOptions = [
  { value: "clean",       label: "无噪声 (Clean)"      },
  { value: "jpeg",        label: "JPEG 压缩噪声"        },
  { value: "additive",    label: "加性高斯噪声"          },
  { value: "poisson",     label: "泊松噪声"             },
  { value: "salt_pepper", label: "椒盐噪声"             },
  { value: "stripes",     label: "条纹噪声"             },
  { value: "deadlines",   label: "死线噪声"             },
  { value: "kernal",      label: "模糊噪声"             },
  { value: "thick_fog",   label: "浓雾噪声"             },
]
 
// ── 格式转换选项（对应前端 toolbar 格式转换下拉）────────────────────
// （这部分在HSI场景下改为数据预处理操作）
export const formatOptions = [
  { value: "tif_to_mat", label: ".tif → .mat" },
  { value: "mat_to_npy",   label: ".mat → .npy"   },
  { value: "apply_pca",    label: "PCA 降维"       },
  { value: "normalize",    label: "光谱归一化"     },
  { value: "patch_crop",   label: "图像块裁剪"     },
  { value: "gt_remap",     label: "标签重映射"     },
]
 
// ── 数据集选项（供后续配置文件选择使用）────────────────────────────
export const datasetOptions = [
  { value: "PaviaU",  label: "Pavia University" },
  { value: "Indian",  label: "Indian Pines"     },
  { value: "WH",      label: "WHU-Hi-WHU"       },
  { value: "LK",      label: "WHU-Hi-LK"        },
  { value: "WHLK",    label: "WHU-Hi-WHLK"      },
]