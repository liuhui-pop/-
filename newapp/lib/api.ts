/**
 * lib/api.ts
 * 前端与后端的所有通信都经过这里，page.tsx 只调用这些函数，不直接写 fetch。
 *
 * 后端地址统一在 BACKEND_URL 配置，开发时改这一处即可。
 */

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000"

// ─────────────────────────────────────────────────────────────
// 通用工具
// ─────────────────────────────────────────────────────────────

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BACKEND_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body?.detail ?? `请求失败 ${res.status}: ${path}`)
  }
  return res.json() as Promise<T>
}

// ─────────────────────────────────────────────────────────────
// 类型定义（与后端 api.py 的返回结构对应）
// ─────────────────────────────────────────────────────────────

/** POST /upload 返回 */
export interface UploadResult {
  file_id: string
  file_name: string
  saved_as: string
  size_mb: number
  message: string
}

/** POST /train 返回 */
export interface TrainSubmitResult {
  job_id: string
  status: string
  message: string
}

/** GET /job/:id 返回 */
export interface JobStatus {
  job_id: string
  /** "pending" | "running" | "done" | "failed" */
  status: string
  message: string
  result?: {
    oa: number
    aa: number
    kappa: number
    each_acc: string
    eval_time: number
    result_path: string
  }
  error?: string
}

/** GET /evaluate/:id 返回 */
export interface EvaluateResult {
  job_id: string
  oa: number
  aa: number
  kappa: number
  each_acc: number[]
  eval_time_seconds: number
}

/** GET /visualize/:id 返回 */
export interface VisualizeResult {
  job_id: string
  metrics_bar: { name: string; value: number }[]
  class_acc: { class: string; accuracy: number }[]
  confusion_matrix_text: string
}

/** GET /configs 返回 */
export interface ConfigsResult {
  configs: string[]
}

// ─────────────────────────────────────────────────────────────
// 接口 1：上传数据文件（对应 Toolbar「导入数据」按钮）
// ─────────────────────────────────────────────────────────────

/**
 * 上传 .mat / .npy / .csv 文件。
 * 用法：const result = await uploadFile(file)
 */
export async function uploadFile(file: File): Promise<UploadResult> {
  const formData = new FormData()
  formData.append("file", file)

  const res = await fetch(`${BACKEND_URL}/upload`, {
    method: "POST",
    body: formData,
    // 注意：使用 FormData 时不要手动设置 Content-Type，浏览器会自动加 boundary
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body?.detail ?? `上传失败 ${res.status}`)
  }
  return res.json()
}

// ─────────────────────────────────────────────────────────────
// 接口 2：发起训练（对应 Toolbar「训练」按钮）
// ─────────────────────────────────────────────────────────────

export interface TrainParams {
  config_name?: string        // 不传时自动选择
  model_value?: string        // 模型 value，如 "simple_cnn"
  dataset_sign?: string       // 数据集，如 "Pavia"、"Indian"
  train_sign: string
  noise_type: string
  uploaded_file_id?: string
  lr?: number
  epochs?: number
  batch_size?: number
}

// ── 数据集检测结果 ────────────────────────────────────────────
export interface DatasetDetectResult {
  detected: boolean
  file_id: string
  file_name: string
  dataset_sign: string | null
  num_classes: number | null
  spectral: number | null
  message: string
}

/**
 * 上传文件后调用，自动识别数据集类型。
 */
export async function detectDataset(fileId: string): Promise<DatasetDetectResult> {
  return request<DatasetDetectResult>(`/detect_dataset/${fileId}`)
}

/**
 * 提交训练任务，立即返回 job_id。
 * 训练在后端异步执行，需要用 pollJobStatus 轮询结果。
 */
export async function submitTrain(params: TrainParams): Promise<TrainSubmitResult> {
  return request<TrainSubmitResult>("/train", {
    method: "POST",
    body: JSON.stringify(params),
  })
}

// ─────────────────────────────────────────────────────────────
// 接口 3：查询训练任务状态（轮询用）
// ─────────────────────────────────────────────────────────────

/**
 * 查询单次任务状态。
 * 通常不直接调用，而是用下面的 pollJobStatus。
 */
export async function getJobStatus(jobId: string): Promise<JobStatus> {
  return request<JobStatus>(`/job/${jobId}`)
}

/**
 * 轮询训练状态，直到完成或失败。
 *
 * @param jobId      训练任务 ID
 * @param onProgress 每次轮询后的回调，可用于更新 UI 状态文字
 * @param intervalMs 轮询间隔，默认 3000ms
 * @returns          最终的 JobStatus（status 为 "done" 或 "failed"）
 *
 * 用法：
 *   const final = await pollJobStatus(jobId, (s) => setStatusMsg(s.message))
 */
export async function pollJobStatus(
  jobId: string,
  onProgress?: (status: JobStatus) => void,
  intervalMs = 3000
): Promise<JobStatus> {
  return new Promise((resolve, reject) => {
    const timer = setInterval(async () => {
      try {
        const status = await getJobStatus(jobId)
        onProgress?.(status)
        if (status.status === "done" || status.status === "failed") {
          clearInterval(timer)
          resolve(status)
        }
      } catch (err) {
        clearInterval(timer)
        reject(err)
      }
    }, intervalMs)
  })
}

// ─────────────────────────────────────────────────────────────
// 接口 4：获取评估指标（对应 Toolbar「评估」按钮）
// ─────────────────────────────────────────────────────────────

/**
 * 获取训练完成后的 OA / AA / Kappa 及逐类精度。
 * 需要先有 job_id 且 status === "done"。
 */
export async function fetchEvaluate(jobId: string): Promise<EvaluateResult> {
  return request<EvaluateResult>(`/evaluate/${jobId}`)
}

// ─────────────────────────────────────────────────────────────
// 接口 5：获取可视化数据（对应 MainView「模型输出」Tab）
// ─────────────────────────────────────────────────────────────

/**
 * 获取图表所需结构化数据：metrics_bar / class_acc / confusion_matrix_text
 */
export async function fetchVisualize(jobId: string): Promise<VisualizeResult> {
  return request<VisualizeResult>(`/visualize/${jobId}`)
}

// ─────────────────────────────────────────────────────────────
// 接口 6：导出结果（对应 Toolbar「导出」按钮）
// ─────────────────────────────────────────────────────────────

/**
 * 触发浏览器下载训练结果 JSON 文件。
 * 直接用 window.open 跳转到下载地址，后端返回 FileResponse。
 */
export function downloadResult(jobId: string): void {
  window.open(`${BACKEND_URL}/export/${jobId}`, "_blank")
}

// ─────────────────────────────────────────────────────────────
// 接口 7：获取可用配置文件列表（对应模型选择下拉）
// ─────────────────────────────────────────────────────────────

/**
 * 获取 ./params_use/ 下所有 .json 配置文件名。
 * 可用于动态填充「选择模型」下拉框。
 */
export async function fetchConfigs(): Promise<string[]> {
  const res = await request<ConfigsResult>("/configs")
  return res.configs
}

// ─────────────────────────────────────────────────────────────
// 接口 8：健康检查（可选，用于检测后端是否在线）
// ─────────────────────────────────────────────────────────────

export async function checkHealth(): Promise<boolean> {
  try {
    await request("/health")
    return true
  } catch {
    return false
  }
}
