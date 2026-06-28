"use client"
import { ImportModelDialog } from "@/components/dashboard/import-model-dialog"
import { useState, useCallback, useRef, useEffect } from "react"
import { Toolbar, type HyperParams, type SplitParams } from "@/components/dashboard/toolbar"
import { FileSidebar } from "@/components/dashboard/file-sidebar"
import { MainView } from "@/components/dashboard/main-view"
import { initialFiles, modelOptions, type FileItem } from "@/lib/mock-data"
import {
  uploadFile, submitTrain, pollJobStatus,
  fetchEvaluate, fetchVisualize, downloadResult,
  detectDataset,
  type EvaluateResult, type VisualizeResult,
} from "@/lib/api"
import { BatchTrainDialog } from "@/components/dashboard/batch-train-dialog"
import { MinesweeperDialog } from "@/components/dashboard/minesweeper-dialog"
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000"

type TrainStatus = "idle" | "pending" | "running" | "done" | "failed"
type SplitStatus = "idle" | "splitting" | "done" | "failed"

export interface TrainState {
  status: TrainStatus
  jobId: string | null
  message: string
  evalResult: EvaluateResult | null
  visualResult: VisualizeResult | null
}

export default function DashboardPage() {
  // ── 文件侧边栏 ──────────────────────────────────────────────
  const [files, setFiles]           = useState<FileItem[]>(initialFiles)
  const [activeFileId, setActiveFileId] = useState<string | null>("1")

  // ── Toolbar 基础状态 ────────────────────────────────────────
  const [activeTab, setActiveTab]       = useState("table")
  const [selectedModel, setSelectedModel] = useState("transformer_DELTA")
  const [selectedFormat, setSelectedFormat] = useState("")
  const [selectedNoise, setSelectedNoise]   = useState("clean")
  const [importModelOpen, setImportModelOpen] = useState(false)
  const [dynamicModels, setDynamicModels] = useState<{value: string; label: string; config?: string; train_sign?: string}[]>([])
  // ── 超参数状态（对应 Popover 输入框）────────────────────────
  const [hyperParams, setHyperParams] = useState<HyperParams>({
    lr: 0.001, epochs: 100, batchSize: 64,
  })

  // ── 数据划分参数 ────────────────────────────────────────────
  const [splitParams, setSplitParams] = useState<SplitParams>({
    mode: "per_class", value: 10,
  })
  const [splitStatus, setSplitStatus] = useState<SplitStatus>("idle")
  // 划分完成后生成的 split 文件 ID（供训练使用）
  const [splitFileId, setSplitFileId] = useState<string | null>(null)
  const [minesweeperOpen, setMinesweeperOpen] = useState(false)

  // 自动识别的数据集信息
  const [datasetInfo, setDatasetInfo] = useState<{
    sign: string; numClasses: number; spectral: number
  } | null>(null)

  // ── 训练任务状态 ────────────────────────────────────────────
  const [trainState, setTrainState] = useState<TrainState>({
    status: "idle", jobId: null, message: "",
    evalResult: null, visualResult: null,
  })
  const pollingRef = useRef(false)
  const [batchOpen, setBatchOpen] = useState(false)

  // ── 当前文件信息 ────────────────────────────────────────────
  const activeFile     = files.find((f) => f.id === activeFileId)
  const currentFileName = activeFile?.name ?? null
  const [currentData, setCurrentData] = useState<Record<string, unknown>[]>([])

  // ═══════════════════════════════════════════════════════════
  // handler：导入数据
  // ═══════════════════════════════════════════════════════════
  const handleImport = useCallback(() => {
    const input = document.createElement("input")
    input.type = "file"
    input.accept = ".mat,.npy,.csv,.tif,.tiff"
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (!file) return
      try {
        const result = await uploadFile(file)
        const newFile: FileItem = {
          id: result.file_id,
          name: result.file_name,
          type: file.name.endsWith(".mat") ? "mat" : "csv",
          size: `${result.size_mb} MB`,
        }
        setFiles((prev) => [...prev, newFile])
        setActiveFileId(newFile.id)
        // 上传新文件后重置划分状态
        setSplitStatus("idle")
        setSplitFileId(null)
        setDatasetInfo(null)

        // 自动识别数据集（同时判断是否已是 split.mat）
        try {
          const detected = await detectDataset(result.file_id)
          if (detected.detected && detected.dataset_sign) {
            setDatasetInfo({
              sign:       detected.dataset_sign,
              numClasses: detected.num_classes ?? 9,
              spectral:   detected.spectral ?? 103,
            })
            // 如果上传的是已划分好的 split.mat，直接标记可训练
            if ((detected as any).is_split) {
              setSplitFileId(result.file_id)
              setSplitStatus("done")
            }
          }
        } catch (_) {}
      } catch (err) {
        alert(`导入失败：${(err as Error).message}`)
      }
    }
    input.click()
  }, [])

  // ═══════════════════════════════════════════════════════════
  // handler：数据划分
  // ═══════════════════════════════════════════════════════════
  const handleSplit = useCallback(async () => {
    if (!activeFileId) {
      alert("请先导入数据文件")
      return
    }
    setSplitStatus("splitting")
    try {
      const res = await fetch(`${BACKEND_URL}/split`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          file_id:    activeFileId,
          split_mode: splitParams.mode,
          split_value: splitParams.value,
        }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail ?? "划分失败")
      }
      const data = await res.json()
      setSplitFileId(data.split_file_id)
      setSplitStatus("done")
      if (data.dataset) {
        setDatasetInfo({
          sign:       data.dataset,
          numClasses: data.num_classes ?? 9,
          spectral:   103,
        })
      }

      // 把生成的 split 文件加入侧边栏
      const splitFile: FileItem = {
        id: data.split_file_id,
        name: data.split_file_name,
        type: "mat",
        size: `${data.size_mb} MB`,
      }
      setFiles((prev) => [...prev, splitFile])
      setActiveFileId(data.split_file_id)
    } catch (err) {
      setSplitStatus("failed")
      alert(`数据划分失败：${(err as Error).message}`)
    }
  }, [activeFileId, splitParams])

  // ═══════════════════════════════════════════════════════════
  // handler：训练
  // ═══════════════════════════════════════════════════════════
  const handleTrain = useCallback(async () => {
    // 优先使用划分后的文件，否则使用当前激活文件
    const trainFileId = splitFileId ?? activeFileId

    setTrainState({
      status: "pending", jobId: null,
      message: "正在提交训练任务...",
      evalResult: null, visualResult: null,
    })
    setActiveTab("model")

    try {
      const modelCfg  = modelOptions.find((m) => m.value === selectedModel)
      const trainSign = modelCfg?.train_sign ?? "train"
      const configName = modelCfg?.config ?? `pavia_${selectedModel.toLowerCase()}.json`

      const { job_id } = await submitTrain({
        // 不传 config_name，由后端根据 dataset+model 自动选择
        model_value:      selectedModel,
        dataset_sign:     datasetInfo?.sign,
        train_sign:       trainSign,
        noise_type:       selectedNoise,
        uploaded_file_id: trainFileId ?? undefined,
        lr:               hyperParams.lr,
        epochs:           hyperParams.epochs,
        batch_size:       hyperParams.batchSize,
      })

      setTrainState((prev) => ({
        ...prev, status: "running", jobId: job_id,
        message: "训练进行中，请稍候...",
      }))

      pollingRef.current = true
      const finalStatus = await pollJobStatus(job_id, (s) => {
        if (!pollingRef.current) return
        setTrainState((prev) => ({ ...prev, message: s.message }))
      }, 3000)
      pollingRef.current = false

      if (finalStatus.status === "failed") {
        setTrainState((prev) => ({
          ...prev, status: "failed",
          message: `训练失败：${finalStatus.error ?? "未知错误"}`,
        }))
        return
      }

      const [evalResult, visualResult] = await Promise.all([
        fetchEvaluate(job_id),
        fetchVisualize(job_id),
      ])
      setTrainState({
        status: "done", jobId: job_id, message: "训练完成！",
        evalResult, visualResult,
      })
    } catch (err) {
      pollingRef.current = false
      setTrainState((prev) => ({
        ...prev, status: "failed",
        message: `错误：${(err as Error).message}`,
      }))
    }
  }, [selectedModel, activeFileId, splitFileId, hyperParams, selectedNoise])

  // ═══════════════════════════════════════════════════════════
  // handler：评估 / 导出
  // ═══════════════════════════════════════════════════════════
  const handleEvaluate = useCallback(async () => {
    const { jobId } = trainState
    if (!jobId) { alert("请先完成训练"); return }
    try {
      const [evalResult, visualResult] = await Promise.all([
        fetchEvaluate(jobId), fetchVisualize(jobId),
      ])
      setTrainState((prev) => ({ ...prev, evalResult, visualResult }))
      setActiveTab("model")
    } catch (err) {
      alert(`评估失败：${(err as Error).message}`)
    }
  }, [trainState])

  const handleExport = useCallback(() => {
    if (!trainState.jobId) { alert("暂无可导出的结果，请先完成训练"); return }
    downloadResult(trainState.jobId)
  }, [trainState])

  // ═══════════════════════════════════════════════════════════
  // handler：终止训练
  // ═══════════════════════════════════════════════════════════
  const handleCancel = useCallback(async () => {
    const { jobId } = trainState
    if (!jobId) return
    try {
      await fetch(`${BACKEND_URL}/cancel/${jobId}`, { method: "POST" })
      setTrainState((prev) => ({
        ...prev, status: "failed", message: "训练已被终止",
      }))
    } catch (err) {
      console.error("[终止失败]", err)
    }
  }, [trainState])

  // ═══════════════════════════════════════════════════════════
  // handler：格式转换（tif → mat）
  // ═══════════════════════════════════════════════════════════
  const handleApplyFormat = useCallback(async () => {
    if (!activeFileId) { alert("请先选中要转换的文件"); return }
    if (selectedFormat !== "tif_to_mat") { alert("请在格式转换下拉框中选择转换格式"); return }

    try {
      const res = await fetch(`${BACKEND_URL}/convert`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_id: activeFileId, format: selectedFormat }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail ?? "转换失败")
      }
      const data = await res.json()
      const newFile: FileItem = {
        id:   data.file_id,
        name: data.file_name,
        type: "mat",
        size: `${data.size_mb} MB`,
      }
      setFiles((prev) => [...prev, newFile])
      setActiveFileId(data.file_id)
      alert(data.message)
    } catch (err) {
      alert(`格式转换失败：${(err as Error).message}`)
    }
  }, [activeFileId, selectedFormat])

  // ═══════════════════════════════════════════════════════════
  // 文件侧边栏 handlers
  // ═══════════════════════════════════════════════════════════
  const handleFileSelect = useCallback(async (id: string) => {
    setActiveFileId(id)
    // 切换文件时重新识别数据集
    try {
      const detected = await detectDataset(id)
      if (detected.detected && detected.dataset_sign) {
        setDatasetInfo({
          sign:       detected.dataset_sign,
          numClasses: detected.num_classes ?? 9,
          spectral:   detected.spectral ?? 103,
        })
        if ((detected as any).is_split) {
          setSplitFileId(id)
          setSplitStatus("done")
        } else {
          setSplitFileId(null)
          setSplitStatus("idle")
        }
      }
    } catch (_) {}
  }, [])
  const handleFileRemove = useCallback((id: string) => {
    setFiles((prev) => {
      const next = prev.filter((f) => f.id !== id)
      if (activeFileId === id) setActiveFileId(next[0]?.id ?? null)
      return next
    })
  }, [activeFileId])
  const handleAddFile  = useCallback(() => handleImport(), [handleImport])
  const handleClearAll = useCallback(() => { setFiles([]); setActiveFileId(null) }, [])
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "F1") {
        e.preventDefault()
        setMinesweeperOpen((prev) => !prev)
      }
    }
    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [])

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <Toolbar
        currentFile={currentFileName}
        onImport={handleImport}
        selectedModel={selectedModel}
        onModelChange={setSelectedModel}
        selectedFormat={selectedFormat}
        onFormatChange={setSelectedFormat}
        selectedNoise={selectedNoise}
        onNoiseChange={setSelectedNoise}
        hyperParams={hyperParams}
        onHyperParamsChange={setHyperParams}
        splitParams={splitParams}
        onSplitParamsChange={setSplitParams}
        onApplyFormat={handleApplyFormat}
        onSplit={handleSplit}
        onTrain={handleTrain}
        onImportModel={() => setImportModelOpen(true)}
        onBatchTrain={() => setBatchOpen(true)}
        
        onCancel={handleCancel}
        onEvaluate={handleEvaluate}
        onExport={handleExport}
        dynamicModels={dynamicModels}
        trainStatus={trainState.status}
        trainMessage={trainState.message}
        splitStatus={splitStatus}
      />
      <div className="flex-1 flex overflow-hidden">
        <FileSidebar
          files={files}
          activeFileId={activeFileId}
          onFileSelect={handleFileSelect}
          onFileRemove={handleFileRemove}
          onAddFile={handleAddFile}
          onClearAll={handleClearAll}
        />
        <MainView
          data={currentData}
          activeTab={activeTab}
          onTabChange={setActiveTab}
          trainState={trainState}
          activeFile={activeFile}
        />
      </div>
      <BatchTrainDialog
          open={batchOpen}
          onClose={() => setBatchOpen(false)}
          files={files}
          activeFileId={activeFileId}
        />
      <ImportModelDialog
          open={importModelOpen}
          onClose={() => setImportModelOpen(false)}
          onSuccess={(model) => {
            setDynamicModels((prev) => [...prev, model])
            setSelectedModel(model.value)
          }}
        />
        <MinesweeperDialog
        open={minesweeperOpen}
        onClose={() => setMinesweeperOpen(false)}
      />
    </div>
  )
}
