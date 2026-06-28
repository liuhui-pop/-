"use client"

import { useState, useEffect, useRef } from "react"
import { Plus, Trash2, Play, Loader2, X, ChevronDown, ChevronUp } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select"
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
} from "@/components/ui/dialog"
import { modelOptions, noiseOptions, datasetOptions } from "@/lib/mock-data"
import type { FileItem } from "@/lib/mock-data"

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000"

interface BatchTask {
  id: string
  model_value: string
  dataset_sign: string
  train_sign: string
  noise_type: string
  repeat: number
  lr: number
  epochs: number
  batch_size: number
  uploaded_file_id?: string
}

interface BatchResult {
  job_id: string
  status: "done" | "failed"
  task_idx: number
  repeat: number
  config: string
  oa?: number
  aa?: number
  kappa?: number
  eval_time?: number
  error?: string
}

interface BatchStatus {
  status: "pending" | "running" | "done"
  total: number
  current: number
  message: string
  results: BatchResult[]
}

interface BatchTrainDialogProps {
  open: boolean
  onClose: () => void
  files: FileItem[]
  activeFileId: string | null
}

const TRAIN_SIGN_OPTIONS = [
  { value: "train", label: "训练" },
  { value: "ctent", label: "C-TENT" },
  { value: "test",  label: "测试"  },
]

function newTask(activeFileId?: string | null): BatchTask {
  return {
    id:           Math.random().toString(36).slice(2),
    model_value:  "simple_cnn",
    dataset_sign: "Pavia",
    train_sign:   "train",
    noise_type:   "clean",
    repeat:       1,
    lr:           0.001,
    epochs:       100,
    batch_size:   64,
    uploaded_file_id: activeFileId ?? undefined,
  }
}

export function BatchTrainDialog({ open, onClose, files, activeFileId }: BatchTrainDialogProps) {
  const [tasks, setTasks]         = useState<BatchTask[]>([newTask(activeFileId)])
  const [batchId, setBatchId]     = useState<string | null>(null)
  const [status, setStatus]       = useState<BatchStatus | null>(null)
  const [running, setRunning]     = useState(false)
  const [expanded, setExpanded]   = useState<string[]>([])
  const pollRef = useRef<NodeJS.Timeout | null>(null)

  // 轮询批量训练进度
  useEffect(() => {
    if (!batchId || !running) return
    pollRef.current = setInterval(async () => {
      try {
        const res  = await fetch(`${BACKEND_URL}/batch_train/${batchId}`)
        const data: BatchStatus = await res.json()
        setStatus(data)
        if (data.status === "done") {
          setRunning(false)
          clearInterval(pollRef.current!)
        }
      } catch {}
    }, 2000)
    return () => clearInterval(pollRef.current!)
  }, [batchId, running])

  const handleStart = async () => {
    setRunning(true)
    setStatus(null)
    try {
      const res = await fetch(`${BACKEND_URL}/batch_train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tasks: tasks.map((t) => ({
            model_value:      t.model_value,
            dataset_sign:     t.dataset_sign,
            train_sign:       t.train_sign,
            noise_type:       t.noise_type,
            repeat:           t.repeat,
            lr:               t.lr,
            epochs:           t.epochs,
            batch_size:       t.batch_size,
            uploaded_file_id: t.uploaded_file_id,
          })),
        }),
      })
      const data = await res.json()
      setBatchId(data.batch_id)
    } catch (e) {
      setRunning(false)
      alert("提交失败")
    }
  }

  const updateTask = (id: string, patch: Partial<BatchTask>) => {
    setTasks((prev) => prev.map((t) => t.id === id ? { ...t, ...patch } : t))
  }

  const removeTask = (id: string) => {
    setTasks((prev) => prev.filter((t) => t.id !== id))
  }

  const toggleExpand = (id: string) => {
    setExpanded((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    )
  }

  const totalRuns = tasks.reduce((s, t) => s + t.repeat, 0)

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-3xl max-h-[85vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="text-sm">批量训练配置</DialogTitle>
        </DialogHeader>

        <div className="flex-1 overflow-auto space-y-3 pr-1">

          {/* 任务列表 */}
          {tasks.map((task, idx) => (
            <div key={task.id} className="border border-border rounded-lg overflow-hidden">
              {/* 任务标题行 */}
              <div className="flex items-center gap-2 px-3 py-2 bg-muted/30 cursor-pointer"
                onClick={() => toggleExpand(task.id)}>
                <span className="text-xs font-mono text-muted-foreground w-5">{idx + 1}</span>
                <Badge variant="outline" className="text-[10px]">
                  {modelOptions.find((m) => m.value === task.model_value)?.label ?? task.model_value}
                </Badge>
                <Badge variant="secondary" className="text-[10px]">{task.dataset_sign}</Badge>
                <Badge variant="secondary" className="text-[10px]">×{task.repeat}</Badge>
                <span className="text-[10px] text-muted-foreground ml-auto">
                  {TRAIN_SIGN_OPTIONS.find((t) => t.value === task.train_sign)?.label}
                </span>
                <Button variant="ghost" size="icon" className="h-5 w-5 text-muted-foreground hover:text-destructive"
                  onClick={(e) => { e.stopPropagation(); removeTask(task.id) }}>
                  <X className="h-3 w-3" />
                </Button>
                {expanded.includes(task.id)
                  ? <ChevronUp className="h-3 w-3 text-muted-foreground" />
                  : <ChevronDown className="h-3 w-3 text-muted-foreground" />}
              </div>

              {/* 展开的参数配置 */}
              {expanded.includes(task.id) && (
                <div className="p-3 grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <Label className="text-xs">模型</Label>
                    <Select value={task.model_value}
                      onValueChange={(v) => updateTask(task.id, { model_value: v })}>
                      <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {modelOptions.map((o) => (
                          <SelectItem key={o.value} value={o.value} className="text-xs">{o.label}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-1">
                    <Label className="text-xs">数据集</Label>
                    <Select value={task.dataset_sign}
                      onValueChange={(v) => updateTask(task.id, { dataset_sign: v })}>
                      <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {datasetOptions.map((o) => (
                          <SelectItem key={o.value} value={o.value} className="text-xs">{o.label}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-1">
                    <Label className="text-xs">训练模式</Label>
                    <Select value={task.train_sign}
                      onValueChange={(v) => updateTask(task.id, { train_sign: v })}>
                      <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {TRAIN_SIGN_OPTIONS.map((o) => (
                          <SelectItem key={o.value} value={o.value} className="text-xs">{o.label}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-1">
                    <Label className="text-xs">噪声类型</Label>
                    <Select value={task.noise_type}
                      onValueChange={(v) => updateTask(task.id, { noise_type: v })}>
                      <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {noiseOptions.map((o) => (
                          <SelectItem key={o.value} value={o.value} className="text-xs">{o.label}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-1">
                    <Label className="text-xs">数据文件</Label>
                    <Select value={task.uploaded_file_id ?? "__default__"}
                      onValueChange={(v) => updateTask(task.id, { uploaded_file_id: v === "__default__" ? undefined : v })}>
                      <SelectTrigger className="h-8 text-xs"><SelectValue placeholder="使用配置文件默认" /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="__default__" className="text-xs text-muted-foreground">使用配置文件默认</SelectItem>
                        {files.filter((f) => f.type === "mat").map((f) => (
                          <SelectItem key={f.id} value={f.id} className="text-xs font-mono">{f.name}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-1">
                    <Label className="text-xs">重复次数</Label>
                    <Input type="number" min={1} max={20} value={task.repeat}
                      onChange={(e) => updateTask(task.id, { repeat: parseInt(e.target.value) || 1 })}
                      className="h-8 text-xs font-mono" />
                  </div>

                  <div className="space-y-1">
                    <Label className="text-xs">学习率</Label>
                    <Input type="number" step="0.0001" value={task.lr}
                      onChange={(e) => updateTask(task.id, { lr: parseFloat(e.target.value) || 0.001 })}
                      className="h-8 text-xs font-mono" />
                  </div>

                  <div className="space-y-1">
                    <Label className="text-xs">迭代次数</Label>
                    <Input type="number" min={1} value={task.epochs}
                      onChange={(e) => updateTask(task.id, { epochs: parseInt(e.target.value) || 100 })}
                      className="h-8 text-xs font-mono" />
                  </div>
                </div>
              )}
            </div>
          ))}

          {/* 添加任务按钮 */}
          <Button variant="outline" size="sm" className="w-full gap-1.5 text-xs"
            onClick={() => { const t = newTask(activeFileId); setTasks((p) => [...p, t]); setExpanded((p) => [...p, t.id]) }}
            disabled={running}>
            <Plus className="h-3.5 w-3.5" />添加任务
          </Button>

          {/* 进度显示 */}
          {status && (
            <div className="border border-border rounded-lg p-3 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium">
                  {status.status === "done" ? "批量训练完成" : "训练中..."}
                </span>
                <span className="text-xs font-mono text-muted-foreground">
                  {status.current} / {status.total}
                </span>
              </div>

              {/* 进度条 */}
              <div className="w-full bg-muted rounded-full h-1.5">
                <div className="bg-primary h-1.5 rounded-full transition-all"
                  style={{ width: `${status.total > 0 ? (status.current / status.total) * 100 : 0}%` }} />
              </div>

              <p className="text-[10px] text-muted-foreground font-mono">{status.message}</p>

              {/* 结果列表 */}
              {status.results.length > 0 && (
                <div className="space-y-1 max-h-40 overflow-auto">
                  {status.results.map((r, i) => (
                    <div key={i} className={`flex items-center gap-2 text-[10px] font-mono px-2 py-1 rounded ${
                      r.status === "done" ? "bg-emerald-500/10" : "bg-rose-500/10"
                    }`}>
                      <span className={r.status === "done" ? "text-emerald-400" : "text-rose-400"}>
                        {r.status === "done" ? "✓" : "✗"}
                      </span>
                      <span className="text-muted-foreground">{r.config} #{r.repeat}</span>
                      {r.status === "done" && (
                        <>
                          <span className="ml-auto text-emerald-400">OA {r.oa?.toFixed(2)}%</span>
                          <span>AA {r.aa?.toFixed(2)}%</span>
                          <span>κ {r.kappa?.toFixed(2)}%</span>
                        </>
                      )}
                      {r.status === "failed" && (
                        <span className="text-rose-400 text-[9px] break-all">
  {r.error?.split('\n').filter(Boolean).slice(-3).join('\n')}
</span>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        <DialogFooter className="gap-2 mt-2">
          <span className="text-xs text-muted-foreground mr-auto">
            共 {tasks.length} 个任务，{totalRuns} 次训练
          </span>
          <Button variant="outline" size="sm" onClick={onClose} disabled={running}>关闭</Button>
          <Button size="sm" className="gap-1.5" onClick={handleStart}
            disabled={running || tasks.length === 0}>
            {running
              ? <><Loader2 className="h-3.5 w-3.5 animate-spin" />训练中...</>
              : <><Play className="h-3.5 w-3.5" />开始批量训练</>}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
