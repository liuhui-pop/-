"use client"

import { useState } from "react"
import { Upload, CheckCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
} from "@/components/ui/dialog"
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select"

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000"

interface ImportModelDialogProps {
  open: boolean
  onClose: () => void
  onSuccess: (model: { value: string; label: string }) => void
}

export function ImportModelDialog({ open, onClose, onSuccess }: ImportModelDialogProps) {
  const [file,        setFile]        = useState<File | null>(null)
  const [trainerType, setTrainerType] = useState("")
  const [className,   setClassName]   = useState("")
  const [label,       setLabel]       = useState("")
  const [lr,          setLr]          = useState("0.001")
  const [optimizer,   setOptimizer]   = useState("adam")
  const [loading,     setLoading]     = useState(false)
  const [success,     setSuccess]     = useState(false)
  const [error,       setError]       = useState("")

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (!f) return
    setFile(f)
    // 自动填充默认值
    const stem = f.name.replace(".py", "")
    if (!trainerType) setTrainerType(stem.toLowerCase().replace(/\s+/g, "_"))
    if (!className)   setClassName(stem)
    if (!label)       setLabel(stem)
    setError("")
    setSuccess(false)
  }

  const handleSubmit = async () => {
    if (!file) { setError("请先选择模型文件"); return }
    if (!className) { setError("请填写模型类名"); return }

    setLoading(true)
    setError("")
    try {
      const form = new FormData()
      form.append("file", file)
      form.append("trainer_type", trainerType || file.name.replace(".py", "").toLowerCase())
      form.append("class_name",   className)
      form.append("label",        label || className)
      form.append("lr",           lr)
      form.append("optimizer",    optimizer)

      const res  = await fetch(`${BACKEND_URL}/upload_model`, { method: "POST", body: form })
      const data = await res.json()

      if (!res.ok) throw new Error(data.detail ?? "上传失败")

      setSuccess(true)
      alert(`注册成功: value=${data.trainer_type}, label=${data.label}`)
      onSuccess({ value: data.trainer_type, label: data.label })
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const handleClose = () => {
    setFile(null); setTrainerType(""); setClassName("")
    setLabel(""); setLr("0.001"); setOptimizer("adam")
    setSuccess(false); setError("")
    onClose()
  }

  return (
    <Dialog open={open} onOpenChange={(o) => !o && handleClose()}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="text-sm">导入自定义模型</DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          {/* 文件选择 */}
          <div className="space-y-1">
            <Label className="text-xs">模型文件 (.py)</Label>
            <div className="flex items-center gap-2">
              <Input type="file" accept=".py"
                onChange={handleFileChange}
                className="text-xs h-8 cursor-pointer" />
            </div>
            {file && (
              <p className="text-[10px] text-muted-foreground font-mono">{file.name}</p>
            )}
          </div>

          {/* 模型类名 */}
          <div className="space-y-1">
            <Label className="text-xs">模型类名 <span className="text-rose-400">*</span></Label>
            <Input value={className} onChange={(e) => setClassName(e.target.value)}
              placeholder="如：MyTransformer" className="h-8 text-xs font-mono" />
            <p className="text-[10px] text-muted-foreground">
              py 文件里定义的类名，forward 返回 (feat, logits) 或单个 tensor
            </p>
          </div>

          {/* 显示名称 */}
          <div className="space-y-1">
            <Label className="text-xs">显示名称</Label>
            <Input value={label} onChange={(e) => setLabel(e.target.value)}
              placeholder="如：我的 Transformer" className="h-8 text-xs" />
          </div>

          {/* 注册名 */}
          <div className="space-y-1">
            <Label className="text-xs">注册名（唯一标识）</Label>
            <Input value={trainerType} onChange={(e) => setTrainerType(e.target.value)}
              placeholder="如：my_transformer" className="h-8 text-xs font-mono" />
          </div>

          <div className="grid grid-cols-2 gap-3">
            {/* 学习率 */}
            <div className="space-y-1">
              <Label className="text-xs">默认学习率</Label>
              <Input type="number" value={lr} step="0.0001"
                onChange={(e) => setLr(e.target.value)}
                className="h-8 text-xs font-mono" />
            </div>

            {/* 优化器 */}
            <div className="space-y-1">
              <Label className="text-xs">优化器</Label>
              <Select value={optimizer} onValueChange={setOptimizer}>
                <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="adam"  className="text-xs">Adam</SelectItem>
                  <SelectItem value="adamw" className="text-xs">AdamW</SelectItem>
                  <SelectItem value="sgd"   className="text-xs">SGD</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* 错误 / 成功提示 */}
          {error && (
            <p className="text-xs text-rose-400 bg-rose-500/10 px-3 py-2 rounded-md">{error}</p>
          )}
          {success && (
            <div className="flex items-center gap-2 text-xs text-emerald-400 bg-emerald-500/10 px-3 py-2 rounded-md">
              <CheckCircle className="h-3.5 w-3.5" />
              模型注册成功！已加入模型下拉列表
            </div>
          )}
        </div>

        <DialogFooter className="gap-2">
          <Button variant="outline" size="sm" onClick={handleClose}>关闭</Button>
          <Button size="sm" className="gap-1.5" onClick={handleSubmit} disabled={loading || success}>
            {loading ? "注册中..." : <><Upload className="h-3.5 w-3.5" />注册模型</>}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
