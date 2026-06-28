"use client"

import { Upload, Settings2, Play, BarChart3, Download, Loader2, Scissors, Square } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select"
import {
  Popover, PopoverContent, PopoverTrigger,
} from "@/components/ui/popover"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import { modelOptions, formatOptions, noiseOptions } from "@/lib/mock-data"

export interface HyperParams {
  lr: number
  epochs: number
  batchSize: number
}

export interface SplitParams {
  mode: "per_class" | "percent"
  value: number
}

interface ToolbarProps {
  currentFile: string | null
  onImport: () => void
  selectedModel: string
  onModelChange: (value: string) => void
  selectedFormat: string
  onFormatChange: (value: string) => void
  selectedNoise: string
  onNoiseChange: (value: string) => void
  hyperParams: HyperParams
  onHyperParamsChange: (params: HyperParams) => void
  splitParams: SplitParams
  onSplitParamsChange: (params: SplitParams) => void
  onApplyFormat: () => void
  onSplit: () => void
  onTrain: () => void
  dynamicModels?: { value: string; label: string }[]
  onImportModel?: () => void
  onBatchTrain?: () => void
  onCancel?: () => void
  onEvaluate: () => void
  onExport: () => void
  trainStatus?: "idle" | "pending" | "running" | "done" | "failed"
  trainMessage?: string
  splitStatus?: "idle" | "splitting" | "done" | "failed"
}

export function Toolbar({
  currentFile, onImport,
  selectedModel, onModelChange,
  selectedFormat, onFormatChange,
  selectedNoise, onNoiseChange,
  hyperParams, onHyperParamsChange,
  splitParams, onSplitParamsChange, onSplit,
  onApplyFormat,
  dynamicModels = [],
  onImportModel,
  onTrain, onBatchTrain, onCancel, onEvaluate, onExport,
  trainStatus = "idle", trainMessage,
  splitStatus = "idle",
}: ToolbarProps) {
  const isTraining  = trainStatus === "pending" || trainStatus === "running"
  const isSplitting = splitStatus === "splitting"
  const hasDone     = trainStatus === "done"

  return (
    <div className="flex items-center gap-1 p-2 bg-card border-b border-border flex-wrap">

      {/* Group 1: 导入数据 */}
      <div className="flex items-center gap-2 px-2">
        <Button variant="outline" size="sm" onClick={onImport}
          disabled={isTraining} className="gap-1.5">
          <Upload className="h-3.5 w-3.5" />导入数据
        </Button>
        {currentFile && (
          <Badge variant="secondary" className="font-mono text-xs">{currentFile}</Badge>
        )}
      </div>

      <Separator orientation="vertical" className="h-6" />

      {/* Group 2: 模型选择 + 超参数 + 导入模型 */}
      <div className="flex items-center gap-1 px-2">
        <Select value={selectedModel} onValueChange={onModelChange} disabled={isTraining}>
          <SelectTrigger className="w-[160px] h-8 text-xs">
            <SelectValue placeholder="选择模型" />
          </SelectTrigger>
          <SelectContent>
            {[...modelOptions, ...dynamicModels].map((o) => (
              <SelectItem key={o.value} value={o.value} className="text-xs">{o.label}</SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Popover>
          <PopoverTrigger asChild>
            <Button variant="ghost" size="icon" className="h-8 w-8" disabled={isTraining}>
              <Settings2 className="h-3.5 w-3.5" />
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-72" align="start">
            <div className="grid gap-4">
              <div className="space-y-1">
                <h4 className="font-medium text-sm">超参数配置</h4>
                <p className="text-xs text-muted-foreground">覆盖默认配置文件中的参数</p>
              </div>
              <div className="grid gap-3">
                <div className="grid grid-cols-3 items-center gap-2">
                  <Label className="text-xs">学习率</Label>
                  <Input type="number" value={hyperParams.lr} step="0.0001" min="0.00001"
                    onChange={(e) => onHyperParamsChange({ ...hyperParams, lr: parseFloat(e.target.value) || 0.001 })}
                    className="col-span-2 h-8 text-xs font-mono" />
                </div>
                <div className="grid grid-cols-3 items-center gap-2">
                  <Label className="text-xs">迭代次数</Label>
                  <Input type="number" value={hyperParams.epochs} min="1"
                    onChange={(e) => onHyperParamsChange({ ...hyperParams, epochs: parseInt(e.target.value) || 100 })}
                    className="col-span-2 h-8 text-xs font-mono" />
                </div>
                <div className="grid grid-cols-3 items-center gap-2">
                  <Label className="text-xs">批量大小</Label>
                  <Input type="number" value={hyperParams.batchSize} min="1"
                    onChange={(e) => onHyperParamsChange({ ...hyperParams, batchSize: parseInt(e.target.value) || 64 })}
                    className="col-span-2 h-8 text-xs font-mono" />
                </div>
              </div>
            </div>
          </PopoverContent>
        </Popover>

        {onImportModel && (
          <Button variant="ghost" size="icon" className="h-8 w-8"
            onClick={onImportModel} title="导入模型">
            <Upload className="h-3.5 w-3.5" />
          </Button>
        )}
      </div>

      <Separator orientation="vertical" className="h-6" />

      {/* Group 3: 格式转换 */}
      <div className="flex items-center gap-1 px-2">
        <Select value={selectedFormat} onValueChange={onFormatChange} disabled={isTraining}>
          <SelectTrigger className="w-[130px] h-8 text-xs">
            <SelectValue placeholder="格式转换" />
          </SelectTrigger>
          <SelectContent>
            {formatOptions.map((o) => (
              <SelectItem key={o.value} value={o.value} className="text-xs">{o.label}</SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button variant="outline" size="sm" className="h-8 text-xs"
          disabled={isTraining} onClick={onApplyFormat}>应用</Button>
      </div>

      <Separator orientation="vertical" className="h-6" />

      {/* Group 4: 噪声类型 */}
      <div className="flex items-center gap-1 px-2">
        <Label className="text-xs text-muted-foreground whitespace-nowrap">噪声类型</Label>
        <Select value={selectedNoise} onValueChange={onNoiseChange} disabled={isTraining}>
          <SelectTrigger className="w-[130px] h-8 text-xs">
            <SelectValue placeholder="选择噪声" />
          </SelectTrigger>
          <SelectContent>
            {noiseOptions.map((o) => (
              <SelectItem key={o.value} value={o.value} className="text-xs">{o.label}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <Separator orientation="vertical" className="h-6" />

      {/* Group 5: 数据划分 */}
      <div className="flex items-center gap-2 px-2">
        <Select value={splitParams.mode}
          onValueChange={(v) => onSplitParamsChange({ ...splitParams, mode: v as SplitParams["mode"] })}
          disabled={isTraining || isSplitting}>
          <SelectTrigger className="w-[110px] h-8 text-xs"><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value="per_class" className="text-xs">每类固定数量</SelectItem>
            <SelectItem value="percent"   className="text-xs">百分比划分</SelectItem>
          </SelectContent>
        </Select>

        <div className="flex items-center gap-1">
          <Input type="number" value={splitParams.value}
            min={1} max={splitParams.mode === "percent" ? 99 : 9999}
            onChange={(e) => onSplitParamsChange({ ...splitParams, value: parseInt(e.target.value) || 10 })}
            disabled={isTraining || isSplitting}
            className="w-16 h-8 text-xs font-mono text-center" />
          <span className="text-xs text-muted-foreground">
            {splitParams.mode === "percent" ? "%" : "张/类"}
          </span>
        </div>

        <Button variant="outline" size="sm" className="h-8 text-xs gap-1.5"
          onClick={onSplit} disabled={isTraining || isSplitting}>
          {isSplitting
            ? <><Loader2 className="h-3.5 w-3.5 animate-spin" />划分中</>
            : <><Scissors className="h-3.5 w-3.5" />拆分</>}
        </Button>

        {splitStatus === "done"   && <Badge variant="secondary"   className="text-xs text-emerald-400">划分完成</Badge>}
        {splitStatus === "failed" && <Badge variant="destructive" className="text-xs">划分失败</Badge>}
      </div>

      <Separator orientation="vertical" className="h-6" />

      {/* Group 6: 操作按钮 */}
      <div className="flex items-center gap-1 px-2">
        <Button size="sm" onClick={onTrain} disabled={isTraining} className="gap-1.5 h-8 min-w-[72px]">
          {isTraining
            ? <><Loader2 className="h-3.5 w-3.5 animate-spin" />训练中</>
            : <><Play className="h-3.5 w-3.5" />训练</>}
        </Button>
        {!isTraining && onBatchTrain && (
          <Button variant="outline" size="sm" onClick={onBatchTrain} className="gap-1.5 h-8">
            <Play className="h-3.5 w-3.5" />批量
          </Button>
        )}
        {isTraining && onCancel && (
          <Button variant="destructive" size="sm" onClick={onCancel} className="gap-1.5 h-8">
            <Square className="h-3.5 w-3.5" />终止
          </Button>
        )}
        <Button variant="outline" size="sm" onClick={onEvaluate}
          disabled={isTraining || !hasDone} className="gap-1.5 h-8">
          <BarChart3 className="h-3.5 w-3.5" />评估
        </Button>
        <Button variant="outline" size="sm" onClick={onExport}
          disabled={isTraining || !hasDone} className="gap-1.5 h-8">
          <Download className="h-3.5 w-3.5" />导出
        </Button>
      </div>

      {isTraining && trainMessage && (
        <>
          <Separator orientation="vertical" className="h-6" />
          <span className="text-xs text-blue-400 font-mono px-2 animate-pulse">{trainMessage}</span>
        </>
      )}
    </div>
  )
}
