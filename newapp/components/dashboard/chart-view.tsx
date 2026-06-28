"use client"

import { useState, useMemo, useEffect } from "react"
import {
  BarChart, Bar, LineChart, Line, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from "recharts"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { FileItem } from "@/lib/mock-data"

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000"

interface ChartViewProps {
  data: Record<string, unknown>[]
  activeFile?: FileItem | null
}

type ChartType = "bar" | "line" | "scatter" | "histogram"

const COLORS = [
  "hsl(var(--chart-1))", "hsl(var(--chart-2))", "hsl(var(--chart-3))",
  "hsl(var(--chart-4))", "hsl(var(--chart-5))",
]

interface MatPreview {
  file_id: string
  meta: {
    height: number
    width: number
    bands: number
    n_classes: number | null
    file_name: string
    r_band: number
    g_band: number
    b_band: number
  }
  rgb_image: string
  spectral_image: string
  gt_image: string | null
}

// ═════════════════════════════════════════════════════════════
// Mat 可视化子组件
// ═════════════════════════════════════════════════════════════
function MatPreviewPanel({ fileId }: { fileId: string }) {
  const [preview, setPreview] = useState<MatPreview | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // 波段选择输入框状态（用字符串方便输入中间状态）
  const [rInput, setRInput] = useState("")
  const [gInput, setGInput] = useState("")
  const [bInput, setBInput] = useState("")

  const fetchPreview = (r = -1, g = -1, b = -1) => {
    setLoading(true)
    setError(null)
    fetch(`${BACKEND_URL}/preview/${fileId}?r_band=${r}&g_band=${g}&b_band=${b}`)
      .then((res) => {
        if (!res.ok) return res.json().then((e) => { throw new Error(e.detail) })
        return res.json()
      })
      .then((data: MatPreview) => {
        setPreview(data)
        // 回填实际使用的波段号
        setRInput(String(data.meta.r_band))
        setGInput(String(data.meta.g_band))
        setBInput(String(data.meta.b_band))
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }

  // 首次加载
  useEffect(() => {
    setPreview(null)
    setError(null)
    setRInput("")
    setGInput("")
    setBInput("")
    fetchPreview()
  }, [fileId])

  const handleApply = () => {
    const maxBand = (preview?.meta.bands ?? 1) - 1
    const r = Math.min(Math.max(0, parseInt(rInput) || 0), maxBand)
    const g = Math.min(Math.max(0, parseInt(gInput) || 0), maxBand)
    const b = Math.min(Math.max(0, parseInt(bInput) || 0), maxBand)
    fetchPreview(r, g, b)
  }

  if (loading && !preview) {
    return (
      <div className="flex items-center justify-center h-full text-sm text-blue-400 animate-pulse">
        正在解析 mat 文件...
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full text-sm text-rose-400">
        解析失败：{error}
      </div>
    )
  }

  if (!preview) return null

  const { meta, rgb_image, spectral_image, gt_image } = preview
  const maxBand = meta.bands - 1

  return (
    <div className="p-4 space-y-4 overflow-auto h-full">

      {/* 数据基本信息 + 波段选择 */}
      <div className="flex flex-wrap items-center gap-4">
        {/* 信息标签 */}
        <div className="flex gap-3 text-xs font-mono text-muted-foreground">
          <span>{meta.height} × {meta.width}</span>
          <span>波段: {meta.bands}</span>
          {meta.n_classes != null && <span>类别: {meta.n_classes}</span>}
          <span className="text-blue-400">{meta.file_name}</span>
        </div>

        {/* 波段选择控件 */}
        <div className="flex items-center gap-2 ml-auto">
          <span className="text-xs text-muted-foreground">波段选择 (0–{maxBand})</span>
          {[
            { label: "R", value: rInput, setter: setRInput, color: "text-rose-400" },
            { label: "G", value: gInput, setter: setGInput, color: "text-emerald-400" },
            { label: "B", value: bInput, setter: setBInput, color: "text-blue-400" },
          ].map(({ label, value, setter, color }) => (
            <div key={label} className="flex items-center gap-1">
              <span className={`text-xs font-bold font-mono ${color}`}>{label}</span>
              <Input
                type="number"
                min={0}
                max={maxBand}
                value={value}
                onChange={(e) => setter(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleApply()}
                className="w-16 h-7 text-xs font-mono text-center"
              />
            </div>
          ))}
          <Button
            size="sm"
            variant="outline"
            className="h-7 text-xs"
            onClick={handleApply}
            disabled={loading}
          >
            {loading ? "更新中..." : "应用"}
          </Button>
        </div>
      </div>

      {/* 三张图 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card className="bg-muted/30">
          <CardHeader className="pb-1 pt-3 px-3">
            <CardTitle className="text-xs">
              HSI 伪彩色图
              <span className="ml-2 font-mono text-muted-foreground">
                R:{meta.r_band} G:{meta.g_band} B:{meta.b_band}
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent className="p-2">
            <img
              key={rgb_image.slice(-20)}
              src={`data:image/png;base64,${rgb_image}`}
              alt="HSI RGB"
              className="w-full rounded-md object-contain"
            />
          </CardContent>
        </Card>

        <Card className="bg-muted/30">
          <CardHeader className="pb-1 pt-3 px-3">
            <CardTitle className="text-xs">光谱曲线图</CardTitle>
          </CardHeader>
          <CardContent className="p-2">
            <img
              key={spectral_image.slice(-20)}
              src={`data:image/png;base64,${spectral_image}`}
              alt="Spectral Curve"
              className="w-full rounded-md object-contain"
            />
          </CardContent>
        </Card>

        <Card className="bg-muted/30">
          <CardHeader className="pb-1 pt-3 px-3">
            <CardTitle className="text-xs">GT 标签图</CardTitle>
          </CardHeader>
          <CardContent className="p-2">
            {gt_image ? (
              <img
                key={gt_image.slice(-20)}
                src={`data:image/png;base64,${gt_image}`}
                alt="GT Labels"
                className="w-full rounded-md object-contain"
              />
            ) : (
              <div className="flex items-center justify-center h-32 text-xs text-muted-foreground">
                未检测到 GT 标签数组
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

// ═════════════════════════════════════════════════════════════
// 主组件
// ═════════════════════════════════════════════════════════════
export function ChartView({ data, activeFile }: ChartViewProps) {
  // mat 文件走 HSI 可视化
  if (activeFile?.type === "mat") {
    return <MatPreviewPanel fileId={activeFile.id} />
  }

  // ── 以下原有普通图表逻辑完全不变 ─────────────────────────────

  const columns = useMemo(() => {
    if (data.length === 0) return []
    return Object.keys(data[0])
  }, [data])

  const numericColumns = useMemo(() => {
    if (data.length === 0) return []
    return columns.filter((col) => typeof data[0][col] === "number")
  }, [data, columns])

  const [xAxis, setXAxis] = useState<string>(columns[0] || "")
  const [yAxis, setYAxis] = useState<string>(numericColumns[0] || "")
  const [chartType, setChartType] = useState<ChartType>("bar")

  const chartData = useMemo(() => {
    if (chartType === "histogram" && yAxis) {
      const values = data.map((d) => d[yAxis] as number).filter((v) => typeof v === "number")
      const min = Math.min(...values)
      const max = Math.max(...values)
      const binCount = 10
      const binSize = (max - min) / binCount
      const bins: { range: string; count: number }[] = []
      for (let i = 0; i < binCount; i++) {
        const binMin = min + i * binSize
        const binMax = binMin + binSize
        const count = values.filter((v) =>
          v >= binMin && (i === binCount - 1 ? v <= binMax : v < binMax)
        ).length
        bins.push({ range: `${binMin.toFixed(0)}-${binMax.toFixed(0)}`, count })
      }
      return bins
    }
    return data.slice(0, 50)
  }, [data, chartType, yAxis])

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground text-sm">
        暂无数据，请先导入数据集
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-4 p-4 border-b border-border">
        <div className="flex items-center gap-2">
          <Label className="text-xs text-muted-foreground">X轴</Label>
          <Select value={xAxis} onValueChange={setXAxis}>
            <SelectTrigger className="w-[140px] h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              {columns.map((col) => (
                <SelectItem key={col} value={col} className="text-xs font-mono">{col}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="flex items-center gap-2">
          <Label className="text-xs text-muted-foreground">Y轴</Label>
          <Select value={yAxis} onValueChange={setYAxis}>
            <SelectTrigger className="w-[140px] h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              {numericColumns.map((col) => (
                <SelectItem key={col} value={col} className="text-xs font-mono">{col}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="flex items-center gap-2">
          <Label className="text-xs text-muted-foreground">图表类型</Label>
          <Select value={chartType} onValueChange={(v) => setChartType(v as ChartType)}>
            <SelectTrigger className="w-[120px] h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="bar" className="text-xs">柱状图</SelectItem>
              <SelectItem value="line" className="text-xs">折线图</SelectItem>
              <SelectItem value="scatter" className="text-xs">散点图</SelectItem>
              <SelectItem value="histogram" className="text-xs">直方图</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="flex-1 p-4">
        <ResponsiveContainer width="100%" height="100%">
          {chartType === "bar" ? (
            <BarChart data={chartData as Record<string, unknown>[]}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey={xAxis} tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }} />
              <YAxis tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }} />
              <Tooltip contentStyle={{ backgroundColor: "hsl(var(--popover))", border: "1px solid hsl(var(--border))", borderRadius: "6px", fontSize: "12px" }} />
              <Bar dataKey={yAxis} fill="hsl(var(--chart-1))" radius={[4, 4, 0, 0]} />
            </BarChart>
          ) : chartType === "line" ? (
            <LineChart data={chartData as Record<string, unknown>[]}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey={xAxis} tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }} />
              <YAxis tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }} />
              <Tooltip contentStyle={{ backgroundColor: "hsl(var(--popover))", border: "1px solid hsl(var(--border))", borderRadius: "6px", fontSize: "12px" }} />
              <Line type="monotone" dataKey={yAxis} stroke="hsl(var(--chart-1))" strokeWidth={2} dot={{ fill: "hsl(var(--chart-1))", r: 4 }} />
            </LineChart>
          ) : chartType === "scatter" ? (
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey={xAxis} type="number" name={xAxis} tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }} />
              <YAxis dataKey={yAxis} type="number" name={yAxis} tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }} />
              <Tooltip contentStyle={{ backgroundColor: "hsl(var(--popover))", border: "1px solid hsl(var(--border))", borderRadius: "6px", fontSize: "12px" }} />
              <Scatter data={chartData as Record<string, unknown>[]} fill="hsl(var(--chart-1))">
                {(chartData as Record<string, unknown>[]).map((_, index) => (
                  <Cell key={index} fill={COLORS[index % COLORS.length]} />
                ))}
              </Scatter>
            </ScatterChart>
          ) : (
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey="range" tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }} />
              <YAxis tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }} />
              <Tooltip contentStyle={{ backgroundColor: "hsl(var(--popover))", border: "1px solid hsl(var(--border))", borderRadius: "6px", fontSize: "12px" }} />
              <Bar dataKey="count" fill="hsl(var(--chart-2))" radius={[4, 4, 0, 0]} />
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  )
}
