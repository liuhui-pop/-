"use client"

import { useState, useMemo, useEffect } from "react"
import { ArrowUpDown, ChevronLeft, ChevronRight } from "lucide-react"
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { FileItem } from "@/lib/mock-data"

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000"

interface DataTableProps {
  data: Record<string, unknown>[]
  activeFile?: FileItem | null
}

// ─────────────────────────────────────────────────────────────
// Mat 可视化面板（从 chart-view 移过来）
// ─────────────────────────────────────────────────────────────
interface MatPreview {
  file_id: string
  meta: {
    height: number; width: number; bands: number
    n_classes: number | null; file_name: string
    r_band: number; g_band: number; b_band: number
  }
  rgb_image: string
  spectral_image: string
  gt_image: string | null
}

function MatPreviewPanel({ fileId }: { fileId: string }) {
  const [preview, setPreview] = useState<MatPreview | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
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
        setRInput(String(data.meta.r_band))
        setGInput(String(data.meta.g_band))
        setBInput(String(data.meta.b_band))
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    setPreview(null)
    setError(null)
    setRInput(""); setGInput(""); setBInput("")
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
      {/* 信息行 + 波段选择 */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex gap-3 text-xs font-mono text-muted-foreground">
          <span>{meta.height} × {meta.width}</span>
          <span>波段: {meta.bands}</span>
          {meta.n_classes != null && <span>类别: {meta.n_classes}</span>}
          <span className="text-blue-400">{meta.file_name}</span>
        </div>
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
                type="number" min={0} max={maxBand}
                value={value}
                onChange={(e) => setter(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleApply()}
                className="w-16 h-7 text-xs font-mono text-center"
              />
            </div>
          ))}
          <Button size="sm" variant="outline" className="h-7 text-xs"
            onClick={handleApply} disabled={loading}>
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
            <img key={rgb_image.slice(-20)}
              src={`data:image/png;base64,${rgb_image}`}
              alt="HSI RGB" className="w-full rounded-md object-contain" />
          </CardContent>
        </Card>

        <Card className="bg-muted/30">
          <CardHeader className="pb-1 pt-3 px-3">
            <CardTitle className="text-xs">光谱曲线图</CardTitle>
          </CardHeader>
          <CardContent className="p-2">
            <img key={spectral_image.slice(-20)}
              src={`data:image/png;base64,${spectral_image}`}
              alt="Spectral Curve" className="w-full rounded-md object-contain" />
          </CardContent>
        </Card>

        <Card className="bg-muted/30">
          <CardHeader className="pb-1 pt-3 px-3">
            <CardTitle className="text-xs">GT 标签图</CardTitle>
          </CardHeader>
          <CardContent className="p-2">
            {gt_image ? (
              <img key={gt_image.slice(-20)}
                src={`data:image/png;base64,${gt_image}`}
                alt="GT Labels" className="w-full rounded-md object-contain" />
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

// ─────────────────────────────────────────────────────────────
// 主组件
// ─────────────────────────────────────────────────────────────
export function DataTable({ data, activeFile }: DataTableProps) {
  // mat 文件走 HSI 可视化
  if (activeFile?.type === "mat") {
    return <MatPreviewPanel fileId={activeFile.id} />
  }

  // ── 以下原有表格逻辑完全不变 ──────────────────────────────────
  const [page, setPage] = useState(0)
  const [sortColumn, setSortColumn] = useState<string | null>(null)
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc")
  const pageSize = 10

  const columns = useMemo(() => {
    if (data.length === 0) return []
    return Object.keys(data[0])
  }, [data])

  const sortedData = useMemo(() => {
    if (!sortColumn) return data
    return [...data].sort((a, b) => {
      const aVal = a[sortColumn]
      const bVal = b[sortColumn]
      if (aVal === bVal) return 0
      if (aVal === null || aVal === undefined) return 1
      if (bVal === null || bVal === undefined) return -1
      const comparison = aVal < bVal ? -1 : 1
      return sortDirection === "asc" ? comparison : -comparison
    })
  }, [data, sortColumn, sortDirection])

  const paginatedData = useMemo(() => {
    const start = page * pageSize
    return sortedData.slice(start, start + pageSize)
  }, [sortedData, page])

  const totalPages = Math.ceil(data.length / pageSize)

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc")
    } else {
      setSortColumn(column)
      setSortDirection("asc")
    }
  }

  const formatValue = (value: unknown) => {
    if (value === null || value === undefined)
      return <span className="text-muted-foreground">null</span>
    if (typeof value === "number")
      return <span className="font-mono">{value.toLocaleString()}</span>
    if (typeof value === "boolean")
      return <Badge variant={value ? "default" : "secondary"}>{value ? "true" : "false"}</Badge>
    return String(value)
  }

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground text-sm">
        暂无数据，请先导入数据集
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-auto">
        <Table>
          <TableHeader className="sticky top-0 bg-card">
            <TableRow>
              {columns.map((column) => (
                <TableHead key={column}
                  className="cursor-pointer hover:bg-muted/50 transition-colors"
                  onClick={() => handleSort(column)}>
                  <div className="flex items-center gap-1">
                    <span className="font-mono text-xs">{column}</span>
                    <ArrowUpDown className="h-3 w-3" />
                  </div>
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {paginatedData.map((row, idx) => (
              <TableRow key={idx}>
                {columns.map((column) => (
                  <TableCell key={column} className="text-xs py-2">
                    {formatValue(row[column])}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
      <div className="flex items-center justify-between p-3 border-t border-border bg-card">
        <p className="text-xs text-muted-foreground">
          共 <span className="font-mono font-medium text-foreground">{data.length}</span> 行
        </p>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="icon" className="h-7 w-7"
            disabled={page === 0} onClick={() => setPage(page - 1)}>
            <ChevronLeft className="h-3.5 w-3.5" />
          </Button>
          <span className="text-xs text-muted-foreground">{page + 1} / {totalPages}</span>
          <Button variant="outline" size="icon" className="h-7 w-7"
            disabled={page >= totalPages - 1} onClick={() => setPage(page + 1)}>
            <ChevronRight className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>
    </div>
  )
}
