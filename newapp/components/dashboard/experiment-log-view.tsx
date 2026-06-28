"use client"

import { useEffect, useState, useCallback } from "react"
import { Trash2, RefreshCw } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table"

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000"

interface ExperimentRecord {
  job_id: string
  time: string
  dataset: string
  model: string
  train_sign: string
  noise_type: string
  epochs: number
  oa: number
  aa: number
  kappa: number
  eval_time: number
  config: string
}

const TRAIN_SIGN_LABEL: Record<string, string> = {
  train: "训练",
  test:  "测试",
  tent:  "TENT",
  ctent: "C-TENT",
}

const NOISE_LABEL: Record<string, string> = {
  clean:       "无噪声",
  jpeg:        "JPEG",
  additive:    "加性",
  poisson:     "泊松",
  salt_pepper: "椒盐",
  stripes:     "条纹",
  deadlines:   "死线",
  kernal:      "模糊",
  thick_fog:   "浓雾",
}

export function ExperimentLogView() {
  const [records, setRecords] = useState<ExperimentRecord[]>([])
  const [loading, setLoading] = useState(false)

  const fetchRecords = useCallback(async () => {
    setLoading(true)
    try {
      const res = await fetch(`${BACKEND_URL}/experiments`)
      const data = await res.json()
      setRecords(data.records ?? [])
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchRecords() }, [fetchRecords])

  const handleDelete = async (jobId: string) => {
    await fetch(`${BACKEND_URL}/experiments/${jobId}`, { method: "DELETE" })
    fetchRecords()
  }

  const handleClearAll = async () => {
    if (!confirm("确认清空所有实验记录？")) return
    await fetch(`${BACKEND_URL}/experiments`, { method: "DELETE" })
    fetchRecords()
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* 标题栏 */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-card">
        <span className="text-xs font-medium text-muted-foreground">
          共 {records.length} 条记录
        </span>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="icon" className="h-7 w-7"
            onClick={fetchRecords} disabled={loading}>
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
          </Button>
          <Button variant="ghost" size="icon" className="h-7 w-7 text-destructive"
            onClick={handleClearAll} disabled={records.length === 0}>
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      {/* 表格 */}
      <div className="flex-1 overflow-auto">
        {records.length === 0 ? (
          <div className="flex items-center justify-center h-full text-xs text-muted-foreground">
            暂无实验记录，完成训练后自动记录
          </div>
        ) : (
          <Table>
            <TableHeader className="sticky top-0 bg-card z-10">
              <TableRow>
                <TableHead className="text-xs w-36">时间</TableHead>
                <TableHead className="text-xs">数据集</TableHead>
                <TableHead className="text-xs">模型</TableHead>
                <TableHead className="text-xs">模式</TableHead>
                <TableHead className="text-xs">噪声</TableHead>
                <TableHead className="text-xs text-right">OA%</TableHead>
                <TableHead className="text-xs text-right">AA%</TableHead>
                <TableHead className="text-xs text-right">Kappa%</TableHead>
                <TableHead className="text-xs text-right">耗时(s)</TableHead>
                <TableHead className="text-xs w-8"></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {records.map((r) => (
                <TableRow key={r.job_id} className="hover:bg-muted/30">
                  <TableCell className="text-[11px] font-mono text-muted-foreground">
                    {r.time}
                  </TableCell>
                  <TableCell className="text-xs">
                    <Badge variant="outline" className="text-[10px] font-mono">
                      {r.dataset}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-xs font-mono">{r.model}</TableCell>
                  <TableCell className="text-xs">
                    <Badge
                      variant={r.train_sign === "ctent" ? "default" : "secondary"}
                      className="text-[10px]">
                      {TRAIN_SIGN_LABEL[r.train_sign] ?? r.train_sign}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">
                    {NOISE_LABEL[r.noise_type] ?? r.noise_type}
                  </TableCell>
                  <TableCell className={`text-xs font-mono text-right ${
                    r.oa > 90 ? "text-emerald-400" : r.oa > 80 ? "text-yellow-400" : "text-rose-400"
                  }`}>
                    {r.oa.toFixed(2)}
                  </TableCell>
                  <TableCell className="text-xs font-mono text-right">
                    {r.aa.toFixed(2)}
                  </TableCell>
                  <TableCell className="text-xs font-mono text-right">
                    {r.kappa.toFixed(2)}
                  </TableCell>
                  <TableCell className="text-xs font-mono text-right text-muted-foreground">
                    {r.eval_time}
                  </TableCell>
                  <TableCell>
                    <Button variant="ghost" size="icon" className="h-6 w-6 text-muted-foreground hover:text-destructive"
                      onClick={() => handleDelete(r.job_id)}>
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </div>
    </div>
  )
}
