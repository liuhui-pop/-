"use client"

import { useEffect, useRef, useState } from "react"
import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table"
import type { TrainState } from "@/app/page"

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000"

interface ModelOutputViewProps {
  trainState?: TrainState
}

interface EpochEvent {
  type: "epoch" | "done" | "error"
  epoch?: number
  loss?: number
  oa?: number
  aa?: number
  kappa?: number
  message?: string
}

interface VisualResult {
  metrics_bar: { name: string; value: number }[]
  class_acc: { class: string; accuracy: number }[]
  pred_image: string | null
}

// PaviaU 类别名
const PAVIA_NAMES = [
  "Asphalt","Meadows","Gravel","Trees",
  "Metal Sheets","Bare Soil","Bitumen","Bricks","Shadows",
]

export function ModelOutputView({ trainState }: ModelOutputViewProps) {
  const status  = trainState?.status  ?? "idle"
  const jobId   = trainState?.jobId   ?? null
  const message = trainState?.message ?? ""

  // ── 实时曲线数据 ────────────────────────────────────────────
  const [lossCurve, setLossCurve] = useState<{ epoch: number; loss: number }[]>([])
  const [oaCurve,   setOaCurve]   = useState<{ epoch: number; oa: number }[]>([])
  const [finalMetrics, setFinalMetrics] = useState<{ oa: number; aa: number; kappa: number } | null>(null)
  const [visualResult, setVisualResult] = useState<VisualResult | null>(null)
  const esRef = useRef<EventSource | null>(null)

  // ── SSE 连接：训练开始时建立，结束时关闭 ────────────────────
  useEffect(() => {
    if (status === "running" && jobId) {
      // 重置曲线
      setLossCurve([])
      setOaCurve([])
      setFinalMetrics(null)
      setVisualResult(null)

      const es = new EventSource(`${BACKEND_URL}/stream/${jobId}`)
      esRef.current = es

      es.onmessage = (e) => {
        try {
          const event: EpochEvent = JSON.parse(e.data)

          if (event.type === "epoch") {
            if (event.epoch != null && event.loss != null) {
              setLossCurve((prev) => [...prev, { epoch: event.epoch!, loss: event.loss! }])
            }
            if (event.epoch != null && event.oa != null) {
              setOaCurve((prev) => [...prev, { epoch: event.epoch!, oa: event.oa! }])
            }
          }

          if (event.type === "done") {
            setFinalMetrics({
              oa:    event.oa    ?? 0,
              aa:    event.aa    ?? 0,
              kappa: event.kappa ?? 0,
            })
            es.close()
            // 拉取完整可视化数据（含分类图和逐类精度）
            // fetch(`${BACKEND_URL}/visualize/${jobId}`)
            //   .then((r) => r.json())
            //   .then((d) => setVisualResult(d))
            //   .catch(console.error)
            setTimeout(() => {
              fetch(`${BACKEND_URL}/visualize/${jobId}`)
                .then((r) => r.json())
                .then((d) => setVisualResult(d))
                .catch(console.error)
            }, 2000)
          }

          if (event.type === "error") {
            es.close()
          }
        } catch (_) {}
      }

      es.onerror = () => es.close()
      return () => es.close()
    }
  }, [status, jobId])

  // ── 训练完成后若有 trainState.visualResult 也同步 ──────────
  useEffect(() => {
    if (status === "done" && trainState?.visualResult) {
      setVisualResult(trainState.visualResult as unknown as VisualResult)
    }
  }, [status, trainState?.visualResult])

  // ── 指标表格数据 ─────────────────────────────────────────────
  const metrics = finalMetrics
    ? [
        { label: "总体精度 (OA)",  value: finalMetrics.oa,    unit: "%" },
        { label: "平均精度 (AA)",  value: finalMetrics.aa,    unit: "%" },
        { label: "Kappa 系数",     value: finalMetrics.kappa, unit: "%" },
      ]
    : [
        { label: "总体精度 (OA)",  value: null, unit: "%" },
        { label: "平均精度 (AA)",  value: null, unit: "%" },
        { label: "Kappa 系数",     value: null, unit: "%" },
      ]

  return (
    <div className="p-4 space-y-4 overflow-auto h-full">

      {/* ── 状态横幅 ─────────────────────────────────────────── */}
      {status !== "idle" && (
        <div className={`px-4 py-2 rounded-md text-xs font-mono border ${
          status === "done"    ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400" :
          status === "failed"  ? "bg-rose-500/10 border-rose-500/30 text-rose-400" :
                                 "bg-blue-500/10 border-blue-500/30 text-blue-400"
        }`}>
          {status === "running" && <span className="mr-2 animate-pulse">●</span>}
          {message}
        </div>
      )}

      {/* ── Loss 曲线（实时更新）────────────────────────────────── */}
      <Card className="bg-muted/30">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">训练 Loss 曲线</CardTitle>
        </CardHeader>
        <CardContent className="h-[220px]">
          {lossCurve.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={lossCurve}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="epoch"
                  tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }}
                  label={{ value: "Epoch", position: "bottom",
                    fill: "hsl(var(--muted-foreground))", fontSize: 10 }} />
                <YAxis tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }} />
                <Tooltip contentStyle={{
                  backgroundColor: "hsl(var(--popover))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "6px", fontSize: "12px" }} />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                <Line type="monotone" dataKey="loss" name="训练Loss"
                  stroke="#3b82f6" strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-full text-xs text-muted-foreground">
              {status === "idle" ? "点击「训练」后显示实时 Loss 曲线" : "等待第一个 epoch..."}
            </div>
          )}
        </CardContent>
      </Card>

      {/* ── OA 曲线（每10epoch更新）──────────────────────────── */}
      {oaCurve.length > 0 && (
        <Card className="bg-muted/30">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">训练 OA 曲线</CardTitle>
          </CardHeader>
          <CardContent className="h-[180px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={oaCurve}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="epoch"
                  tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }} />
                <YAxis tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }}
                  tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                <Tooltip formatter={(v: number) => [`${(v * 100).toFixed(2)}%`, "OA"]}
                  contentStyle={{ backgroundColor: "hsl(var(--popover))",
                    border: "1px solid hsl(var(--border))", borderRadius: "6px", fontSize: "12px" }} />
                <Line type="monotone" dataKey="oa" name="OA"
                  stroke="#22c55e" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

        {/* ── 评估指标表格 ──────────────────────────────────────── */}
        <Card className="bg-muted/30">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">最终评估指标</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="text-xs">指标</TableHead>
                  <TableHead className="text-xs text-right">值</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {metrics.map(({ label, value, unit }) => (
                  <TableRow key={label}>
                    <TableCell className="text-xs">{label}</TableCell>
                    <TableCell className="text-xs font-mono text-right">
                      {value == null
                        ? <span className="text-muted-foreground">—</span>
                        : `${value.toFixed(2)}${unit}`}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            {!finalMetrics && status === "idle" && (
              <p className="text-[10px] text-muted-foreground mt-3 text-center">
                训练完成后显示真实指标
              </p>
            )}
          </CardContent>
        </Card>

        {/* ── 逐类精度柱状图 ────────────────────────────────────── */}
        <Card className="bg-muted/30">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">逐类精度</CardTitle>
          </CardHeader>
          <CardContent className="h-[220px]">
            {visualResult?.class_acc?.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={visualResult.class_acc} margin={{ bottom: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="class"
                    tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 9 }}
                    angle={-35} textAnchor="end" interval={0} />
                  <YAxis domain={[0, 100]}
                    tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }}
                    tickFormatter={(v) => `${v}%`} />
                  <Tooltip formatter={(v: number) => [`${v.toFixed(2)}%`, "精度"]}
                    contentStyle={{ backgroundColor: "hsl(var(--popover))",
                      border: "1px solid hsl(var(--border))", borderRadius: "6px", fontSize: "12px" }} />
                  <Bar dataKey="accuracy" name="精度" fill="#3b82f6" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full text-xs text-muted-foreground">
                训练完成后显示逐类精度
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* ── 分类结果图 ────────────────────────────────────────── */}
      {visualResult?.pred_image && (
        <Card className="bg-muted/30">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">分类结果图</CardTitle>
          </CardHeader>
          <CardContent className="p-2">
            <img
              src={`data:image/png;base64,${visualResult.pred_image}`}
              alt="Prediction Map"
              className="w-full max-w-lg mx-auto rounded-md object-contain"
            />
          </CardContent>
        </Card>
      )}

      {/* ── 混淆矩阵文本 ──────────────────────────────────────── */}
      {visualResult?.confusion_matrix_text && (
        <Card className="bg-muted/30">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">混淆矩阵</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="text-[10px] font-mono text-muted-foreground overflow-auto whitespace-pre-wrap leading-5">
              {visualResult.confusion_matrix_text}
            </pre>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
