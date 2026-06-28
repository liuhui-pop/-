"use client"

import { useMemo } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface StatisticsViewProps {
  data: Record<string, unknown>[]
}

interface ColumnStats {
  column: string
  type: "number" | "string" | "mixed"
  count: number
  nullCount: number
  unique: number
  mean?: number
  median?: number
  std?: number
  min?: number
  max?: number
}

export function StatisticsView({ data }: StatisticsViewProps) {
  const stats = useMemo(() => {
    if (data.length === 0) return []

    const columns = Object.keys(data[0])
    const result: ColumnStats[] = []

    for (const column of columns) {
      const values = data.map((row) => row[column])
      const nonNullValues = values.filter((v) => v !== null && v !== undefined)
      const numericValues = nonNullValues.filter((v) => typeof v === "number") as number[]

      const stat: ColumnStats = {
        column,
        type: numericValues.length === nonNullValues.length && numericValues.length > 0 ? "number" : "string",
        count: data.length,
        nullCount: values.filter((v) => v === null || v === undefined).length,
        unique: new Set(nonNullValues.map(String)).size,
      }

      if (stat.type === "number" && numericValues.length > 0) {
        stat.mean = numericValues.reduce((a, b) => a + b, 0) / numericValues.length
        stat.min = Math.min(...numericValues)
        stat.max = Math.max(...numericValues)

        const sorted = [...numericValues].sort((a, b) => a - b)
        const mid = Math.floor(sorted.length / 2)
        stat.median = sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]

        const variance = numericValues.reduce((sum, val) => sum + Math.pow(val - stat.mean!, 2), 0) / numericValues.length
        stat.std = Math.sqrt(variance)
      }

      result.push(stat)
    }

    return result
  }, [data])

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground text-sm">
        暂无数据，请先导入数据集
      </div>
    )
  }

  return (
    <div className="p-4 grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 overflow-auto">
      {stats.map((stat) => (
        <Card key={stat.column} className="bg-muted/30">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-mono truncate">{stat.column}</CardTitle>
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
              {stat.type === "number" ? "数值型" : "字符型"}
            </p>
          </CardHeader>
          <CardContent className="grid grid-cols-2 gap-2 text-xs">
            {stat.type === "number" ? (
              <>
                <StatItem label="均值" value={stat.mean?.toFixed(2)} />
                <StatItem label="中位数" value={stat.median?.toFixed(2)} />
                <StatItem label="标准差" value={stat.std?.toFixed(2)} />
                <StatItem label="最小值" value={stat.min?.toFixed(2)} />
                <StatItem label="最大值" value={stat.max?.toFixed(2)} />
                <StatItem label="空值数" value={stat.nullCount} />
              </>
            ) : (
              <>
                <StatItem label="计数" value={stat.count} />
                <StatItem label="唯一值" value={stat.unique} />
                <StatItem label="空值数" value={stat.nullCount} />
              </>
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  )
}

function StatItem({ label, value }: { label: string; value?: string | number }) {
  return (
    <div>
      <p className="text-[10px] text-muted-foreground">{label}</p>
      <p className="font-mono font-medium">{value ?? "-"}</p>
    </div>
  )
}
