"use client"

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { DataTable } from "./data-table"
import { ChartView } from "./chart-view"
import { StatisticsView } from "./statistics-view"
import { ModelOutputView } from "./model-output-view"
import { TableIcon, BarChart3, PieChart, Cpu, ClipboardList } from "lucide-react"
import { ExperimentLogView } from "./experiment-log-view"
import type { TrainState } from "@/app/page"
import type { FileItem } from "@/lib/mock-data"

interface MainViewProps {
  data: Record<string, unknown>[]
  activeTab: string
  onTabChange: (tab: string) => void
  trainState?: TrainState
  activeFile?: FileItem | null
  onTrainDone?: () => void  // 训练完成后通知实验记录刷新
}

export function MainView({ data, activeTab, onTabChange, trainState, activeFile, onTrainDone }: MainViewProps) {
  return (
    <div className="flex-1 flex flex-col bg-background overflow-auto">
      <Tabs value={activeTab} onValueChange={onTabChange} className="flex-1 flex flex-col overflow-auto">
        <div className="border-b border-border bg-card">
          <TabsList className="h-10 bg-transparent rounded-none p-0 gap-0">
            <TabsTrigger
              value="table"
              className="data-[state=active]:bg-background data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none px-4 gap-1.5"
            >
              <TableIcon className="h-3.5 w-3.5" />
              数据预览
            </TabsTrigger>
            <TabsTrigger
              value="chart"
              className="data-[state=active]:bg-background data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none px-4 gap-1.5"
            >
              <BarChart3 className="h-3.5 w-3.5" />
              图表
            </TabsTrigger>
            <TabsTrigger
              value="statistics"
              className="data-[state=active]:bg-background data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none px-4 gap-1.5"
            >
              <PieChart className="h-3.5 w-3.5" />
              统计
            </TabsTrigger>
            <TabsTrigger
              value="model"
              className="data-[state=active]:bg-background data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none px-4 gap-1.5"
            >
              <Cpu className="h-3.5 w-3.5" />
              模型输出
              {/* 训练中在 Tab 标题上显示小圆点 */}
              {(trainState?.status === "pending" || trainState?.status === "running") && (
                <span className="ml-1 h-1.5 w-1.5 rounded-full bg-blue-400 animate-pulse inline-block" />
              )}
              {trainState?.status === "done" && (
                <span className="ml-1 h-1.5 w-1.5 rounded-full bg-emerald-400 inline-block" />
              )}
              {trainState?.status === "failed" && (
                <span className="ml-1 h-1.5 w-1.5 rounded-full bg-rose-400 inline-block" />
              )}
            </TabsTrigger>
            <TabsTrigger
              value="experiments"
              className="data-[state=active]:bg-background data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none px-4 gap-1.5"
            >
              <ClipboardList className="h-3.5 w-3.5" />
              实验记录
            </TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="table" className="flex-1 m-0 overflow-hidden">
          <DataTable data={data} activeFile={activeFile} />
        </TabsContent>
        <TabsContent value="chart" className="flex-1 m-0 overflow-hidden">
          <ChartView data={data} />
        </TabsContent>
        <TabsContent value="statistics" className="flex-1 m-0 overflow-hidden">
          <StatisticsView data={data} />
        </TabsContent>
        <TabsContent value="model" className="flex-1 m-0 overflow-auto">
          {/* 将 trainState 传入，ModelOutputView 从这里取真实数据 */}
          <ModelOutputView trainState={trainState} />
        </TabsContent>
         <TabsContent value="experiments" className="flex-1 m-0 overflow-hidden">
          <ExperimentLogView />
        </TabsContent>
      </Tabs>
    </div>
  )
}
