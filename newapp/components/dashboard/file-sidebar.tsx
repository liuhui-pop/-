"use client"

import { FileSpreadsheet, FileJson, FileText, Cpu, Plus, Trash2, X, Download } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger,
} from "@/components/ui/context-menu"
import { cn } from "@/lib/utils"
import type { FileItem } from "@/lib/mock-data"

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000"

interface FileSidebarProps {
  files: FileItem[]
  activeFileId: string | null
  onFileSelect: (id: string) => void
  onFileRemove: (id: string) => void
  onAddFile: () => void
  onClearAll: () => void
}

const fileIcons: Record<string, React.ElementType> = {
  csv:   FileSpreadsheet,
  json:  FileJson,
  xlsx:  FileSpreadsheet,
  model: Cpu,
  txt:   FileText,
  mat:   FileSpreadsheet,
}

const fileColors: Record<string, string> = {
  csv:   "text-emerald-400",
  json:  "text-amber-400",
  xlsx:  "text-blue-400",
  model: "text-purple-400",
  txt:   "text-slate-400",
  mat:   "text-orange-400",
}

function handleDownload(fileId: string, fileName: string) {
  const a = document.createElement("a")
  a.href = `${BACKEND_URL}/download/${fileId}`
  a.download = fileName
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
}

export function FileSidebar({
  files, activeFileId,
  onFileSelect, onFileRemove,
  onAddFile, onClearAll,
}: FileSidebarProps) {
  return (
    <div className="w-64 border-r border-border bg-sidebar flex flex-col">
      <div className="flex items-center justify-between p-3 border-b border-border">
        <h2 className="text-sm font-medium text-sidebar-foreground">打开的文件</h2>
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="icon"
            className="h-6 w-6 text-muted-foreground hover:text-foreground"
            onClick={onAddFile}>
            <Plus className="h-3.5 w-3.5" />
          </Button>
          <Button variant="ghost" size="icon"
            className="h-6 w-6 text-muted-foreground hover:text-destructive"
            onClick={onClearAll}>
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      <ScrollArea className="flex-1">
        {files.length > 0 ? (
          <div className="p-2 space-y-1">
            {files.map((file) => {
              const Icon = fileIcons[file.type] ?? FileSpreadsheet
              const isActive = activeFileId === file.id

              return (
                <ContextMenu key={file.id}>
                  <ContextMenuTrigger>
                    <div
                      className={cn(
                        "group flex items-center gap-2 px-2 py-1.5 rounded-md cursor-pointer transition-colors",
                        isActive
                          ? "bg-sidebar-accent text-sidebar-accent-foreground ring-1 ring-primary/50"
                          : "hover:bg-sidebar-accent/50"
                      )}
                      onClick={() => onFileSelect(file.id)}
                    >
                      <Icon className={cn("h-4 w-4 shrink-0", fileColors[file.type] ?? "text-slate-400")} />
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-mono truncate">{file.name}</p>
                        <p className="text-[10px] text-muted-foreground">{file.size}</p>
                      </div>
                      <Button
                        variant="ghost" size="icon"
                        className="h-5 w-5 opacity-0 group-hover:opacity-100 transition-opacity"
                        onClick={(e) => { e.stopPropagation(); onFileRemove(file.id) }}
                      >
                        <X className="h-3 w-3" />
                      </Button>
                    </div>
                  </ContextMenuTrigger>

                  {/* 右键菜单 */}
                  <ContextMenuContent className="w-48">
                    <ContextMenuItem
                      className="gap-2 text-xs cursor-pointer"
                      onClick={() => handleDownload(file.id, file.name)}
                    >
                      <Download className="h-3.5 w-3.5" />
                      保存到本地
                    </ContextMenuItem>
                    <ContextMenuItem
                      className="gap-2 text-xs cursor-pointer text-destructive focus:text-destructive"
                      onClick={() => onFileRemove(file.id)}
                    >
                      <X className="h-3.5 w-3.5" />
                      从列表移除
                    </ContextMenuItem>
                  </ContextMenuContent>
                </ContextMenu>
              )
            })}
          </div>
        ) : (
          <div className="p-4">
            <div className="border-2 border-dashed border-border rounded-lg p-6 text-center">
              <FileSpreadsheet className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
              <p className="text-xs text-muted-foreground">暂无打开的文件</p>
              <p className="text-[10px] text-muted-foreground mt-1">导入数据以开始</p>
            </div>
          </div>
        )}
      </ScrollArea>
    </div>
  )
}
