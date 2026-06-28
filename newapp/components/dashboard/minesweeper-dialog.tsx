"use client"

import { useState, useEffect } from "react"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"

// 定义 Windows 经典扫雷的三种默认难度配置
const DIFFICULTIES = {
  beginner: { label: "初级 (9x9)", rows: 9, cols: 9, mines: 10, panelWidth: 490 },
  intermediate: { label: "中级 (16x16)", rows: 16, cols: 16, mines: 40, panelWidth: 700 },
  expert: { label: "高级 (16x30)", rows: 16, cols: 30, mines: 99, panelWidth: 1120 },
} as const

type DifficultyKey = keyof typeof DIFFICULTIES

// 配置你的 GIF 路径 (图片请放在项目的 public/images/ 目录下)
const WIN_GIF_PATH = "/images/win.gif"
const LOSE_GIF_PATH = "/images/lose.gif"

type Cell = {
  isMine: boolean
  isRevealed: boolean
  isFlagged: boolean
  neighborCount: number
}

function createEmptyBoard(rows: number, cols: number): Cell[][] {
  return Array(rows).fill(null).map(() =>
    Array(cols).fill(null).map(() => ({
      isMine: false, isRevealed: false, isFlagged: false, neighborCount: 0,
    }))
  )
}

function placeMinesAndCalculate(board: Cell[][], firstR: number, firstC: number, rows: number, cols: number, mines: number): Cell[][] {
  const newBoard = board.map(row => row.map(cell => ({ ...cell })))
  let placed = 0
  while (placed < mines) {
    const r = Math.floor(Math.random() * rows)
    const c = Math.floor(Math.random() * cols)
    if (!newBoard[r][c].isMine && (r !== firstR || c !== firstC)) { 
      newBoard[r][c].isMine = true; 
      placed++; 
    }
  }
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (!newBoard[r][c].isMine) {
        let count = 0
        for (let dr = -1; dr <= 1; dr++)
          for (let dc = -1; dc <= 1; dc++) {
            const nr = r + dr, nc = c + dc
            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && newBoard[nr][nc].isMine) count++
          }
        newBoard[r][c].neighborCount = count
      }
    }
  }
  return newBoard
}

function revealCells(board: Cell[][], r: number, c: number, rows: number, cols: number): Cell[][] {
  const newBoard = board.map((row) => row.map((cell) => ({ ...cell })))
  const stack = [[r, c]]
  while (stack.length) {
    const [cr, cc] = stack.pop()!
    if (cr < 0 || cr >= rows || cc < 0 || cc >= cols) continue
    const cell = newBoard[cr][cc]
    if (cell.isRevealed || cell.isFlagged || cell.isMine) continue
    cell.isRevealed = true
    if (cell.neighborCount === 0)
      for (let dr = -1; dr <= 1; dr++)
        for (let dc = -1; dc <= 1; dc++)
          stack.push([cr + dr, cc + dc])
  }
  return newBoard
}

const NUM_COLORS: Record<number, string> = {
  1: "text-blue-400", 2: "text-emerald-400", 3: "text-rose-400",
  4: "text-purple-400", 5: "text-orange-400", 6: "text-cyan-400",
  7: "text-pink-400", 8: "text-gray-400",
}

interface MinesweeperDialogProps {
  open: boolean
  onClose: () => void
}

export function MinesweeperDialog({ open, onClose }: MinesweeperDialogProps) {
  const [difficulty, setDifficulty] = useState<DifficultyKey>("beginner")
  const { rows, cols, mines, panelWidth } = DIFFICULTIES[difficulty]

  const [board, setBoard]       = useState<Cell[][]>(() => createEmptyBoard(rows, cols))
  const [status, setStatus]     = useState<"playing" | "won" | "lost">("playing")
  const [flags, setFlags]       = useState(0)
  const [time, setTime]         = useState(0)
  const [started, setStarted]   = useState(false)
  const [exploded, setExploded] = useState<[number, number] | null>(null)
  const [isMouseDown, setIsMouseDown] = useState(false)

  useEffect(() => {
    if (!started || status !== "playing") return
    const t = setInterval(() => setTime((p) => p + 1), 1000)
    return () => clearInterval(t)
  }, [started, status])

  const reset = (targetDiff: DifficultyKey = difficulty) => {
    const config = DIFFICULTIES[targetDiff]
    setDifficulty(targetDiff)
    setBoard(createEmptyBoard(config.rows, config.cols))
    setStatus("playing")
    setFlags(0)
    setTime(0)
    setStarted(false)
    setExploded(null)
    setIsMouseDown(false)
  }

  const handleClick = (r: number, c: number) => {
    if (status !== "playing") return
    const cell = board[r][c]
    if (cell.isRevealed || cell.isFlagged) return

    let currentBoard = board
    if (!started) {
      currentBoard = placeMinesAndCalculate(board, r, c, rows, cols, mines)
      setBoard(currentBoard)
      setStarted(true)
    }

    if (currentBoard[r][c].isMine) {
      const newBoard = currentBoard.map((row) =>
        row.map((cell) => ({ ...cell, isRevealed: cell.isMine ? true : cell.isRevealed }))
      )
      setBoard(newBoard)
      setStatus("lost")
      setExploded([r, c])
      return
    }

    const newBoard = revealCells(currentBoard, r, c, rows, cols)
    setBoard(newBoard)
    if (newBoard.flat().filter((cell) => !cell.isRevealed && !cell.isMine).length === 0)
      setStatus("won")
  }

  const handleRightClick = (e: React.MouseEvent, r: number, c: number) => {
    e.preventDefault()
    if (status !== "playing") return
    const cell = board[r][c]
    if (cell.isRevealed) return
    if (!cell.isFlagged && flags >= mines) return
    const newBoard = board.map((row, ri) =>
      row.map((cell, ci) => ri === r && ci === c ? { ...cell, isFlagged: !cell.isFlagged } : cell)
    )
    setBoard(newBoard)
    setFlags((p) => cell.isFlagged ? p - 1 : p + 1)
  }

  const handleChord = (r: number, c: number) => {
    if (status !== "playing") return
    const cell = board[r][c]
    if (!cell.isRevealed || cell.neighborCount === 0) return

    let flagCount = 0
    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        const nr = r + dr, nc = c + dc
        if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && board[nr][nc].isFlagged) {
          flagCount++
        }
      }
    }

    if (flagCount === cell.neighborCount) {
      let newBoard = board.map(row => row.map(c => ({ ...c })))
      let hitMine: [number, number] | null = null
      const stack: [number, number][] = []

      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          const nr = r + dr, nc = c + dc
          if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
            const neighbor = newBoard[nr][nc]
            if (!neighbor.isRevealed && !neighbor.isFlagged) {
              if (neighbor.isMine) {
                hitMine = [nr, nc]
                neighbor.isRevealed = true
              } else {
                stack.push([nr, nc])
              }
            }
          }
        }
      }

      if (hitMine) {
        newBoard = newBoard.map(row =>
          row.map(c => ({ ...c, isRevealed: c.isMine ? true : c.isRevealed }))
        )
        setBoard(newBoard)
        setStatus("lost")
        setExploded(hitMine)
        return
      }

      while (stack.length) {
        const [cr, cc] = stack.pop()!
        const currentCell = newBoard[cr][cc]
        if (currentCell.isRevealed || currentCell.isFlagged || currentCell.isMine) continue
        currentCell.isRevealed = true
        if (currentCell.neighborCount === 0) {
          for (let dr = -1; dr <= 1; dr++)
            for (let dc = -1; dc <= 1; dc++) {
              const nnr = cr + dr, nnc = cc + dc
              if (nnr >= 0 && nnr < rows && nnc >= 0 && nnc < cols) stack.push([nnr, nnc])
            }
        }
      }
      setBoard(newBoard)
      if (newBoard.flat().filter(c => !c.isRevealed && !c.isMine).length === 0) setStatus("won")
    }
  }

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      {/* 动态计算宽度，完美承载各个难度尺寸 */}
      <DialogContent 
        className="px-4 py-5 transition-all duration-200"
        style={{ maxWidth: `${panelWidth}px` }}
      >
        {/* 修改点 1：清理标题栏，使其回归纯粹，给右上角留出足够空间，杜绝重叠 */}
        <DialogHeader className="mb-2">
          <DialogTitle className="text-sm font-bold text-foreground">
            💣 扫雷 (Classic)
          </DialogTitle>
        </DialogHeader>

        {/* 经典 3D 银灰色外壳 */}
        <div className="bg-zinc-300 dark:bg-zinc-800 p-2 rounded-sm select-none border-2 border-t-zinc-100 border-l-zinc-100 border-b-zinc-500 border-r-zinc-500 dark:border-t-zinc-600 dark:border-l-zinc-600 dark:border-b-black dark:border-r-black">
          
          {/* 修改点 2：将难度选择调整到 3D 外壳内部最上方，做成仿 Windows 系统菜单选项的样式，极为自然 */}
          <div className="flex items-center gap-2 mb-2.5 pb-2 px-1 border-b border-zinc-400 dark:border-zinc-700">
            <span className="text-[11px] font-bold text-zinc-600 dark:text-zinc-400">游戏难度:</span>
            <div className="flex gap-1 bg-zinc-400 dark:bg-zinc-900 p-0.5 rounded-sm border-[1.5px] border-b-zinc-100 border-r-zinc-100 border-t-zinc-600 border-l-zinc-600">
              {(Object.keys(DIFFICULTIES) as DifficultyKey[]).map((key) => (
                <button
                  key={key}
                  onClick={() => reset(key)}
                  className={`px-2 py-0.5 text-[11px] font-bold transition-all ${
                    difficulty === key
                      ? "bg-zinc-600 text-white shadow-inner border-[1px] border-black"
                      : "bg-zinc-200 text-zinc-800 border-[1px] border-t-white border-l-white border-b-zinc-400 border-r-zinc-400 active:border-zinc-400"
                  }`}
                >
                  {DIFFICULTIES[key].label}
                </button>
              ))}
            </div>
          </div>

          {/* 顶部状态栏：LED 与 经典笑脸 */}
          <div className="flex items-center justify-between bg-zinc-300 dark:bg-zinc-800 p-1.5 mb-2 border-2 border-t-zinc-500 border-l-zinc-500 border-b-zinc-100 border-r-zinc-100 dark:border-t-black dark:border-l-black dark:border-b-zinc-600 dark:border-r-zinc-600">
            {/* 剩余雷数 LED */}
            <div className="bg-black text-[#ff0701] font-mono text-2xl px-1.5 py-0.5 border-2 border-t-zinc-700 border-l-zinc-700 border-b-zinc-400 border-r-zinc-400 leading-none shadow-inner tracking-widest">
              {String(Math.max(0, mines - flags)).padStart(3, "0")}
            </div>

            {/* 笑脸按钮 */}
            <button 
              onClick={() => reset()} 
              className="w-10 h-10 flex items-center justify-center bg-zinc-200 text-2xl outline-none border-2 border-t-white border-l-white border-b-zinc-500 border-r-zinc-500 active:border-t-zinc-500 active:border-l-zinc-500 active:border-b-white active:border-r-white"
            >
              {status === "won" ? "😎" : status === "lost" ? "😵" : isMouseDown ? "😮" : "🙂"}
            </button>

            {/* 计时器 LED */}
            <div className="bg-black text-[#ff0701] font-mono text-2xl px-1.5 py-0.5 border-2 border-t-zinc-700 border-l-zinc-700 border-b-zinc-400 border-r-zinc-400 leading-none shadow-inner tracking-widest">
              {String(Math.min(time, 999)).padStart(3, "0")}
            </div>
          </div>

          {/* 内凹大框：包裹左侧动态雷区与右侧 GIF */}
          <div 
            className="flex flex-row gap-0 bg-zinc-400 dark:bg-zinc-700 p-1 border-2 border-t-zinc-500 border-l-zinc-500 border-b-zinc-100 border-r-zinc-100 dark:border-t-black dark:border-l-black dark:border-b-zinc-500 dark:border-r-zinc-500"
            onMouseDown={() => setIsMouseDown(true)}
            onMouseUp={() => setIsMouseDown(false)}
            onMouseLeave={() => setIsMouseDown(false)}
            onContextMenu={(e) => e.preventDefault()}
          >
            {/* 左侧：动态雷区按钮阵列 */}
            <div className="flex-none flex flex-col gap-0">
              {board.map((row, r) => (
                <div key={r} className="flex gap-0">
                  {row.map((cell, c) => {
                    const isExploded = exploded?.[0] === r && exploded?.[1] === c
                    const isWrongFlag = status === "lost" && cell.isFlagged && !cell.isMine
                    const shouldRevealMine = status === "lost" && cell.isMine && !cell.isFlagged

                    if (!cell.isRevealed && !shouldRevealMine && !isWrongFlag) {
                      return (
                        <button 
                          key={c}
                          onClick={() => handleClick(r, c)}
                          onContextMenu={(e) => handleRightClick(e, r, c)}
                          className="w-[30px] h-[30px] flex items-center justify-center text-sm outline-none box-border select-none bg-zinc-300 dark:bg-zinc-600 border-[3px] border-t-zinc-100 border-l-zinc-100 border-b-zinc-500 border-r-zinc-500 dark:border-t-zinc-500 dark:border-l-zinc-500 dark:border-b-zinc-800 dark:border-r-zinc-800 active:border-zinc-400 dark:active:border-zinc-700"
                        >
                          {cell.isFlagged ? "🚩" : ""}
                        </button>       
                      )
                    }

                    if (isWrongFlag) {
                      return (
                        <div key={c} className="w-[30px] h-[30px] flex items-center justify-center bg-zinc-300 dark:bg-zinc-800 border-[0.5px] border-zinc-500 dark:border-zinc-700 relative">
                          <span className="opacity-60 text-sm">💣</span>
                          <span className="absolute text-red-500 text-xl font-bold">×</span>
                        </div>
                      )
                    }

                    return (
                      <div 
                        key={c}
                        onMouseDown={(e) => {
                          if (e.buttons === 3) handleChord(r, c)
                        }}
                        onContextMenu={(e) => e.preventDefault()}
                        className={`w-[30px] h-[30px] rounded-none flex items-center justify-center text-[15px] font-extrabold select-none border-[0.5px] border-zinc-400 dark:border-zinc-700/50
                          ${cell.isMine
                            ? isExploded
                              ? "bg-red-600 text-white"
                              : "bg-zinc-300 dark:bg-zinc-800"
                            : "bg-zinc-200 dark:bg-zinc-900/60"
                          } ${cell.isRevealed && !cell.isMine && cell.neighborCount > 0
                            ? NUM_COLORS[cell.neighborCount] ?? ""
                            : ""
                          }`}
                      >
                        {cell.isMine ? "💣" : cell.neighborCount > 0 ? cell.neighborCount : ""}
                      </div>
                    )
                  })}
                </div>
              ))}
            </div>

            {/* 右侧：自然拉伸的灰色空白，高度自动与左侧雷区对齐 */}
            <div 
              className="flex-grow flex items-center justify-center p-3 min-w-[180px]"
              style={{ height: `${rows * 30}px` }}
            >
              {status === "won" && (
                <img 
                  src={WIN_GIF_PATH} 
                  alt="游戏胜利" 
                  className="max-w-full max-h-full object-contain rounded-sm animate-fade-in"
                />
              )}
              {status === "lost" && (
                <img 
                  src={LOSE_GIF_PATH} 
                  alt="游戏失败" 
                  className="max-w-full max-h-full object-contain rounded-sm animate-fade-in"
                />
              )}
            </div>

          </div> {/* 内凹大框闭合 */}
        </div>

        {/* 底部操作与提示 */}
        <div className="flex items-center justify-between mt-2">
          <p className="text-[10px] text-muted-foreground">左键揭开 · 右键插旗 · 左右键同按范围排雷</p>
          <Button size="sm" variant="outline" className="h-7 text-xs" onClick={() => reset()}>
            {status === "playing" ? "重置" : "再来一局"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}