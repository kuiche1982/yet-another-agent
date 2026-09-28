package main

import (
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"time"
)

const (
	width  = 40
	height = 20
)

// newGrid 分配 height×width 的布尔网格。
func newGrid() [][]bool {
	g := make([][]bool, height)
	for i := range g {
		g[i] = make([]bool, width)
	}
	return g
}

// randomGrid 用给定随机源填充约 1/4 的格子为存活。
func randomGrid(r *rand.Rand) [][]bool {
	g := newGrid()
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			g[y][x] = r.Intn(4) == 0
		}
	}
	return g
}

// glider 在左上角放置一个滑翔机（glider）初始图案。
func glider(g [][]bool) {
	pts := [][2]int{{1, 0}, {2, 1}, {0, 2}, {1, 2}, {2, 2}}
	for _, p := range pts {
		g[p[1]][p[0]] = true
	}
}

// neighbors 统计 (x,y) 周围 8 格中存活的邻居数（环形边界）。
func neighbors(g [][]bool, x, y int) int {
	count := 0
	for dy := -1; dy <= 1; dy++ {
		for dx := -1; dx <= 1; dx++ {
			if dx == 0 && dy == 0 {
				continue
			}
			nx, ny := x+dx, y+dy
			if nx < 0 {
				nx += width
			} else if nx >= width {
				nx -= width
			}
			if ny < 0 {
				ny += height
			} else if ny >= height {
				ny -= height
			}
			if g[ny][nx] {
				count++
			}
		}
	}
	return count
}

// step 根据康威规则计算下一代。
// 存活格子：邻居为 2 或 3 时继续存活；
// 死亡格子：邻居恰好为 3 时复活。
func step(g [][]bool) [][]bool {
	next := newGrid()
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			n := neighbors(g, x, y)
			alive := g[y][x]
			next[y][x] = (alive && (n == 2 || n == 3)) || (!alive && n == 3)
		}
	}
	return next
}

// render 把网格渲染成 #/. 字符画（# 为存活）。
func render(g [][]bool) string {
	out := ""
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			if g[y][x] {
				out += "#"
			} else {
				out += "."
			}
		}
		out += "\n"
	}
	return out
}

func main() {
	mode := "glider"
	generations := 60
	args := os.Args[1:]
	if len(args) > 0 {
		switch args[0] {
		case "random", "glider":
			mode = args[0]
			if len(args) > 1 {
				if n, err := strconv.Atoi(args[1]); err == nil && n > 0 {
					generations = n
				}
			}
		default:
			if n, err := strconv.Atoi(args[0]); err == nil && n > 0 {
				generations = n
			}
		}
	}

	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	g := newGrid()
	if mode == "random" {
		g = randomGrid(r)
	} else {
		glider(g)
	}

	for gen := 0; gen < generations; gen++ {
		fmt.Print("\x1b[2J\x1b[H") // 清屏
		fmt.Print(render(g))
		fmt.Printf("generation %d (%s)\n", gen, mode)
		g = step(g)
		time.Sleep(120 * time.Millisecond)
	}
}
