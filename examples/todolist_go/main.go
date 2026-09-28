package main

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"
)

// Task 是单条待办事项。
type Task struct {
	ID        int       `json:"id"`
	Title     string    `json:"title"`
	Done      bool      `json:"done"`
	CreatedAt time.Time `json:"created_at"`
}

// storePath 是持久化文件路径（工作目录下的隐藏文件）。
const storePath = ".todolist_go.json"

// loadTasks 读取并解析任务列表；文件不存在时返回空列表。
func loadTasks() ([]Task, error) {
	data, err := os.ReadFile(storePath)
	if err != nil {
		if os.IsNotExist(err) {
			return []Task{}, nil
		}
		return nil, err
	}
	if len(data) == 0 {
		return []Task{}, nil
	}
	var tasks []Task
	if err := json.Unmarshal(data, &tasks); err != nil {
		return nil, err
	}
	return tasks, nil
}

// saveTasks 把任务列表写回磁盘。
func saveTasks(tasks []Task) error {
	data, err := json.MarshalIndent(tasks, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(storePath, data, 0o644)
}

// nextID 返回下一个可用 ID（当前最大 ID + 1）。
func nextID(tasks []Task) int {
	max := 0
	for _, t := range tasks {
		if t.ID > max {
			max = t.ID
		}
	}
	return max + 1
}

func main() {
	tasks, err := loadTasks()
	if err != nil {
		fmt.Fprintf(os.Stderr, "load error: %v\n", err)
		os.Exit(1)
	}

	args := os.Args[1:]
	if len(args) == 0 {
		printHelp()
		return
	}

	switch args[0] {
	case "add":
		if len(args) < 2 {
			fmt.Println("usage: todo add <title>")
			return
		}
		title := strings.Join(args[1:], " ")
		newTask := Task{
			ID:        nextID(tasks),
			Title:     title,
			Done:      false,
			CreatedAt: time.Now(),
		}
		tasks = append(tasks, newTask)
		if err := saveTasks(tasks); err != nil {
			fmt.Fprintf(os.Stderr, "save error: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("added: #%d %s\n", newTask.ID, newTask.Title)

	case "list":
		listTasks(tasks)

	case "done":
		if len(args) < 2 {
			fmt.Println("usage: todo done <id>")
			return
		}
		var id int
		if _, err := fmt.Sscanf(args[1], "%d", &id); err != nil {
			fmt.Println("invalid id")
			return
		}
		found := false
		for i := range tasks {
			if tasks[i].ID == id {
				tasks[i].Done = true
				found = true
			}
		}
		if !found {
			fmt.Printf("no task with id #%d\n", id)
			return
		}
		if err := saveTasks(tasks); err != nil {
			fmt.Fprintf(os.Stderr, "save error: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("marked #%d done\n", id)

	case "rm":
		if len(args) < 2 {
			fmt.Println("usage: todo rm <id>")
			return
		}
		var id int
		if _, err := fmt.Sscanf(args[1], "%d", &id); err != nil {
			fmt.Println("invalid id")
			return
		}
		newTasks := tasks[:0]
		for _, t := range tasks {
			if t.ID != id {
				newTasks = append(newTasks, t)
			}
		}
		if len(newTasks) == len(tasks) {
			fmt.Printf("no task with id #%d\n", id)
			return
		}
		if err := saveTasks(newTasks); err != nil {
			fmt.Fprintf(os.Stderr, "save error: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("removed #%d\n", id)

	case "help", "--help", "-h":
		printHelp()

	default:
		fmt.Printf("unknown command: %s\n", args[0])
		printHelp()
	}
}

// listTasks 按 ID 升序打印所有任务。
func listTasks(tasks []Task) {
	if len(tasks) == 0 {
		fmt.Println("no tasks yet")
		return
	}
	sort.SliceStable(tasks, func(i, j int) bool {
		return tasks[i].ID < tasks[j].ID
	})
	for _, t := range tasks {
		mark := "[ ]"
		if t.Done {
			mark = "[x]"
		}
		fmt.Printf("%s #%d %s\n", mark, t.ID, t.Title)
	}
}

func printHelp() {
	fmt.Println(`todolist - a simple command-line todo manager

Usage:
  todo add <title>      add a new task
  todo list             list all tasks
  todo done <id>        mark a task as done
  todo rm <id>          remove a task
  todo help             show this help`)
}
