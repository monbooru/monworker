package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/monbooru/monworker/internal/infer"
	"github.com/monbooru/monworker/internal/ipc"
)

func runChild(argv []string) {
	fs := flag.NewFlagSet("child", flag.ExitOnError)
	addr := fs.String("addr", "", "the front's address to connect to")
	var cfg infer.Config
	fs.StringVar(&cfg.ModelsDir, "models", "", "models directory")
	fs.StringVar(&cfg.CacheDir, "cache", "", "providers' compile caches")
	fs.StringVar(&cfg.Provider, "provider", "cpu", "execution provider")
	fs.IntVar(&cfg.Threads, "threads", 1, "threads ONNX Runtime may use")
	fs.IntVar(&cfg.Slots, "slots", 1, "calls a 448 px model runs at once")
	_ = fs.Parse(argv)
	if *addr == "" {
		fmt.Fprintln(os.Stderr, "monworker child: --addr is required")
		os.Exit(2)
	}
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT)
	defer cancel()
	if err := ipc.Serve(ctx, *addr, os.Getenv(ipc.TokenEnv), infer.New(cfg)); err != nil {
		fmt.Fprintf(os.Stderr, "monworker child: %v\n", err)
		os.Exit(1)
	}
}
