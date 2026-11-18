package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	"github.com/monbooru/monworker/internal/config"
	"github.com/monbooru/monworker/internal/infer"
	"github.com/monbooru/monworker/internal/ipc"
	"github.com/monbooru/monworker/internal/models"
	"github.com/monbooru/monworker/internal/pair"
	"github.com/monbooru/monworker/internal/server"
	"github.com/monbooru/monworker/internal/shared/fsx"
	"github.com/monbooru/monworker/internal/shared/logx"
)

var version = "dev"

func main() {
	if len(os.Args) >= 2 && os.Args[1] == "child" {
		runChild(os.Args[2:])
		return
	}
	if len(os.Args) >= 2 && os.Args[1] == "healthcheck" {
		os.Exit(healthcheck(os.Getenv))
	}
	if len(os.Args) == 2 && (os.Args[1] == "-version" || os.Args[1] == "--version") {
		fmt.Println(version)
		return
	}
	cfg, err := config.Parse(os.Args[1:], os.Getenv)
	if err != nil {
		fmt.Fprintf(os.Stderr, "monworker: %v\n", err)
		os.Exit(2)
	}
	logx.Set(cfg.LogLevel)
	if err := run(cfg); err != nil {
		logx.Errorf("monworker: %v", err)
		os.Exit(1)
	}
}

func run(cfg config.Config) error {
	store, err := config.Load(cfg.File)
	if err != nil {
		return err
	}
	if missing := infer.MissingRuntimeLibrary(); missing != "" {
		logx.Warnf("infer: %s; no model will run", missing)
	}
	child := &ipc.Child{Provider: cfg.Provider, Args: []string{"child",
		"--models", cfg.Models, "--cache", cfg.Cache, "--provider", cfg.Provider,
		"--threads", strconv.Itoa(cfg.Threads), "--slots", strconv.Itoa(cfg.Parallel)}}
	defer child.Close()
	digests := models.NewDigests(filepath.Join(cfg.Cache, "digests.json"))
	srv := server.New(cfg, version, child, store, digests)
	// The weights are read now rather than on monbooru's first listing,
	// which would find every model not ready yet.
	for _, f := range models.Scan(cfg.Models) {
		digests.Of(f)
	}

	ln, err := net.Listen("tcp", cfg.Listen)
	if err != nil {
		return err
	}
	addr := ln.Addr().String()
	if cfg.AddrFile != "" {
		if err := fsx.WriteFile(cfg.AddrFile, []byte(addr)); err != nil {
			return err
		}
	}
	logx.Infof("monworker %s listening on %s: models %s, %s, %d threads, %d images at once", version, addr, cfg.Models, cfg.Provider, cfg.Threads, cfg.Parallel)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if cfg.ExitOnStdinEOF {
		go func() {
			_, _ = io.Copy(io.Discard, os.Stdin)
			logx.Infof("monworker: stdin closed, stopping")
			stop()
		}()
	}
	go srv.Idle(ctx, time.Duration(cfg.IdleMinutes)*time.Minute)
	if cfg.MonbooruURL != "" {
		self := cfg.Advertise
		if self == "" {
			_, port, _ := net.SplitHostPort(addr)
			self = "http://localhost:" + port
		}
		go pair.Offer{Monbooru: cfg.MonbooruURL, Name: cfg.Name, Version: version, Self: self, Store: store}.Run(ctx)
	}

	hs := &http.Server{Handler: srv, ReadHeaderTimeout: 10 * time.Second}
	served := make(chan error, 1)
	go func() { served <- hs.Serve(ln) }()
	select {
	case err := <-served:
		if !errors.Is(err, http.ErrServerClosed) {
			return err
		}
	case <-ctx.Done():
	}
	shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = hs.Shutdown(shutdown)
	return nil
}

// healthcheck is the container's: 0 when the worker answers /health on
// the port it listens on.
func healthcheck(getenv func(string) string) int {
	cfg, err := config.Parse(nil, getenv)
	if err != nil {
		return 1
	}
	_, port, _ := net.SplitHostPort(cfg.Listen)
	c := &http.Client{Timeout: 3 * time.Second}
	resp, err := c.Get("http://127.0.0.1:" + port + "/health")
	if err != nil {
		return 1
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 1
	}
	return 0
}
