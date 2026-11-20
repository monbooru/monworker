package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/monbooru/monworker/internal/config"
)

func TestStartCreatesTheModelsFolder(t *testing.T) {
	dir := t.TempDir()
	models := filepath.Join(dir, "data", "models")
	cfg := config.Config{File: filepath.Join(dir, "monworker.toml"), Cache: dir, Models: models, Listen: "127.0.0.1:-1", Parallel: 1, Threads: 1}
	if err := run(cfg); err == nil {
		t.Fatal("listened on port -1")
	}
	if fi, err := os.Stat(models); err != nil || !fi.IsDir() {
		t.Errorf("models folder: %v", err)
	}
}
