package models

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/monbooru/monworker/internal/registry"
)

func mustWriteFile(t *testing.T, path string, data []byte, perm os.FileMode) {
	t.Helper()
	if err := os.WriteFile(path, data, perm); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func mustMkdirAll(t *testing.T, path string, perm os.FileMode) {
	t.Helper()
	if err := os.MkdirAll(path, perm); err != nil {
		t.Fatalf("mkdir %s: %v", path, err)
	}
}

func mustResolveProfile(t *testing.T, modelPath, taggerName, tagsFile string) Profile {
	t.Helper()
	p, err := ResolveProfile(modelPath, taggerName, tagsFile)
	if err != nil {
		t.Fatalf("ResolveProfile(%q): %v", taggerName, err)
	}
	return p
}

func installRegistryTagger(t *testing.T, modelPath, name string) {
	t.Helper()
	src := filepath.Join("..", "registry", registry.TaggersDir, name)
	entries, err := os.ReadDir(src)
	if err != nil {
		t.Fatalf("read registry folder %s: %v", name, err)
	}
	dst := filepath.Join(modelPath, name)
	mustMkdirAll(t, dst, 0o755)
	for _, e := range entries {
		data, err := os.ReadFile(filepath.Join(src, e.Name()))
		if err != nil {
			t.Fatalf("read %s: %v", e.Name(), err)
		}
		mustWriteFile(t, filepath.Join(dst, e.Name()), data, 0o644)
	}
}
