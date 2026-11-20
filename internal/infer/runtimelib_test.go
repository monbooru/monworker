package infer

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestOnlyAMissingORTLibPathMeansNoModelWillRun(t *testing.T) {
	lib := filepath.Join(t.TempDir(), "libonnxruntime.so")
	t.Setenv("ORT_LIB_PATH", lib)
	if got := MissingRuntimeLibrary(); !strings.HasSuffix(got, "; no model will run") {
		t.Errorf("a missing ORT_LIB_PATH: %q", got)
	}
	if err := os.WriteFile(lib, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if got := MissingRuntimeLibrary(); got != "" {
		t.Errorf("a present ORT_LIB_PATH: %q", got)
	}
	t.Setenv("ORT_LIB_PATH", "libonnxruntime-nowhere.so")
	if got := MissingRuntimeLibrary(); !strings.HasSuffix(got, "is tried at the first call") {
		t.Errorf("a bare name: %q", got)
	}
}
