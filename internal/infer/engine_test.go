package infer

import (
	"context"
	"encoding/json"
	"errors"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/monbooru/monworker/internal/ipc"
	"github.com/monbooru/monworker/internal/models"
)

func TestBudgetServesInTurnAndGivesBackACancelledShare(t *testing.T) {
	b := newBudget(4)
	ctx := context.Background()
	if err := b.acquire(ctx, 3); err != nil {
		t.Fatal(err)
	}
	big := make(chan error, 1)
	go func() { big <- b.acquire(ctx, 4) }()
	time.Sleep(20 * time.Millisecond)

	small, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
	defer cancel()
	if err := b.acquire(small, 1); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("a later small call went ahead: %v", err)
	}
	b.release(3)
	if err := <-big; err != nil {
		t.Fatal(err)
	}
	if b.free != 0 || len(b.waiters) != 0 {
		t.Errorf("free %d, waiters %d", b.free, len(b.waiters))
	}
}

func TestSessionWorkersShrinkWithTheInputArea(t *testing.T) {
	for _, tc := range []struct {
		provider        string
		parallel, input int
		want            int
	}{
		{"cpu", 4, 448, 4},
		{"cpu", 4, 512, 3},
		{"cpu", 4, 1008, 1},
		{"cuda", 8, 1008, 1},
		{"directml", 4, 448, 1},
	} {
		if got := sessionWorkers(tc.provider, tc.parallel, tc.input); got != tc.want {
			t.Errorf("sessionWorkers(%s, %d, %d) = %d, want %d", tc.provider, tc.parallel, tc.input, got, tc.want)
		}
	}
}

func TestParseDirectMLDeviceID(t *testing.T) {
	for in, want := range map[string]int{"": 0, "2": 2, "-1": 0, "gpu": 0} {
		if got := parseDirectMLDeviceID(in); got != want {
			t.Errorf("parseDirectMLDeviceID(%q) = %d, want %d", in, got, want)
		}
	}
}

func TestARunNamingAModelThatIsNotThereIsRefused(t *testing.T) {
	e := New(Config{ModelsDir: t.TempDir(), Slots: 1, Threads: 1})
	for _, name := range []string{"missing", "../escape"} {
		if _, err := e.Run(context.Background(), ipc.RunRequest{Models: []string{name}, Size: 2, Frames: [][]byte{make([]byte, 12)}}); err == nil {
			t.Errorf("%s: ran", name)
		}
	}
}

func requireORT(t *testing.T) {
	t.Helper()
	if _, err := os.Stat(sharedLibPath()); err != nil {
		t.Skipf("onnxruntime shared library not available: %v", err)
	}
}

func modelsDir(t *testing.T) string {
	t.Helper()
	dir := os.Getenv("MONWORKER_TEST_MODELS")
	if dir == "" {
		t.Skip("MONWORKER_TEST_MODELS is not set")
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		t.Fatal(err)
	}
	return abs
}

func frameOf(t *testing.T, path string, f models.Folder) ([]byte, int) {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = file.Close() }()
	img, _, err := image.Decode(file)
	if err != nil {
		t.Fatal(err)
	}
	size, err := f.InputSide()
	if err != nil {
		t.Fatal(err)
	}
	return rgb8(padAndResize(img, size, models.Fill(f.Profile))), size
}

func TestConformance_MatchesTheReferenceScores(t *testing.T) {
	requireORT(t)
	evalDir := os.Getenv("MONWORKER_TEST_EVAL")
	if evalDir == "" {
		t.Skip("MONWORKER_TEST_EVAL is not set")
	}
	dir := modelsDir(t)
	expected, _ := filepath.Glob(filepath.Join(evalDir, "*", "expected.json"))
	ran := 0
	for _, path := range expected {
		name := filepath.Base(filepath.Dir(path))
		f := models.Open(dir, name)
		if f.Err != nil {
			t.Logf("%s: %v, skipped", name, f.Err)
			continue
		}
		var doc struct {
			Images map[string]map[string]float64 `json:"images"`
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(data, &doc); err != nil {
			t.Fatal(err)
		}
		labels, err := models.ReadLabels(f.LabelPath(), f.Profile)
		if err != nil {
			t.Fatal(err)
		}
		t.Run(name, func(t *testing.T) {
			e := New(Config{ModelsDir: dir, Slots: 1, Threads: 4})
			defer e.ReleaseIdle(0)
			for image, ref := range doc.Images {
				frame, size := frameOf(t, filepath.Join(evalDir, "images", image), f)
				resp, err := e.Run(context.Background(), ipc.RunRequest{Models: []string{name}, Size: size, Frames: [][]byte{frame}})
				if err != nil || resp.Results[0][0].Err != "" {
					t.Fatalf("%s: %v %+v", image, err, resp)
				}
				got := map[string]float32{}
				fr := resp.Results[0][0]
				for j, idx := range fr.I {
					got[labels[idx].Name] = fr.S[j]
				}
				for label, want := range ref {
					if want < 0.35 {
						continue
					}
					if g := got[label]; math.Abs(float64(g)-want) > 0.05 {
						t.Errorf("%s: %s scored %.3f, reference %.3f", image, label, g, want)
					}
				}
				for label, g := range got {
					if g >= 0.5 && ref[label] < 0.4 {
						t.Errorf("%s: %s scored %.3f, reference %.3f", image, label, g, ref[label])
					}
				}
			}
		})
		ran++
	}
	if ran == 0 {
		t.Skip("no expected model is installed")
	}
}
