package server

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/monbooru/monworker/internal/config"
	"github.com/monbooru/monworker/internal/ipc"
	"github.com/monbooru/monworker/internal/models"
	"github.com/monbooru/monworker/internal/shared/worker"
)

type fakeBackend struct {
	mu    sync.Mutex
	calls []ipc.RunRequest
	block chan struct{}
}

func (b *fakeBackend) Run(ctx context.Context, req ipc.RunRequest) (ipc.RunResponse, error) {
	b.mu.Lock()
	b.calls = append(b.calls, req)
	block := b.block
	b.mu.Unlock()
	if block != nil {
		select {
		case <-block:
		case <-ctx.Done():
			return ipc.RunResponse{}, ctx.Err()
		}
	}
	resp := ipc.RunResponse{}
	for range req.Models {
		frames := make([]ipc.FrameScores, len(req.Frames))
		for i := range frames {
			frames[i] = ipc.FrameScores{I: []int32{1}, S: []float32{0.7}}
		}
		resp.Results = append(resp.Results, frames)
	}
	return resp, nil
}
func (b *fakeBackend) Status() ipc.Status               { return ipc.Status{Loaded: []string{"wd"}} }
func (b *fakeBackend) ReleaseIdle(_ time.Duration) bool { return false }

func mustWrite(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func newWorker(t *testing.T, b Backend) (*worker.Client, *config.Store) {
	t.Helper()
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "models", "wd", "model.onnx"), "weights")
	mustWrite(t, filepath.Join(dir, "models", "wd", "tags.csv"), "id,name,category\n0,1girl,0\n1,hatsune miku,4\n")
	mustWrite(t, filepath.Join(dir, "models", "wd", "tagger.json"), `{"version": 1, "profile": {"input_size": 8}}`)
	store, err := config.Load(filepath.Join(dir, "monworker.toml"))
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Put(config.Pairing{URL: "http://nas:8455", Token: "mb", PeerToken: "paired"}); err != nil {
		t.Fatal(err)
	}
	cfg := config.Config{Name: "gpu-box", Models: filepath.Join(dir, "models"), Provider: "cpu", Threads: 4, Parallel: 2, Token: "run-secret"}
	srv := httptest.NewServer(New(cfg, "v1.0.0", b, store, models.NewDigests("")))
	t.Cleanup(srv.Close)
	return &worker.Client{BaseURL: srv.URL, Token: "paired"}, store
}

func frames(n int) [][]byte {
	out := make([][]byte, n)
	for i := range out {
		out[i] = make([]byte, 8*8*3)
	}
	return out
}

func TestEveryRouteButHealthWantsAToken(t *testing.T) {
	c, _ := newWorker(t, &fakeBackend{})
	resp, err := http.Get(c.BaseURL + "/health")
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("health: %v %v", resp, err)
	}
	_ = resp.Body.Close()
	for _, tok := range []string{"", "nope"} {
		bad := &worker.Client{BaseURL: c.BaseURL, Token: tok}
		var se *worker.StatusError
		if _, err := bad.Models(context.Background()); !errors.As(err, &se) || se.Status != http.StatusUnauthorized {
			t.Errorf("token %q: %v", tok, err)
		}
	}
	for _, tok := range []string{"paired", "run-secret"} {
		if _, err := (&worker.Client{BaseURL: c.BaseURL, Token: tok}).Info(context.Background()); err != nil {
			t.Errorf("token %q: %v", tok, err)
		}
	}
}

func TestModelsAndLabelsDescribeTheFolder(t *testing.T) {
	c, _ := newWorker(t, &fakeBackend{})
	ctx := context.Background()
	var ms []worker.Model
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		var err error
		if ms, err = c.Models(ctx); err != nil {
			t.Fatal(err)
		}
		if len(ms) == 1 && ms[0].Ready {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if len(ms) != 1 || !ms[0].Ready || ms[0].Digest == "" || ms[0].Input.Size != 8 || ms[0].Labels != 2 {
		t.Fatalf("models = %+v", ms)
	}
	l, changed, err := c.Labels(ctx, "wd", "")
	if err != nil || !changed || l.Labels[1].Name != "hatsune miku" || l.Labels[1].CategoryID != 4 || l.ETag != ms[0].LabelsETag {
		t.Fatalf("labels = %+v, %v, %v", l, changed, err)
	}
	if _, changed, err := c.Labels(ctx, "wd", l.ETag); err != nil || changed {
		t.Errorf("labels by etag: changed %v, %v", changed, err)
	}
	var se *worker.StatusError
	if _, _, err := c.Labels(ctx, "..", ""); !errors.As(err, &se) || se.Status != http.StatusNotFound {
		t.Errorf("a name that leaves the folder: %v", err)
	}
	info, err := c.Info(ctx)
	if err != nil || info.Name != "gpu-box" || info.Slots != 2 || len(info.Loaded) != 1 || filepath.Base(info.ModelsDir) != "models" || !filepath.IsAbs(info.ModelsDir) {
		t.Errorf("info = %+v, %v", info, err)
	}
}

func TestTagAnswersPerModelAndFrame(t *testing.T) {
	b := &fakeBackend{}
	c, _ := newWorker(t, b)
	resp, err := c.Tag(context.Background(), worker.TagRequest{Models: []string{"wd"}, Priority: worker.PriorityBatch, Frames: worker.FrameSpec{Size: 8}}, frames(3))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Results[0].Model != "wd" || len(resp.Results[0].Frames) != 3 || resp.Results[0].Frames[2].S[0] != 0.7 {
		t.Errorf("resp = %+v", resp)
	}
	if len(b.calls) != 1 || b.calls[0].Size != 8 || len(b.calls[0].Frames) != 3 {
		t.Errorf("backend got %+v", b.calls)
	}
	var se *worker.StatusError
	if _, err := c.Tag(context.Background(), worker.TagRequest{Models: []string{"wd"}, Frames: worker.FrameSpec{Size: 4}}, [][]byte{make([]byte, 48)}); !errors.As(err, &se) || se.Status != http.StatusBadRequest {
		t.Errorf("frames of the wrong size: %v", err)
	}
	if _, err := c.Tag(context.Background(), worker.TagRequest{Models: []string{"nope"}, Frames: worker.FrameSpec{Size: 8}}, frames(1)); !errors.As(err, &se) || se.Code != worker.CodeNotReady {
		t.Errorf("a model the worker lacks: %v", err)
	}
}

func TestTagTakesPNGFramesAsTheSamePixels(t *testing.T) {
	b := &fakeBackend{}
	c, _ := newWorker(t, b)
	info, err := c.Info(context.Background())
	if err != nil || !slices.Contains(info.Formats, worker.FormatPNG) || !slices.Contains(info.Formats, worker.FormatRGB8) {
		t.Fatalf("formats = %v, %v", info.Formats, err)
	}
	sent := frames(2)
	for _, format := range []string{worker.FormatRGB8, worker.FormatPNG} {
		c.Format = format
		if _, err := c.Tag(context.Background(), worker.TagRequest{Models: []string{"wd"}, Frames: worker.FrameSpec{Size: 8}}, sent); err != nil {
			t.Fatalf("%s: %v", format, err)
		}
	}
	if len(b.calls) != 2 || !slices.EqualFunc(b.calls[0].Frames, b.calls[1].Frames, bytes.Equal) || !slices.EqualFunc(b.calls[1].Frames, sent, bytes.Equal) {
		t.Errorf("the backend did not get the same pixels raw and as PNG")
	}
}

func TestQueueLetsAnInteractiveCallPassTheBatch(t *testing.T) {
	q := newQueue(1)
	ctx := context.Background()
	if err := q.admit(ctx, false); err != nil {
		t.Fatal(err)
	}
	order := make(chan string, 2)
	var wg sync.WaitGroup
	wg.Go(func() {
		_ = q.admit(ctx, false)
		order <- "batch"
		q.done()
	})
	time.Sleep(20 * time.Millisecond)
	wg.Go(func() {
		_ = q.admit(ctx, true)
		order <- "interactive"
		q.done()
	})
	time.Sleep(20 * time.Millisecond)
	q.done()
	wg.Wait()
	if first := <-order; first != "interactive" {
		t.Errorf("first served: %s", first)
	}
}

func TestRemoveForgetsThePairingAndItsToken(t *testing.T) {
	c, store := newWorker(t, &fakeBackend{})
	req, _ := http.NewRequest(http.MethodPost, c.BaseURL+"/api/v1/pair/remove", nil)
	req.Header.Set("Authorization", "Bearer paired")
	resp, err := http.DefaultClient.Do(req)
	if err != nil || resp.StatusCode != http.StatusNoContent {
		t.Fatalf("remove: %v %v", resp, err)
	}
	_ = resp.Body.Close()
	if len(store.Pairings()) != 0 {
		t.Errorf("pairings left: %+v", store.Pairings())
	}
	if _, err := c.Info(context.Background()); err == nil {
		t.Error("the removed token still opens the worker")
	}
}
