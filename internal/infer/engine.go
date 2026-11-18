package infer

import (
	"cmp"
	"context"
	"fmt"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/monbooru/monworker/internal/ipc"
	"github.com/monbooru/monworker/internal/models"
	"github.com/monbooru/monworker/internal/shared/logx"
	"github.com/monbooru/monworker/internal/shared/worker"
	ort "github.com/yalue/onnxruntime_go"
)

type Config struct {
	ModelsDir string
	CacheDir  string
	Provider  string
	Threads   int
	Slots     int
}

type Engine struct {
	cfg    Config
	budget *budget

	mu          sync.Mutex
	idle        *sync.Cond
	inUse       int
	initialized bool
	sessions    map[string]*session
	lastUsed    time.Time
	sinceTrim   int
	trimTimer   *time.Timer
}

func New(cfg Config) *Engine {
	cfg.Provider = cmp.Or(cfg.Provider, "cpu")
	cfg.Threads, cfg.Slots = max(1, cfg.Threads), max(1, cfg.Slots)
	e := &Engine{cfg: cfg, budget: newBudget(cfg.Threads), sessions: map[string]*session{}}
	e.idle = sync.NewCond(&e.mu)
	return e
}

type session struct {
	modelStamp fileStamp
	labelStamp fileStamp
	profileFP  string
	s          *ort.DynamicAdvancedSession
	opts       *ort.SessionOptions
	cleanup    func()
	profile    models.Profile
	size       int
	threads    int
}

type fileStamp struct {
	size    int64
	modTime time.Time
}

func stampOf(path string) (fileStamp, bool) {
	fi, err := os.Stat(path)
	if err != nil {
		return fileStamp{}, false
	}
	return fileStamp{fi.Size(), fi.ModTime()}, true
}

func (s *session) fresh(f models.Folder) bool {
	model, okModel := stampOf(f.ModelPath())
	labels, okLabels := stampOf(f.LabelPath())
	return f.Err == nil && okModel && okLabels && model == s.modelStamp && labels == s.labelStamp &&
		f.Profile.Fingerprint() == s.profileFP
}

func (s *session) destroy() {
	_ = s.s.Destroy()
	if s.cleanup != nil {
		s.cleanup()
	}
	_ = s.opts.Destroy()
}

func (e *Engine) Run(ctx context.Context, req ipc.RunRequest) (ipc.RunResponse, error) {
	if len(req.Models) == 0 {
		return ipc.RunResponse{}, fmt.Errorf("no model named")
	}
	e.mu.Lock()
	run, err := e.prepare(req.Models)
	if err != nil {
		e.mu.Unlock()
		return ipc.RunResponse{}, err
	}
	weight := 1
	for _, s := range run {
		if s.size != req.Size {
			e.mu.Unlock()
			return ipc.RunResponse{}, fmt.Errorf("model takes %d px frames, not %d", s.size, req.Size)
		}
		weight = max(weight, s.threads)
	}
	e.inUse++
	if e.trimTimer != nil {
		e.trimTimer.Stop()
	}
	e.mu.Unlock()
	defer e.release()

	if err := e.budget.acquire(ctx, weight); err != nil {
		return ipc.RunResponse{}, err
	}
	defer e.budget.release(weight)

	resp := ipc.RunResponse{Results: make([][]ipc.FrameScores, len(run))}
	for m, s := range run {
		resp.Results[m] = make([]ipc.FrameScores, len(req.Frames))
		for f, frame := range req.Frames {
			if err := ctx.Err(); err != nil {
				return ipc.RunResponse{}, err
			}
			scores, err := infer(s, frame)
			if err != nil {
				resp.Results[m][f].Err = err.Error()
				continue
			}
			sparse := worker.Sparse(scores)
			resp.Results[m][f].I, resp.Results[m][f].S = sparse.I, sparse.S
		}
	}
	return resp, nil
}

// prepare returns the sessions of names, loading what is missing and
// reloading what changed on disk once no call uses it. Caller holds e.mu.
func (e *Engine) prepare(names []string) ([]*session, error) {
	out := make([]*session, len(names))
	for i, name := range names {
		if err := models.ValidateName(name); err != nil {
			return nil, err
		}
		f := models.Open(e.cfg.ModelsDir, name)
		if s := e.sessions[name]; s != nil {
			if s.fresh(f) {
				out[i] = s
				continue
			}
			for e.inUse > 0 {
				e.idle.Wait()
			}
			s.destroy()
			delete(e.sessions, name)
			logx.Infof("infer: %s changed on disk, loading it again", name)
		}
		if f.Err != nil {
			return nil, fmt.Errorf("model %q: %w", name, f.Err)
		}
		s, err := e.load(f)
		if err != nil {
			return nil, fmt.Errorf("model %q: %w", name, err)
		}
		e.sessions[name] = s
		out[i] = s
	}
	return out, nil
}

// Caller holds e.mu.
func (e *Engine) init() error {
	if e.initialized {
		return nil
	}
	// CUDA caches JIT-compiled kernels under $HOME/.nv, which a container
	// loses on restart; keep them in the cache folder unless
	// CUDA_CACHE_PATH is already set.
	if e.cfg.Provider == "cuda" && os.Getenv("CUDA_CACHE_PATH") == "" && e.cfg.CacheDir != "" {
		dir := filepath.Join(e.cfg.CacheDir, ".nv-cache")
		if err := os.MkdirAll(dir, 0o755); err == nil {
			_ = os.Setenv("CUDA_CACHE_PATH", dir)
		}
	}
	ort.SetSharedLibraryPath(sharedLibPath())
	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("ort init: %w", err)
	}
	// Microsoft's Windows builds report to Microsoft unless told not to.
	if err := ort.DisableTelemetry(); err != nil {
		_ = ort.DestroyEnvironment()
		return fmt.Errorf("ort telemetry: %w", err)
	}
	e.initialized = true
	return nil
}

// Caller holds e.mu.
func (e *Engine) load(f models.Folder) (_ *session, err error) {
	if err := e.init(); err != nil {
		return nil, err
	}
	start := time.Now()
	s := &session{profile: f.Profile, profileFP: f.Profile.Fingerprint()}
	// Before the reads: a write landing during the load then reloads.
	s.modelStamp, _ = stampOf(f.ModelPath())
	s.labelStamp, _ = stampOf(f.LabelPath())
	labels, err := models.ReadLabels(f.LabelPath(), f.Profile)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", f.LabelFile, err)
	}
	inputs, outputs, err := ort.GetInputOutputInfo(f.ModelPath())
	if err != nil {
		return nil, fmt.Errorf("inspect the onnx model: %w", err)
	}
	if len(inputs) == 0 || len(outputs) == 0 {
		return nil, fmt.Errorf("the onnx model has no input or output")
	}
	outIdx := f.Profile.OutputIndex
	if outIdx < 0 || outIdx >= len(outputs) {
		return nil, fmt.Errorf("profile output_index %d out of range (have %d)", outIdx, len(outputs))
	}
	if err := checkLabelWidth(outputs[outIdx].Dimensions, len(labels)); err != nil {
		return nil, err
	}
	s.size = f.Profile.InputSize
	if s.size == 0 {
		if s.size = models.InputSize(inputs[0].Dimensions, f.Profile.Layout); s.size <= 0 {
			return nil, fmt.Errorf("cannot infer the input size from dimensions %v", inputs[0].Dimensions)
		}
	}
	workers := sessionWorkers(e.cfg.Provider, e.cfg.Slots, s.size)
	s.threads = max(1, e.cfg.Threads/workers)

	if s.opts, err = ort.NewSessionOptions(); err != nil {
		return nil, fmt.Errorf("ort session options: %w", err)
	}
	defer func() {
		if err != nil {
			if s.cleanup != nil {
				s.cleanup()
			}
			_ = s.opts.Destroy()
		}
	}()
	if err := s.opts.SetIntraOpNumThreads(s.threads); err != nil {
		return nil, fmt.Errorf("set intra-op threads: %w", err)
	}
	if err := s.opts.SetInterOpNumThreads(1); err != nil {
		return nil, fmt.Errorf("set inter-op threads: %w", err)
	}
	// The arena keeps every concurrent Run's peak working set until the
	// session is destroyed; plain malloc lets mallocTrim hand it back.
	if err := s.opts.SetCpuMemArena(false); err != nil {
		return nil, fmt.Errorf("disable cpu mem arena: %w", err)
	}
	epCacheDir := ""
	if e.cfg.Provider == "openvino" && e.cfg.CacheDir != "" {
		dir := filepath.Join(e.cfg.CacheDir, ".openvino-cache")
		if err := os.MkdirAll(dir, 0o755); err == nil {
			epCacheDir = dir
		}
	}
	if s.cleanup, err = appendExecutionProvider(s.opts, e.cfg.Provider, epCacheDir); err != nil {
		return nil, fmt.Errorf("append execution provider %q: %w", e.cfg.Provider, err)
	}
	newSession := ort.NewDynamicAdvancedSession
	if e.cfg.Provider == "directml" {
		newSession = newDirectMLSession
	}
	if s.s, err = newSession(f.ModelPath(), []string{inputs[0].Name}, []string{outputs[outIdx].Name}, s.opts); err != nil {
		return nil, fmt.Errorf("create the ort session: %w", err)
	}
	logx.Infof("infer: %s loaded on %s in %s (%d px, %d %s at once, %d %s each)", f.Name, strings.ToUpper(e.cfg.Provider),
		time.Since(start).Round(10*time.Millisecond), s.size, workers, plural(workers, "call", "calls"), s.threads, plural(s.threads, "thread", "threads"))
	return s, nil
}

func plural(n int, one, many string) string {
	if n == 1 {
		return one
	}
	return many
}

// Without a periodic trim a long run grows glibc's arena until the next
// idle release; trimIdle gives a burst's heap back once it ends.
const (
	trimInterval = 256
	trimIdle     = 2 * time.Second
)

func (e *Engine) release() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.inUse--
	e.lastUsed = time.Now()
	if e.sinceTrim++; e.sinceTrim >= trimInterval {
		e.sinceTrim = 0
		mallocTrim()
	}
	if e.inUse > 0 {
		return
	}
	e.idle.Broadcast()
	e.trimTimer = time.AfterFunc(trimIdle, func() {
		e.mu.Lock()
		defer e.mu.Unlock()
		if e.inUse == 0 {
			mallocTrim()
		}
	})
}

func (e *Engine) ReleaseIdle(after time.Duration) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.inUse > 0 || len(e.sessions) == 0 || time.Since(e.lastUsed) < after {
		return false
	}
	e.teardownLocked()
	return true
}

// Caller holds e.mu.
func (e *Engine) teardownLocked() {
	for name, s := range e.sessions {
		s.destroy()
		delete(e.sessions, name)
	}
	if e.initialized {
		_ = ort.DestroyEnvironment()
		e.initialized = false
	}
	mallocTrim()
}

func (e *Engine) Status() ipc.Status {
	e.mu.Lock()
	defer e.mu.Unlock()
	return ipc.Status{
		Loaded:   slices.Sorted(maps.Keys(e.sessions)),
		Provider: e.cfg.Provider,
		InUse:    e.inUse > 0,
		LastUsed: e.lastUsed,
	}
}

// tagger.parallel is sized for the catalog's 448 px inputs. Activations
// grow with the input's pixel count, so a larger input runs fewer calls:
// on CPU the intra-op threads already fill the cores, and on a GPU every
// call's working set shares the device's memory.
func sessionWorkers(provider string, parallel, inputSize int) int {
	const refSize = 448
	if provider == "directml" {
		// Concurrent Run on one session crashes the EP's allocator
		// (onnxruntime issue #22147), so its one call gets every core.
		return 1
	}
	if inputSize <= refSize {
		return parallel
	}
	return max(1, parallel*refSize*refSize/(inputSize*inputSize))
}

func checkLabelWidth(dims ort.Shape, n int) error {
	if len(dims) == 0 {
		return nil
	}
	if w := dims[len(dims)-1]; w > 0 && w != int64(n) {
		return fmt.Errorf("%d labels for %d outputs", n, w)
	}
	return nil
}

func infer(s *session, frame []byte) ([]float32, error) {
	tensor := acquireTensor(s.size)
	inputShape, err := buildTensor(frame, tensor, s.size, s.profile)
	if err != nil {
		releaseTensor(s.size, tensor)
		return nil, err
	}
	inputTensor, err := ort.NewTensor(inputShape, tensor)
	if err != nil {
		releaseTensor(s.size, tensor)
		return nil, err
	}
	// The pool gets the buffer back only after Destroy: the tensor aliases it.
	outputs := []ort.Value{nil}
	runErr := s.s.Run([]ort.Value{inputTensor}, outputs)
	_ = inputTensor.Destroy()
	releaseTensor(s.size, tensor)
	if runErr != nil {
		return nil, runErr
	}
	defer func() { _ = outputs[0].Destroy() }()
	outTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected output type: %T", outputs[0])
	}
	// GetData is ORT-owned memory that the deferred Destroy frees, so it
	// is copied out.
	data := outTensor.GetData()
	out := make([]float32, len(data))
	if s.profile.Activation == "logits" {
		for i, v := range data {
			out[i] = float32(1 / (1 + math.Exp(-float64(v))))
		}
		return out, nil
	}
	copy(out, data)
	return out, nil
}

func parseDirectMLDeviceID(v string) int {
	if v == "" {
		return 0
	}
	id, err := strconv.Atoi(v)
	if err != nil || id < 0 {
		logx.Warnf("infer: invalid MONWORKER_DIRECTML_DEVICE_ID %q, falling back to device 0", v)
		return 0
	}
	return id
}
