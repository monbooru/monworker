// Package server is the worker's front: it serves monbooru's calls over
// HTTP, lists the model folders, queues the tag calls and hands them to
// the inference child. It never loads ONNX Runtime.
package server

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"time"

	"github.com/monbooru/monworker/internal/config"
	"github.com/monbooru/monworker/internal/ipc"
	"github.com/monbooru/monworker/internal/models"
	"github.com/monbooru/monworker/internal/shared/contract"
	"github.com/monbooru/monworker/internal/shared/logx"
	"github.com/monbooru/monworker/internal/shared/worker"
)

// APIVersion moves on any change to /v1 a paired monbooru would trip
// over; MinPeerAPI is the oldest monbooru API it still answers.
const (
	APIVersion = 1
	MinPeerAPI = 1
)

type Backend interface {
	Run(ctx context.Context, req ipc.RunRequest) (ipc.RunResponse, error)
	Status() ipc.Status
	ReleaseIdle(after time.Duration) bool
}

type Server struct {
	cfg     config.Config
	version string
	backend Backend
	store   *config.Store
	digests *models.Digests
	queue   *queue
	calls   atomic.Int32
	mux     *http.ServeMux
}

func New(cfg config.Config, version string, backend Backend, store *config.Store, digests *models.Digests) *Server {
	s := &Server{cfg: cfg, version: version, backend: backend, store: store, digests: digests, queue: newQueue(cfg.Parallel), mux: http.NewServeMux()}
	s.mux.HandleFunc("GET /health", s.health)
	s.mux.HandleFunc("GET "+worker.PathInfo, s.auth(s.info))
	s.mux.HandleFunc("GET "+worker.PathModels, s.auth(s.models))
	s.mux.HandleFunc("GET "+worker.PathModels+"/{name}/labels", s.auth(s.labels))
	s.mux.HandleFunc("GET "+worker.PathCatalog, s.auth(s.catalog))
	s.mux.HandleFunc("POST "+worker.PathTag, s.auth(s.tag))
	s.mux.HandleFunc("POST /api/v1/pair/remove", s.auth(s.pairRemove))
	return s
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	s.mux.ServeHTTP(w, r)
	logx.Debugf("%s %s %s", r.Method, r.URL.Path, time.Since(start).Round(time.Millisecond))
}

func bearer(r *http.Request) string {
	tok, _ := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
	return strings.TrimSpace(tok)
}

// allowed is the run's secret or a token a paired monbooru calls with.
func (s *Server) allowed(tok string) bool {
	if tok == "" {
		return false
	}
	if s.cfg.Token != "" && subtle.ConstantTimeCompare([]byte(tok), []byte(s.cfg.Token)) == 1 {
		return true
	}
	return slices.ContainsFunc(s.store.Pairings(), func(p config.Pairing) bool {
		return p.PeerToken != "" && subtle.ConstantTimeCompare([]byte(tok), []byte(p.PeerToken)) == 1
	})
}

func (s *Server) auth(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !s.allowed(bearer(r)) {
			writeError(w, http.StatusUnauthorized, worker.CodeUnauthorized, "a token from a pairing is required")
			return
		}
		h(w, r)
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, code, msg string) {
	writeJSON(w, status, worker.Error{Error: msg, Code: code})
}

func (s *Server) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, contract.Health{App: contract.AppMonworker, Status: "ok", Version: s.version, API: APIVersion, MinPeerAPI: MinPeerAPI})
}

func (s *Server) info(w http.ResponseWriter, _ *http.Request) {
	st := s.backend.Status()
	writeJSON(w, http.StatusOK, worker.Info{
		Name:      s.cfg.Name,
		Version:   s.version,
		Provider:  s.cfg.Provider,
		Providers: []string{s.cfg.Provider},
		Threads:   s.cfg.Threads,
		Slots:     s.cfg.Parallel,
		ModelsDir: absDir(s.cfg.Models),
		Tasks:     []string{worker.TaskTag},
		Loaded:    nonNil(st.Loaded),
		Busy:      st.InUse || s.queue.busy(),
	})
}

func nonNil(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}

func (s *Server) models(w http.ResponseWriter, _ *http.Request) {
	out := []worker.Model{}
	for _, f := range models.Scan(s.cfg.Models) {
		out = append(out, f.Describe(s.cfg.Models, s.digests.Of(f)))
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) labels(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if models.ValidateName(name) != nil {
		writeError(w, http.StatusNotFound, worker.CodeUnknownModel, "no model "+name)
		return
	}
	f := models.Open(s.cfg.Models, name)
	l, err := f.Labels(s.cfg.Models)
	if err != nil {
		writeError(w, http.StatusNotFound, worker.CodeUnknownModel, err.Error())
		return
	}
	w.Header().Set("ETag", l.ETag)
	if r.Header.Get("If-None-Match") == l.ETag {
		w.WriteHeader(http.StatusNotModified)
		return
	}
	writeJSON(w, http.StatusOK, l)
}

func (s *Server) catalog(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, nonNilCatalog(models.Catalog(s.cfg.Models)))
}

func nonNilCatalog(c []worker.CatalogEntry) []worker.CatalogEntry {
	if c == nil {
		return []worker.CatalogEntry{}
	}
	return c
}

func (s *Server) tag(w http.ResponseWriter, r *http.Request) {
	req, frames, err := worker.ReadTag(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, worker.CodeBadRequest, err.Error())
		return
	}
	for _, name := range req.Models {
		if models.ValidateName(name) != nil {
			writeError(w, http.StatusNotFound, worker.CodeUnknownModel, "no model "+name)
			return
		}
		f := models.Open(s.cfg.Models, name)
		if f.Err != nil {
			writeError(w, http.StatusConflict, worker.CodeNotReady, name+": "+f.Err.Error())
			return
		}
		size, err := f.InputSide()
		if err != nil {
			writeError(w, http.StatusConflict, worker.CodeNotReady, name+": "+err.Error())
			return
		}
		if size != req.Frames.Size {
			writeError(w, http.StatusBadRequest, worker.CodeBadRequest, fmt.Sprintf("model %s takes frames of %d px", name, size))
			return
		}
	}
	ctx := r.Context()
	if err := s.queue.admit(ctx, req.Priority == worker.PriorityInteractive); err != nil {
		return
	}
	s.calls.Add(1)
	resp, err := s.backend.Run(ctx, ipc.RunRequest{Models: req.Models, Size: req.Frames.Size, Frames: frames})
	s.calls.Add(-1)
	s.queue.done()
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return
		}
		logx.Warnf("tag: %v", err)
		writeError(w, http.StatusInternalServerError, "", err.Error())
		return
	}
	out := worker.TagResponse{Results: make([]worker.ModelScores, len(req.Models))}
	for m, name := range req.Models {
		out.Results[m].Model = name
		out.Results[m].Frames = make([]worker.FrameScores, len(resp.Results[m]))
		for f, fr := range resp.Results[m] {
			out.Results[m].Frames[f] = worker.FrameScores{I: nonNilI(fr.I), S: nonNilS(fr.S), Err: fr.Err}
		}
	}
	writeJSON(w, http.StatusOK, out)
}

func nonNilI(v []int32) []int32 {
	if v == nil {
		return []int32{}
	}
	return v
}

func nonNilS(v []float32) []float32 {
	if v == nil {
		return []float32{}
	}
	return v
}

func (s *Server) pairRemove(w http.ResponseWriter, r *http.Request) {
	found, err := s.store.RemoveByPeerToken(bearer(r))
	if err != nil {
		writeError(w, http.StatusInternalServerError, "", err.Error())
		return
	}
	if found {
		logx.Infof("pair: monbooru removed this worker")
	}
	w.WriteHeader(http.StatusNoContent)
}

// Idle runs the idle release until ctx ends: the child is stopped once
// its models have gone unused for idle, 0 meaning half a minute.
func (s *Server) Idle(ctx context.Context, idle time.Duration) {
	every := time.Minute
	if idle <= 0 {
		idle, every = 30*time.Second, 10*time.Second
	}
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if s.calls.Load() == 0 && s.backend.ReleaseIdle(idle) {
				logx.Infof("infer: models unloaded after %s idle", idle)
			}
		}
	}
}

func absDir(dir string) string {
	if abs, err := filepath.Abs(dir); err == nil {
		return abs
	}
	return dir
}
