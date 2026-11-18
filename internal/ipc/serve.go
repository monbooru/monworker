package ipc

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"sync"
	"time"
)

type Handler interface {
	Run(ctx context.Context, req RunRequest) (RunResponse, error)
	Status() Status
	ReleaseIdle(after time.Duration) bool
}

// Serve dials the front at addr, greets it with token and answers its
// calls until the front shuts it down or goes away. Each call runs in its
// own goroutine; the handler bounds how many infer at once.
func Serve(ctx context.Context, addr, token string, h Handler) error {
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		return fmt.Errorf("dial %q: %w", addr, err)
	}
	defer func() { _ = conn.Close() }()
	if err := Greet(conn, token); err != nil {
		return fmt.Errorf("handshake: %w", err)
	}
	return serve(ctx, conn, h)
}

func serve(ctx context.Context, conn net.Conn, h Handler) error {
	var wg sync.WaitGroup
	defer wg.Wait()
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var writeMu sync.Mutex
	write := func(resp Response) error {
		writeMu.Lock()
		defer writeMu.Unlock()
		return writeFrame(conn, resp)
	}
	var callsMu sync.Mutex
	calls := map[uint64]context.CancelFunc{}
	for {
		var req Request
		if err := readFrame(conn, &req); err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return fmt.Errorf("read: %w", err)
		}
		switch req.Method {
		case MethodShutdown:
			return write(Response{ID: req.ID})
		case MethodCancel:
			callsMu.Lock()
			if stop, ok := calls[req.ID]; ok {
				stop()
			}
			callsMu.Unlock()
		case MethodRun:
			callCtx, stop := context.WithCancel(ctx)
			callsMu.Lock()
			calls[req.ID] = stop
			callsMu.Unlock()
			wg.Go(func() {
				resp := Response{ID: req.ID}
				if req.Run == nil {
					resp.Err = "run: empty request"
				} else if run, err := h.Run(callCtx, *req.Run); err != nil {
					resp.Err = err.Error()
				} else {
					resp.Run = &run
				}
				callsMu.Lock()
				delete(calls, req.ID)
				callsMu.Unlock()
				stop()
				_ = write(resp)
			})
		case MethodStatus:
			st := h.Status()
			if err := write(Response{ID: req.ID, Status: &st}); err != nil {
				return err
			}
		case MethodReleaseIdle:
			if err := write(Response{ID: req.ID, Released: h.ReleaseIdle(req.IdleAfter)}); err != nil {
				return err
			}
		default:
			if err := write(Response{ID: req.ID, Err: fmt.Sprintf("unknown method %d", req.Method)}); err != nil {
				return err
			}
		}
	}
}
