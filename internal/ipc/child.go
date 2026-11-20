package ipc

import (
	"cmp"
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"log"
	"net"
	"os"
	"os/exec"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/monbooru/monworker/internal/shared/logx"
)

// Child runs inference in a child process of the worker: only its exit
// gives back the GPU libraries and the context ONNX Runtime loaded. It is
// started on the first call.
type Child struct {
	Args []string
	// A call holds mu for reading; starting and stopping the child hold
	// it for writing.
	mu         sync.RWMutex
	cmd        *exec.Cmd
	conn       *Conn
	inFlight   atomic.Int32
	pid        atomic.Int32
	lastStatus atomic.Pointer[Status]
	Provider   string
}

// Caller must hold c.mu.
func (c *Child) ensureRunning() error {
	if c.cmd != nil && c.conn != nil {
		return nil
	}
	token := rand.Text()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return fmt.Errorf("listen tcp: %w", err)
	}
	defer func() { _ = listener.Close() }()
	addr := listener.Addr().String()

	exe, err := os.Executable()
	if err != nil {
		return fmt.Errorf("resolve own executable: %w", err)
	}
	cmd := exec.Command(exe, append(c.Args, "--addr="+addr)...)
	cmd.Env = append(os.Environ(), TokenEnv+"="+token)
	cmd.Stdout = os.Stdout
	cmd.Stderr = log.Writer()
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start the inference child: %w", err)
	}
	logx.Infof("infer: child started pid=%d", cmd.Process.Pid)

	type accepted struct {
		conn net.Conn
		err  error
	}
	got := make(chan accepted, 1)
	go func() {
		conn, err := Accept(listener, token)
		got <- accepted{conn, err}
	}()
	select {
	case a := <-got:
		if a.err != nil {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
			return fmt.Errorf("accept the inference child: %w", a.err)
		}
		c.cmd, c.conn = cmd, NewConn(a.conn)
		c.pid.Store(int32(cmd.Process.Pid))
		return nil
	case <-time.After(15 * time.Second):
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return errors.New("the inference child did not connect within 15s")
	}
}

// Caller must hold c.mu for writing.
func (c *Child) terminate() {
	if c.cmd == nil {
		return
	}
	if c.cmd.Process != nil {
		_ = c.cmd.Process.Signal(syscall.SIGTERM)
	}
	c.reap(true, "terminated")
}

// dropConnFirst is for an unresponsive child; a graceful shutdown keeps
// the socket open until the child has exited.
func (c *Child) reap(dropConnFirst bool, verb string) {
	if dropConnFirst && c.conn != nil {
		_ = c.conn.Close()
		c.conn = nil
	}
	done := make(chan struct{})
	go func() {
		if c.cmd != nil {
			_ = c.cmd.Wait()
		}
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		if c.cmd != nil && c.cmd.Process != nil {
			_ = c.cmd.Process.Kill()
		}
		<-done
	}
	if c.conn != nil {
		_ = c.conn.Close()
		c.conn = nil
	}
	pid := 0
	var ps *os.ProcessState
	if c.cmd != nil && c.cmd.ProcessState != nil {
		ps = c.cmd.ProcessState
		pid = ps.Pid()
	}
	c.cmd = nil
	c.pid.Store(0)
	c.lastStatus.Store(nil)
	if ps != nil && !ps.Success() {
		if ws, ok := ps.Sys().(syscall.WaitStatus); !ok || !ws.Signaled() || ws.Signal() != syscall.SIGTERM {
			logx.Warnf("infer: child %s pid=%d: %s", verb, pid, ps)
			return
		}
	}
	logx.Infof("infer: child %s pid=%d", verb, pid)
}

// Well above a teardown, short enough that a wedged child can't stall
// the idle release or a shutdown.
const shortCallTimeout = 30 * time.Second

// Caller must hold c.mu for writing. A failed short call takes the child
// down.
func (c *Child) shortCall(req Request) (Response, error) {
	ctx, cancel := context.WithTimeout(context.Background(), shortCallTimeout)
	defer cancel()
	resp, err := c.conn.Call(ctx, req)
	if err != nil {
		c.terminate()
	}
	return resp, err
}

func (c *Child) Run(ctx context.Context, req RunRequest) (RunResponse, error) {
	c.inFlight.Add(1)
	defer c.inFlight.Add(-1)
	conn, err := c.acquire()
	if err != nil {
		return RunResponse{}, err
	}
	resp, err := conn.Call(ctx, Request{Method: MethodRun, Run: &req})
	c.mu.RUnlock()
	if err != nil {
		if ctx.Err() != nil {
			return RunResponse{}, ctx.Err()
		}
		c.drop(conn)
		return RunResponse{}, err
	}
	if resp.Err != "" {
		return RunResponse{}, errors.New(resp.Err)
	}
	if resp.Run == nil {
		return RunResponse{}, errors.New("the inference child returned an empty response")
	}
	return *resp.Run, nil
}

// acquire returns the live connection with c.mu held for reading,
// starting the child first when there is none.
func (c *Child) acquire() (*Conn, error) {
	for {
		c.mu.RLock()
		if c.conn != nil {
			return c.conn, nil
		}
		c.mu.RUnlock()
		c.mu.Lock()
		err := c.ensureRunning()
		c.mu.Unlock()
		if err != nil {
			return nil, err
		}
	}
}

// drop takes down a child whose connection failed, unless another call
// already replaced it.
func (c *Child) drop(conn *Conn) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.conn == conn {
		c.terminate()
	}
}

// A busy child answers between two frames, but not while it loads a
// model, which can take longer than a caller of /v1/info waits.
const busyStatusTimeout = time.Second

// Status asks the child, a busy one alongside its calls; one that does not
// answer in time is given its last snapshot. A child not running reports
// nothing loaded.
func (c *Child) Status() Status {
	if c.inFlight.Load() > 0 {
		return c.busyStatus()
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.conn == nil {
		return Status{}
	}
	resp, err := c.shortCall(Request{Method: MethodStatus})
	if err != nil || resp.Status == nil {
		return Status{}
	}
	snap := *resp.Status
	c.lastStatus.Store(&snap)
	return snap
}

// busyStatus takes no lock it would wait for: a start or a stop waiting on
// the calls in flight would hold it up as long as they run.
func (c *Child) busyStatus() Status {
	snap := Status{Provider: cmp.Or(c.Provider, "cpu")}
	if cached := c.lastStatus.Load(); cached != nil {
		snap = *cached
	}
	if c.mu.TryRLock() {
		if c.conn != nil {
			ctx, cancel := context.WithTimeout(context.Background(), busyStatusTimeout)
			resp, err := c.conn.Call(ctx, Request{Method: MethodStatus})
			cancel()
			if err == nil && resp.Status != nil {
				fresh := *resp.Status
				c.lastStatus.Store(&fresh)
				snap = fresh
			}
		}
		c.mu.RUnlock()
	}
	snap.InUse = true
	return snap
}

func (c *Child) ReleaseIdle(after time.Duration) bool {
	if c.inFlight.Load() > 0 {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.conn == nil {
		return false
	}
	resp, err := c.shortCall(Request{Method: MethodReleaseIdle, IdleAfter: after})
	if err != nil || !resp.Released {
		return false
	}
	c.shutdownLocked()
	return true
}

func (c *Child) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.shutdownLocked()
}

// Caller must hold c.mu for writing.
func (c *Child) shutdownLocked() {
	if c.conn == nil {
		return
	}
	resp, err := c.shortCall(Request{Method: MethodShutdown})
	if err != nil {
		return
	}
	if resp.Err != "" {
		c.terminate()
		return
	}
	c.reap(false, "stopped")
}

func (c *Child) PID() int { return int(c.pid.Load()) }
