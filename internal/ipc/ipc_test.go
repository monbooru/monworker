package ipc

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net"
	"os/exec"
	"slices"
	"testing"
	"time"
)

func TestFramesRoundTrip(t *testing.T) {
	in := Request{ID: 7, Method: MethodRun, Run: &RunRequest{Models: []string{"a", "b"}, Size: 2, Frames: [][]byte{bytes.Repeat([]byte{9}, 12)}}}
	var buf bytes.Buffer
	if err := writeFrame(&buf, in); err != nil {
		t.Fatal(err)
	}
	var got Request
	if err := readFrame(&buf, &got); err != nil {
		t.Fatal(err)
	}
	if fmt.Sprint(got.ID, got.Method, *got.Run) != fmt.Sprint(in.ID, in.Method, *in.Run) {
		t.Errorf("got %+v, want %+v", got, in)
	}
}

func TestHandshakeAcceptsOnlyTheSpawnToken(t *testing.T) {
	for _, tc := range []struct {
		name  string
		greet func(c net.Conn)
		want  bool
	}{
		{"spawn token", func(c net.Conn) { _ = Greet(c, "tok") }, true},
		{"other token", func(c net.Conn) { _ = Greet(c, "nope") }, false},
		{"closed before greeting", func(c net.Conn) { _ = c.Close() }, false},
		{"oversized header", func(c net.Conn) {
			var hdr [4]byte
			binary.BigEndian.PutUint32(hdr[:], maxFrameBytes+1)
			_, _ = c.Write(hdr[:])
		}, false},
	} {
		server, client := net.Pipe()
		go tc.greet(client)
		if got := handshakeOK(server, "tok"); got != tc.want {
			t.Errorf("%s: handshakeOK = %v, want %v", tc.name, got, tc.want)
		}
		_ = server.Close()
		_ = client.Close()
	}
}

type fakeHandler struct{ started chan struct{} }

func (h fakeHandler) Run(ctx context.Context, req RunRequest) (RunResponse, error) {
	if req.Size == 0 {
		close(h.started)
		<-ctx.Done()
		return RunResponse{}, ctx.Err()
	}
	return RunResponse{Results: [][]FrameScores{{{Err: fmt.Sprint(req.Size)}}}}, nil
}
func (fakeHandler) Status() Status                   { return Status{Loaded: []string{"m"}} }
func (fakeHandler) ReleaseIdle(_ time.Duration) bool { return true }

func served(t *testing.T, h Handler) *Conn {
	t.Helper()
	front, child := net.Pipe()
	done := make(chan error, 1)
	go func() { done <- serve(context.Background(), child, h) }()
	c := NewConn(front)
	t.Cleanup(func() {
		_ = front.Close()
		if err := <-done; err != nil && !errors.Is(err, io.ErrClosedPipe) {
			t.Logf("serve: %v", err)
		}
	})
	return c
}

func TestCallsInFlightGetTheirOwnRepliesAndACancelStopsOnlyItsCall(t *testing.T) {
	h := fakeHandler{started: make(chan struct{})}
	c := served(t, h)
	ctx, cancel := context.WithCancel(context.Background())
	slow := make(chan error, 1)
	go func() {
		_, err := c.Call(ctx, Request{Method: MethodRun, Run: &RunRequest{}})
		slow <- err
	}()
	<-h.started
	for _, size := range []int{2, 1} {
		resp, err := c.Call(context.Background(), Request{Method: MethodRun, Run: &RunRequest{Size: size}})
		if err != nil || resp.Run.Results[0][0].Err != fmt.Sprint(size) {
			t.Errorf("size %d: %+v, %v", size, resp, err)
		}
	}
	cancel()
	if err := <-slow; !errors.Is(err, context.Canceled) {
		t.Errorf("cancelled call: %v", err)
	}
	if resp, err := c.Call(context.Background(), Request{Method: MethodStatus}); err != nil || !slices.Equal(resp.Status.Loaded, []string{"m"}) {
		t.Errorf("status after the cancel: %+v, %v", resp, err)
	}
}

func TestAChildGoneFailsTheCallAndIsDropped(t *testing.T) {
	front, child := net.Pipe()
	c := &Child{cmd: &exec.Cmd{}, conn: NewConn(front)}
	go func() {
		var req Request
		_ = readFrame(child, &req)
		_ = child.Close()
	}()
	if _, err := c.Run(context.Background(), RunRequest{}); err == nil {
		t.Fatal("Run: want an error once the child is gone")
	}
	if c.conn != nil {
		t.Error("the dead child's connection was kept")
	}
}

func TestAReleaseLeavesACallInFlightAlone(t *testing.T) {
	front, child := net.Pipe()
	t.Cleanup(func() { _ = front.Close(); _ = child.Close() })
	c := &Child{cmd: &exec.Cmd{}, conn: NewConn(front)}
	c.inFlight.Add(1)
	c.mu.RLock()
	defer func() {
		c.mu.RUnlock()
		c.inFlight.Add(-1)
	}()
	done := make(chan struct{})
	go func() {
		c.ReleaseIdle(0)
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("a release waited on the call in flight")
	}
	if c.conn == nil {
		t.Error("a release dropped the connection of the call in flight")
	}
}

func TestStatusWhileBusyAsksTheChild(t *testing.T) {
	h := fakeHandler{started: make(chan struct{})}
	c := &Child{cmd: &exec.Cmd{}, conn: served(t, h), Provider: "cpu"}
	ctx, cancel := context.WithCancel(context.Background())
	slow := make(chan struct{})
	go func() {
		_, _ = c.Run(ctx, RunRequest{Models: []string{"m"}})
		close(slow)
	}()
	<-h.started
	st := c.Status()
	cancel()
	<-slow
	if !st.InUse || !slices.Equal(st.Loaded, []string{"m"}) {
		t.Errorf("status while busy: %+v, want in use with m loaded", st)
	}
}

func TestStatusWhileBusyKeepsTheSnapshotWhenTheChildIsSilent(t *testing.T) {
	front, child := net.Pipe()
	t.Cleanup(func() { _ = front.Close(); _ = child.Close() })
	go func() { _, _ = io.Copy(io.Discard, child) }()
	c := &Child{cmd: &exec.Cmd{}, conn: NewConn(front)}
	c.lastStatus.Store(&Status{Loaded: []string{"m"}, Provider: "cuda"})
	c.inFlight.Add(1)
	defer c.inFlight.Add(-1)
	start := time.Now()
	st := c.Status()
	if !st.InUse || !slices.Equal(st.Loaded, []string{"m"}) || st.Provider != "cuda" {
		t.Errorf("status of a silent child: %+v, want the snapshot in use", st)
	}
	if d := time.Since(start); d > 3*busyStatusTimeout {
		t.Errorf("status waited %s on a silent child", d)
	}
}
