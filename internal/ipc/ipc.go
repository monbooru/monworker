// Package ipc is the private wire between the worker's front and its
// inference child: length-prefixed gob frames over a loopback socket,
// several calls in flight at once, each reply carrying its call's id.
package ipc

import (
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/binary"
	"encoding/gob"
	"fmt"
	"io"
	"net"
	"sync"
	"time"
)

type Method uint8

const (
	MethodRun Method = iota + 1
	MethodStatus
	MethodReleaseIdle
	MethodShutdown
	MethodCancel
)

// TokenEnv carries the spawn secret: the environment, not argv, since
// /proc/<pid>/environ is owner-only while cmdline is world-readable.
const TokenEnv = "MONWORKER_IPC_TOKEN"

const handshakeTimeout = 5 * time.Second

// RunRequest is one call: Frames, each a Size x Size RGB8 square, through
// each of Models, which share that input.
type RunRequest struct {
	Models []string
	Size   int
	Frames [][]byte
}

// FrameScores is one frame through one model: the label indexes at or
// above the score floor and their scores, or Err for a refused frame.
type FrameScores struct {
	I   []int32
	S   []float32
	Err string
}

// RunResponse.Results follows RunRequest.Models, one entry per frame.
type RunResponse struct {
	Results [][]FrameScores
}

type Status struct {
	Loaded   []string
	Provider string
	InUse    bool
	LastUsed time.Time
}

type Request struct {
	ID        uint64
	Method    Method
	Run       *RunRequest
	IdleAfter time.Duration
}

type Response struct {
	ID       uint64
	Run      *RunResponse
	Status   *Status
	Released bool
	Err      string
}

type hello struct{ Token string }

func writeFrame(w io.Writer, payload any) error {
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(payload); err != nil {
		return fmt.Errorf("gob encode: %w", err)
	}
	body := buf.Bytes()
	if len(body) > maxFrameBytes {
		return fmt.Errorf("ipc frame too large: %d bytes", len(body))
	}
	hdr := make([]byte, 4)
	binary.BigEndian.PutUint32(hdr, uint32(len(body)))
	if _, err := w.Write(hdr); err != nil {
		return err
	}
	_, err := w.Write(body)
	return err
}

// The largest call, 64 frames of 2048 px, is 768 MiB; the cap bounds what
// a corrupt length header can make readFrame allocate.
const maxFrameBytes = 1 << 30

func readFrame(r io.Reader, dst any) error {
	var hdr [4]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return err
	}
	n := binary.BigEndian.Uint32(hdr[:])
	if n > maxFrameBytes {
		return fmt.Errorf("frame body %d bytes exceeds cap %d", n, maxFrameBytes)
	}
	body := make([]byte, n)
	if _, err := io.ReadFull(r, body); err != nil {
		return fmt.Errorf("read frame body: %w", err)
	}
	return gob.NewDecoder(bytes.NewReader(body)).Decode(dst)
}

// Accept waits for the child's connection: any local process can reach
// the listener, so one that cannot echo the spawn secret is dropped.
func Accept(l net.Listener, token string) (net.Conn, error) {
	for {
		c, err := l.Accept()
		if err != nil {
			return nil, err
		}
		if handshakeOK(c, token) {
			return c, nil
		}
		_ = c.Close()
	}
}

func handshakeOK(c net.Conn, want string) bool {
	_ = c.SetReadDeadline(time.Now().Add(handshakeTimeout))
	var h hello
	if err := readFrame(c, &h); err != nil {
		return false
	}
	_ = c.SetReadDeadline(time.Time{})
	return subtle.ConstantTimeCompare([]byte(h.Token), []byte(want)) == 1
}

func Greet(c net.Conn, token string) error { return writeFrame(c, hello{Token: token}) }

// Conn carries the calls to one child and routes each reply to its
// caller; once its reader stops, every call fails with Err.
type Conn struct {
	net.Conn
	writeMu sync.Mutex
	mu      sync.Mutex
	nextID  uint64
	pending map[uint64]chan Response
	err     error
	done    chan struct{}
}

func NewConn(c net.Conn) *Conn {
	ic := &Conn{Conn: c, pending: map[uint64]chan Response{}, done: make(chan struct{})}
	go ic.readLoop()
	return ic
}

func (c *Conn) Done() <-chan struct{} { return c.done }

func (c *Conn) readLoop() {
	defer close(c.done)
	for {
		var resp Response
		if err := readFrame(c.Conn, &resp); err != nil {
			c.mu.Lock()
			c.err = fmt.Errorf("ipc read: %w", err)
			for id, ch := range c.pending {
				close(ch)
				delete(c.pending, id)
			}
			c.mu.Unlock()
			return
		}
		c.mu.Lock()
		ch := c.pending[resp.ID]
		delete(c.pending, resp.ID)
		c.mu.Unlock()
		if ch != nil {
			ch <- resp
		}
	}
}

func (c *Conn) send(req Request) error {
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	return writeFrame(c.Conn, req)
}

// Call returns ctx's error on a cancel, which the child is told of, and
// any other error when the connection is gone.
func (c *Conn) Call(ctx context.Context, req Request) (Response, error) {
	ch := make(chan Response, 1)
	c.mu.Lock()
	if c.err != nil {
		c.mu.Unlock()
		return Response{}, c.err
	}
	c.nextID++
	req.ID = c.nextID
	c.pending[req.ID] = ch
	c.mu.Unlock()
	forget := func() {
		c.mu.Lock()
		delete(c.pending, req.ID)
		c.mu.Unlock()
	}
	if err := c.send(req); err != nil {
		forget()
		_ = c.Close()
		return Response{}, fmt.Errorf("ipc write: %w", err)
	}
	select {
	case resp, ok := <-ch:
		if !ok {
			c.mu.Lock()
			defer c.mu.Unlock()
			return Response{}, c.err
		}
		return resp, nil
	case <-ctx.Done():
		forget()
		_ = c.send(Request{ID: req.ID, Method: MethodCancel})
		return Response{}, ctx.Err()
	}
}
