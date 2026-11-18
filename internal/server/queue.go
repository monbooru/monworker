package server

import (
	"context"
	"sync"
)

// queue admits calls up to the worker's slots; when one frees, an
// interactive call waiting goes before every batch call.
type queue struct {
	mu          sync.Mutex
	free        int
	interactive []chan struct{}
	batch       []chan struct{}
}

func newQueue(slots int) *queue { return &queue{free: slots} }

func (q *queue) admit(ctx context.Context, interactive bool) error {
	q.mu.Lock()
	if q.free > 0 && len(q.interactive) == 0 && (interactive || len(q.batch) == 0) {
		q.free--
		q.mu.Unlock()
		return nil
	}
	ch := make(chan struct{})
	if interactive {
		q.interactive = append(q.interactive, ch)
	} else {
		q.batch = append(q.batch, ch)
	}
	q.mu.Unlock()
	select {
	case <-ch:
		return nil
	case <-ctx.Done():
		q.mu.Lock()
		if q.drop(ch) {
			q.mu.Unlock()
			return ctx.Err()
		}
		q.mu.Unlock()
		// Admitted meanwhile: hand the slot on.
		q.done()
		return ctx.Err()
	}
}

// Caller holds q.mu.
func (q *queue) drop(ch chan struct{}) bool {
	for _, list := range []*[]chan struct{}{&q.interactive, &q.batch} {
		for i, c := range *list {
			if c == ch {
				*list = append((*list)[:i], (*list)[i+1:]...)
				return true
			}
		}
	}
	return false
}

func (q *queue) done() {
	q.mu.Lock()
	defer q.mu.Unlock()
	for _, list := range []*[]chan struct{}{&q.interactive, &q.batch} {
		if len(*list) > 0 {
			ch := (*list)[0]
			*list = (*list)[1:]
			close(ch)
			return
		}
	}
	q.free++
}

func (q *queue) busy() bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.interactive)+len(q.batch) > 0
}
