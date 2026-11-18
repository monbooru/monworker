package infer

import (
	"context"
	"sync"
)

// budget hands out the worker's threads in turn: a call waits until its
// share is free and every call before it has been served.
type budget struct {
	mu      sync.Mutex
	total   int
	free    int
	waiters []*waiter
}

type waiter struct {
	n     int
	ready chan struct{}
}

func newBudget(total int) *budget { return &budget{total: total, free: total} }

func (b *budget) acquire(ctx context.Context, n int) error {
	n = min(n, b.total)
	b.mu.Lock()
	if len(b.waiters) == 0 && b.free >= n {
		b.free -= n
		b.mu.Unlock()
		return nil
	}
	w := &waiter{n: n, ready: make(chan struct{})}
	b.waiters = append(b.waiters, w)
	b.mu.Unlock()
	select {
	case <-w.ready:
		return nil
	case <-ctx.Done():
		b.mu.Lock()
		for i, q := range b.waiters {
			if q == w {
				b.waiters = append(b.waiters[:i], b.waiters[i+1:]...)
				b.mu.Unlock()
				return ctx.Err()
			}
		}
		b.mu.Unlock()
		// Granted meanwhile: give it back.
		b.release(n)
		return ctx.Err()
	}
}

func (b *budget) release(n int) {
	n = min(n, b.total)
	b.mu.Lock()
	defer b.mu.Unlock()
	b.free += n
	for len(b.waiters) > 0 && b.free >= b.waiters[0].n {
		w := b.waiters[0]
		b.waiters = b.waiters[1:]
		b.free -= w.n
		close(w.ready)
	}
}
