package pair

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/monbooru/monworker/internal/config"
	"github.com/monbooru/monworker/internal/shared/pairing"
)

type fakeMonbooru struct {
	mu      sync.Mutex
	offers  []pairing.RequestBody
	answers []string
	polls   int
}

func (f *fakeMonbooru) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	switch r.URL.Path {
	case "/api/v1/pair/request":
		var body pairing.RequestBody
		_ = json.NewDecoder(r.Body).Decode(&body)
		f.offers = append(f.offers, body)
		_ = json.NewEncoder(w).Encode(pairing.Accepted{RequestID: "r1", Status: pairing.Pending})
	case "/api/v1/pair/status":
		a := f.answers[min(f.polls, len(f.answers)-1)]
		f.polls++
		switch a {
		case "gone":
			w.WriteHeader(http.StatusNotFound)
		case "approved":
			_ = json.NewEncoder(w).Encode(pairing.StatusBody{Status: pairing.Approved, Token: "mb-token"})
		default:
			_ = json.NewEncoder(w).Encode(pairing.StatusBody{Status: pairing.State(a)})
		}
	}
}

func runOffer(t *testing.T, f *fakeMonbooru) (*config.Store, string) {
	t.Helper()
	srv := httptest.NewServer(f)
	t.Cleanup(srv.Close)
	store, err := config.Load(filepath.Join(t.TempDir(), "monworker.toml"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	Offer{Monbooru: srv.URL, Name: "gpu-box", Version: "v1.0.0", Self: "http://localhost:8457", Store: store, Poll: time.Millisecond, Retry: time.Millisecond}.Run(ctx)
	return store, srv.URL
}

func TestAnApprovalKeepsBothTokens(t *testing.T) {
	f := &fakeMonbooru{answers: []string{"pending", "pending", "approved"}}
	store, base := runOffer(t, f)
	p := store.Pairings()
	if len(p) != 1 || p[0].URL != base || p[0].Token != "mb-token" || p[0].PeerToken != f.offers[0].PeerToken {
		t.Fatalf("pairings = %+v", p)
	}
	o := f.offers[0]
	if o.App != "monworker" || o.Label != "gpu-box" || len(o.RequestedScopes) != 0 || o.URL != "http://localhost:8457" {
		t.Errorf("offer = %+v", o)
	}
}

func TestADroppedRequestIsOfferedAgainAndADenialStops(t *testing.T) {
	f := &fakeMonbooru{answers: []string{"pending", "gone", "pending", "denied"}}
	store, _ := runOffer(t, f)
	if len(f.offers) != 2 || f.offers[0].PeerToken == f.offers[1].PeerToken {
		t.Errorf("offers = %+v, want two, each its own token", f.offers)
	}
	if len(store.Pairings()) != 0 {
		t.Errorf("a denial was stored: %+v", store.Pairings())
	}
}

func TestAPairedWorkerOffersNothing(t *testing.T) {
	f := &fakeMonbooru{answers: []string{"approved"}}
	srv := httptest.NewServer(f)
	defer srv.Close()
	store, _ := config.Load(filepath.Join(t.TempDir(), "monworker.toml"))
	_ = store.Put(config.Pairing{URL: srv.URL + "/", Token: "t", PeerToken: "p"})
	Offer{Monbooru: srv.URL, Store: store}.Run(context.Background())
	if len(f.offers) != 0 {
		t.Errorf("offered %d times", len(f.offers))
	}
}

func TestMonbooruDownIsRetried(t *testing.T) {
	f := &fakeMonbooru{answers: []string{"approved"}}
	ln := httptest.NewUnstartedServer(f)
	addr := ln.Listener.Addr().String()
	store, _ := config.Load(filepath.Join(t.TempDir(), "monworker.toml"))
	_ = ln.Listener.Close()
	done := make(chan struct{})
	go func() {
		Offer{Monbooru: "http://" + addr, Name: "w", Store: store, Poll: time.Millisecond, Retry: 10 * time.Millisecond}.Run(context.Background())
		close(done)
	}()
	time.Sleep(50 * time.Millisecond)
	srv := httptest.NewUnstartedServer(f)
	l, err := net.Listen("tcp", addr)
	if err != nil {
		t.Skipf("port taken meanwhile: %v", err)
	}
	srv.Listener = l
	srv.Start()
	defer srv.Close()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the offer never got through")
	}
	if len(store.Pairings()) != 1 {
		t.Errorf("pairings = %+v", store.Pairings())
	}
}
