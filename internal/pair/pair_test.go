package pair

import (
	"bytes"
	"context"
	"encoding/json"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/monbooru/monworker/internal/config"
	"github.com/monbooru/monworker/internal/shared/contract"
	"github.com/monbooru/monworker/internal/shared/pairing"
)

type fakeMonbooru struct {
	mu       sync.Mutex
	offers   []pairing.RequestBody
	refusals []int
	answers  []string
	polls    int
	checks   []int
	checked  []pairing.RequestBody
	auths    []string
}

func (f *fakeMonbooru) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	switch r.URL.Path {
	case "/api/v1/pair/request":
		var body pairing.RequestBody
		_ = json.NewDecoder(r.Body).Decode(&body)
		f.offers = append(f.offers, body)
		if n := len(f.offers); n <= len(f.refusals) {
			w.WriteHeader(f.refusals[n-1])
			_ = json.NewEncoder(w).Encode(contract.Error{Error: "the name gpu-box is taken", Code: "invalid_request"})
			return
		}
		_ = json.NewEncoder(w).Encode(pairing.Accepted{RequestID: "r1", Status: pairing.Pending})
	case "/api/v1/pair/check":
		var body pairing.RequestBody
		_ = json.NewDecoder(r.Body).Decode(&body)
		f.checked = append(f.checked, body)
		f.auths = append(f.auths, r.Header.Get("Authorization"))
		if len(f.checks) > 0 {
			if c := f.checks[min(len(f.checked), len(f.checks))-1]; c != http.StatusOK {
				w.WriteHeader(c)
				return
			}
		}
		_ = json.NewEncoder(w).Encode(pairing.StatusBody{Status: pairing.Approved})
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

func TestARefusedOfferStops(t *testing.T) {
	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(os.Stderr)
	f := &fakeMonbooru{refusals: []int{http.StatusBadRequest}, answers: []string{"approved"}}
	store, base := runOffer(t, f)
	if len(f.offers) != 1 || len(store.Pairings()) != 0 {
		t.Errorf("offers = %d, pairings = %+v", len(f.offers), store.Pairings())
	}
	if want := "pair: " + base + " refused the offer: the name gpu-box is taken; restart the worker to offer again"; !strings.Contains(buf.String(), want) {
		t.Errorf("log = %q, want %q", buf.String(), want)
	}
}

func TestAnOfferMonbooruCannotTakeNowIsMadeAgain(t *testing.T) {
	f := &fakeMonbooru{refusals: []int{http.StatusServiceUnavailable, http.StatusTooManyRequests, http.StatusNotFound}, answers: []string{"approved"}}
	store, _ := runOffer(t, f)
	if len(f.offers) != 4 || len(store.Pairings()) != 1 {
		t.Errorf("offers = %d, pairings = %+v", len(f.offers), store.Pairings())
	}
}

func TestAPairedWorkerOffersAgainOnlyWhenMonbooruForgotIt(t *testing.T) {
	for _, tc := range []struct {
		checks []int
		token  string
		offers int
	}{
		{nil, "t", 0},
		{[]int{http.StatusNotFound}, "t", 0},
		{[]int{http.StatusForbidden}, "t", 0},
		{[]int{http.StatusServiceUnavailable, http.StatusOK}, "t", 0},
		{[]int{http.StatusUnauthorized}, "mb-token", 1},
	} {
		f := &fakeMonbooru{checks: tc.checks, answers: []string{"approved"}}
		srv := httptest.NewServer(f)
		store, _ := config.Load(filepath.Join(t.TempDir(), "monworker.toml"))
		_ = store.Put(config.Pairing{URL: srv.URL + "/", Token: "t", PeerToken: "p"})
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		Offer{Monbooru: srv.URL, Name: "gpu-box", Version: "v1.0.0", Self: "http://localhost:8457", Store: store, Poll: time.Millisecond, Retry: time.Millisecond}.Run(ctx)
		cancel()
		srv.Close()
		if p := store.Pairings(); len(p) != 1 || p[0].Token != tc.token {
			t.Errorf("checks %v: pairings = %+v, want monbooru's token %s", tc.checks, p, tc.token)
		}
		if len(f.offers) != tc.offers {
			t.Errorf("checks %v: offered %d times, want %d", tc.checks, len(f.offers), tc.offers)
		}
		if len(f.checked) != max(1, len(tc.checks)) || f.auths[0] != "Bearer t" {
			t.Fatalf("checks %v: checked %d times, with %q", tc.checks, len(f.checked), f.auths)
		}
		if c := f.checked[0]; c.App != "monworker" || c.Label != "gpu-box" || c.URL != "http://localhost:8457" || c.Version != "v1.0.0" || c.RequestedScopes == nil || c.PeerToken != "" {
			t.Errorf("checks %v: body = %+v", tc.checks, c)
		}
	}
}

func TestMonbooruDownIsRetried(t *testing.T) {
	for _, paired := range []bool{false, true} {
		f := &fakeMonbooru{answers: []string{"approved"}}
		ln := httptest.NewUnstartedServer(f)
		addr := ln.Listener.Addr().String()
		store, _ := config.Load(filepath.Join(t.TempDir(), "monworker.toml"))
		if paired {
			_ = store.Put(config.Pairing{URL: "http://" + addr, Token: "t", PeerToken: "p"})
		}
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
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Fatalf("paired %v: monbooru was never reached", paired)
		}
		srv.Close()
		if p := store.Pairings(); len(p) != 1 || (paired && (p[0].Token != "t" || len(f.offers) != 0 || len(f.checked) != 1)) {
			t.Errorf("paired %v: pairings = %+v, %d offers, %d checks", paired, p, len(f.offers), len(f.checked))
		}
	}
}
