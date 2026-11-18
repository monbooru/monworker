package pair

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/monbooru/monworker/internal/config"
	"github.com/monbooru/monworker/internal/shared/contract"
	"github.com/monbooru/monworker/internal/shared/logx"
	"github.com/monbooru/monworker/internal/shared/pairing"
)

var client = &http.Client{
	Timeout:       15 * time.Second,
	CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
}

type Offer struct {
	Monbooru string
	Name     string
	Version  string
	// Self is the address monbooru calls the worker at; its host is
	// replaced by the source monbooru sees, its port kept.
	Self  string
	Store *config.Store
	// Poll and Retry pace the loop; zero takes the defaults.
	Poll, Retry time.Duration
}

var errDenied = errors.New("monbooru denied the pairing")

// Run offers once per start until the operator answers, renewing an offer
// monbooru dropped and retrying while monbooru is down; it returns when
// the worker is paired, denied, or ctx ends. A worker already paired
// with that monbooru offers nothing.
func (o Offer) Run(ctx context.Context) {
	base := strings.TrimRight(o.Monbooru, "/")
	for _, p := range o.Store.Pairings() {
		if strings.TrimRight(p.URL, "/") == base {
			return
		}
	}
	poll, retry := o.Poll, o.Retry
	if poll == 0 {
		poll = 2 * time.Second
	}
	if retry == 0 {
		retry = 5 * time.Second
	}
	logx.Infof("pair: offering this worker to %s as %s", base, o.Name)
	wait := retry
	for {
		err := o.offer(ctx, base, poll)
		switch {
		case err == nil:
			logx.Infof("pair: paired with %s", base)
			return
		case errors.Is(err, errDenied):
			logx.Warnf("pair: %s denied the pairing; restart the worker to offer again", base)
			return
		case ctx.Err() != nil:
			return
		}
		logx.Infof("pair: %v; trying again in %s", err, wait)
		select {
		case <-ctx.Done():
			return
		case <-time.After(wait):
		}
		wait = min(2*wait, time.Minute)
	}
}

// offer makes one request and polls it to an answer; a request monbooru
// dropped (its five minutes, a restart) is an error the loop renews.
func (o Offer) offer(ctx context.Context, base string, poll time.Duration) error {
	peerToken := rand.Text()
	body, _ := json.Marshal(pairing.RequestBody{
		App: contract.AppMonworker, Label: o.Name, URL: o.Self,
		RequestedScopes: []string{}, PeerToken: peerToken, Version: o.Version,
	})
	var accepted pairing.Accepted
	if status, err := call(ctx, http.MethodPost, base+"/api/v1/pair/request", body, &accepted); err != nil {
		return err
	} else if status != http.StatusOK || accepted.RequestID == "" {
		return fmt.Errorf("monbooru answered the offer with %d", status)
	}
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(poll):
		}
		var st pairing.StatusBody
		status, err := call(ctx, http.MethodGet, base+"/api/v1/pair/status?id="+url.QueryEscape(accepted.RequestID), nil, &st)
		if err != nil {
			return err
		}
		switch {
		case status == http.StatusNotFound:
			return errors.New("monbooru dropped the request")
		case status != http.StatusOK:
			return fmt.Errorf("monbooru answered the poll with %d", status)
		case st.Status == pairing.Denied || st.Status == pairing.Removed:
			return errDenied
		case st.Status == pairing.Approved && st.Token != "":
			return o.Store.Put(config.Pairing{URL: base, Token: st.Token, PeerToken: peerToken})
		}
	}
}

func call(ctx context.Context, method, u string, body []byte, out any) (int, error) {
	req, err := http.NewRequestWithContext(ctx, method, u, bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := client.Do(req)
	if err != nil {
		return 0, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode == http.StatusOK {
		_ = json.NewDecoder(resp.Body).Decode(out)
	}
	return resp.StatusCode, nil
}
