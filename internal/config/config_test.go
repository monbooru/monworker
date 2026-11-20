package config

import (
	"path/filepath"
	"testing"
)

func env(m map[string]string) func(string) string { return func(k string) string { return m[k] } }

func TestTheEnvironmentOverridesTheFlags(t *testing.T) {
	c, err := Parse([]string{"-provider", "cpu", "-parallel", "2"}, env(map[string]string{
		"MONWORKER_PROVIDER": "CUDA", "MONWORKER_PARALLEL": "6", "MONWORKER_THREADS": "3",
		"MONWORKER_NAME": "gpu-box", "MONWORKER_EXIT_ON_STDIN_EOF": "1", "MONWORKER_CONFIG": "/config/monworker.toml",
	}))
	if err != nil {
		t.Fatal(err)
	}
	if c.Provider != "cuda" || c.Parallel != 6 || c.Threads != 3 || c.Name != "gpu-box" || !c.ExitOnStdinEOF || c.Cache != "/config" {
		t.Errorf("config = %+v", c)
	}
	if c.Listen != "0.0.0.0:8457" {
		t.Errorf("listen = %s, want port 8457", c.Listen)
	}
}

func TestParseRefusesWhatCannotRun(t *testing.T) {
	for _, e := range []map[string]string{
		{"MONWORKER_PROVIDER": "vulkan"},
		{"MONWORKER_THREADS": "-1"},
		{"MONWORKER_LISTEN": "nowhere"},
	} {
		if _, err := Parse(nil, env(e)); err == nil {
			t.Errorf("%v: parsed", e)
		}
	}
}

func TestANameIsWhatAPairingLabelTakes(t *testing.T) {
	for in, want := range map[string]string{"gpu-box": "gpu-box", "gpu.lan": "gpu", "my box": "my-box", "": ""} {
		if got := sanitizeName(in); got != want {
			t.Errorf("sanitizeName(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestPairingsSurviveAReload(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sub", "monworker.toml")
	s, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range []Pairing{{URL: "http://a", PeerToken: "1"}, {URL: "http://b", PeerToken: "2"}, {URL: "http://a", PeerToken: "3"}} {
		if err := s.Put(p); err != nil {
			t.Fatal(err)
		}
	}
	again, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := again.Pairings(); len(got) != 2 || got[1].PeerToken != "3" {
		t.Errorf("pairings = %+v", got)
	}
	if found, err := again.RemoveByPeerToken("2"); err != nil || !found || len(again.Pairings()) != 1 {
		t.Errorf("remove: %v %v %+v", found, err, again.Pairings())
	}
	if err := again.Remove("http://a"); err != nil || len(again.Pairings()) != 0 {
		t.Errorf("remove by url: %v %+v", err, again.Pairings())
	}
}
