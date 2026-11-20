package config

import (
	"cmp"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/BurntSushi/toml"
	"github.com/monbooru/monworker/internal/shared/contract"
	"github.com/monbooru/monworker/internal/shared/fsx"
)

var Providers = []string{"cpu", "cuda", "directml", "tensorrt", "openvino", "coreml", "coremlv2"}

type Config struct {
	Listen string
	// Advertise is the address the worker gives monbooru to call it at;
	// empty, monbooru takes the source of the offer.
	Advertise   string
	Models      string
	Cache       string
	Provider    string
	Threads     int
	Parallel    int
	Name        string
	MonbooruURL string
	IdleMinutes int
	// Token, set by a monbooru that starts the worker itself, opens every
	// route for that run without a pairing.
	Token          string
	ExitOnStdinEOF bool
	LogLevel       string
	File           string
	AddrFile       string
}

// Parse reads argv, then lets the environment override what it sets.
func Parse(argv []string, getenv func(string) string) (Config, error) {
	fs := flag.NewFlagSet("monworker", flag.ContinueOnError)
	c := Config{}
	fs.StringVar(&c.File, "config", "./monworker.toml", "where the pairings are kept")
	fs.StringVar(&c.Listen, "listen", "0.0.0.0:"+contract.PortMonworker, "address to listen on")
	fs.StringVar(&c.Advertise, "advertise", "", "address monbooru calls the worker at (default: the one it sees)")
	fs.StringVar(&c.Models, "models", "./models", "models directory")
	fs.StringVar(&c.Cache, "cache", "", "the providers' compile caches (default: beside -config)")
	fs.StringVar(&c.Provider, "provider", "cpu", "execution provider: "+strings.Join(Providers, ", "))
	fs.IntVar(&c.Threads, "threads", 0, "threads ONNX Runtime may use, 0 for every core")
	fs.IntVar(&c.Parallel, "parallel", 4, "images tagged at once")
	fs.StringVar(&c.Name, "name", "", "the worker's name in monbooru (default: the host name)")
	fs.StringVar(&c.MonbooruURL, "monbooru", "", "monbooru's address, to pair with")
	fs.IntVar(&c.IdleMinutes, "idle", 15, "minutes idle before the models are unloaded, 0 is half a minute")
	fs.StringVar(&c.LogLevel, "log", "info", "log level")
	if err := fs.Parse(argv); err != nil {
		return c, err
	}
	for _, s := range []struct {
		key string
		dst *string
	}{
		{"MONWORKER_CONFIG", &c.File}, {"MONWORKER_LISTEN", &c.Listen}, {"MONWORKER_ADVERTISE", &c.Advertise},
		{"MONWORKER_MODELS", &c.Models}, {"MONWORKER_CACHE", &c.Cache}, {"MONWORKER_PROVIDER", &c.Provider},
		{"MONWORKER_NAME", &c.Name}, {"MONWORKER_MONBOORU_URL", &c.MonbooruURL}, {"MONWORKER_LOG", &c.LogLevel},
		{"MONWORKER_TOKEN", &c.Token}, {"MONWORKER_ADDR_FILE", &c.AddrFile},
	} {
		if v := strings.TrimSpace(getenv(s.key)); v != "" {
			*s.dst = v
		}
	}
	for _, n := range []struct {
		key string
		dst *int
	}{
		{"MONWORKER_THREADS", &c.Threads}, {"MONWORKER_PARALLEL", &c.Parallel}, {"MONWORKER_IDLE_MINUTES", &c.IdleMinutes},
	} {
		if v := strings.TrimSpace(getenv(n.key)); v != "" {
			i, err := strconv.Atoi(v)
			if err != nil || i < 0 {
				return c, fmt.Errorf("%s=%q is not a whole number", n.key, v)
			}
			*n.dst = i
		}
	}
	c.ExitOnStdinEOF = getenv("MONWORKER_EXIT_ON_STDIN_EOF") == "1"
	c.Provider = strings.ToLower(c.Provider)
	if !slices.Contains(Providers, c.Provider) {
		return c, fmt.Errorf("provider %q is not one of %s", c.Provider, strings.Join(Providers, ", "))
	}
	if _, _, err := net.SplitHostPort(c.Listen); err != nil {
		return c, fmt.Errorf("listen address %q: %w", c.Listen, err)
	}
	if c.Threads == 0 {
		// GOMAXPROCS, not NumCPU: a container's CPU limit counts.
		c.Threads = runtime.GOMAXPROCS(0)
	}
	c.Parallel = max(1, c.Parallel)
	if c.Name == "" {
		host, _ := os.Hostname()
		c.Name = cmp.Or(sanitizeName(host), "monworker")
	}
	if c.Cache == "" {
		c.Cache = filepath.Dir(c.File)
	}
	return c, nil
}

func sanitizeName(s string) string {
	var b strings.Builder
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			b.WriteRune(r)
		case r == '.':
			return b.String()
		default:
			b.WriteRune('-')
		}
		if b.Len() == 32 {
			break
		}
	}
	return b.String()
}

type Pairing struct {
	URL string `toml:"url"`
	// Token is monbooru's, for the worker to remove its own pairing.
	Token string `toml:"token"`
	// PeerToken is the worker's, which monbooru calls it with.
	PeerToken string `toml:"peer_token"`
}

type fileDoc struct {
	Pairings []Pairing `toml:"pairing"`
}

type Store struct {
	mu   sync.Mutex
	path string
	doc  fileDoc
}

func Load(path string) (*Store, error) {
	s := &Store{path: path}
	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return s, nil
	}
	if err != nil {
		return nil, err
	}
	if err := toml.Unmarshal(data, &s.doc); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return s, nil
}

func (s *Store) Pairings() []Pairing {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.doc.Pairings)
}

func (s *Store) Put(p Pairing) error {
	return s.update(func(d *fileDoc) {
		d.Pairings = slices.DeleteFunc(d.Pairings, func(q Pairing) bool { return q.URL == p.URL })
		d.Pairings = append(d.Pairings, p)
	})
}

func (s *Store) Remove(url string) error {
	return s.update(func(d *fileDoc) {
		d.Pairings = slices.DeleteFunc(d.Pairings, func(q Pairing) bool { return q.URL == url })
	})
}

func (s *Store) RemoveByPeerToken(token string) (bool, error) {
	found := false
	err := s.update(func(d *fileDoc) {
		d.Pairings = slices.DeleteFunc(d.Pairings, func(q Pairing) bool {
			hit := q.PeerToken == token
			found = found || hit
			return hit
		})
	})
	return found, err
}

func (s *Store) update(fn func(*fileDoc)) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	next := fileDoc{Pairings: slices.Clone(s.doc.Pairings)}
	fn(&next)
	var b strings.Builder
	b.WriteString("# monworker's pairings with monbooru, written by monworker.\n\n")
	if err := toml.NewEncoder(&b).Encode(next); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return err
	}
	if err := fsx.WriteFile(s.path, []byte(b.String())); err != nil {
		return err
	}
	s.doc = next
	return nil
}
