package models

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"sync"
	"time"

	"github.com/monbooru/monworker/internal/shared/fsx"
	"github.com/monbooru/monworker/internal/shared/logx"
)

// Digests names a model by its bytes: the weights, the label file and the
// resolved profile. A file is hashed again only when its size or
// modification time moves, and the sums are kept in a file, so a
// multi-GB model is read once.
type Digests struct {
	mu   sync.Mutex
	path string
	sums map[string]fileSum
	busy map[string]bool
}

type fileSum struct {
	Size    int64     `json:"size"`
	ModTime time.Time `json:"mtime"`
	Sum     string    `json:"sha256"`
}

// NewDigests keeps its sums in path; an empty path keeps them in memory.
func NewDigests(path string) *Digests {
	d := &Digests{path: path, sums: map[string]fileSum{}, busy: map[string]bool{}}
	if path != "" {
		if data, err := os.ReadFile(path); err == nil {
			if err := json.Unmarshal(data, &d.sums); err != nil {
				logx.Warnf("models: %s: %v; the weights are read again", path, err)
				d.sums = map[string]fileSum{}
			}
		}
	}
	return d
}

// Of returns f's digest, or "" while its files are hashed in the
// background; call it again to read the result.
func (d *Digests) Of(f Folder) string {
	if f.Err != nil {
		return ""
	}
	files := append(f.Weights(), f.LabelPath())
	d.mu.Lock()
	defer d.mu.Unlock()
	var stale []string
	sums := make([]string, 0, len(files))
	for _, p := range files {
		fi, err := os.Stat(p)
		if err != nil {
			return ""
		}
		s, ok := d.sums[p]
		if !ok || s.Size != fi.Size() || !s.ModTime.Equal(fi.ModTime()) {
			stale = append(stale, p)
			continue
		}
		sums = append(sums, s.Sum)
	}
	if len(stale) > 0 {
		if !d.busy[f.Dir] {
			d.busy[f.Dir] = true
			go d.hash(f.Dir, stale)
		}
		return ""
	}
	h := sha256.New()
	for _, s := range sums {
		h.Write([]byte(s))
	}
	profile, _ := json.Marshal(f.Profile)
	h.Write(profile)
	return hex.EncodeToString(h.Sum(nil))
}

func (d *Digests) hash(dir string, files []string) {
	defer func() {
		d.mu.Lock()
		delete(d.busy, dir)
		d.mu.Unlock()
	}()
	for _, p := range files {
		fi, err := os.Stat(p)
		if err != nil {
			return
		}
		start := time.Now()
		sum, err := hashFile(p)
		if err != nil {
			logx.Warnf("models: hash %s: %v", p, err)
			return
		}
		if fi.Size() > 64<<20 {
			logx.Infof("models: read %s in %s", p, time.Since(start).Round(time.Millisecond))
		}
		d.mu.Lock()
		d.sums[p] = fileSum{Size: fi.Size(), ModTime: fi.ModTime(), Sum: sum}
		d.mu.Unlock()
	}
	d.save()
}

func (d *Digests) save() {
	if d.path == "" {
		return
	}
	d.mu.Lock()
	data, err := json.MarshalIndent(d.sums, "", "  ")
	d.mu.Unlock()
	if err == nil {
		err = fsx.WriteFile(d.path, data)
	}
	if err != nil {
		logx.Warnf("models: %s: %v", d.path, err)
	}
}
