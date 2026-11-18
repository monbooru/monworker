package models

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/monbooru/monworker/internal/shared/worker"
)

const (
	DefaultModelFile    = "model.onnx"
	DefaultTagsFile     = "tags.csv"
	DefaultTextTagsFile = "tags.txt"
)

type Folder struct {
	Name      string
	Dir       string
	ModelFile string
	LabelFile string
	Profile   Profile
	Err       error
}

func (f Folder) ModelPath() string { return filepath.Join(f.Dir, f.ModelFile) }
func (f Folder) LabelPath() string { return filepath.Join(f.Dir, f.LabelFile) }

func Scan(dir string) []Folder {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var out []Folder
	for _, e := range entries {
		name := e.Name()
		// Stat, not the entry's type: a symlinked folder reports as a link.
		if fi, err := os.Stat(filepath.Join(dir, name)); err != nil || !fi.IsDir() {
			continue
		}
		if onnx, labels := listFiles(filepath.Join(dir, name)); len(onnx)+len(labels) == 0 {
			continue
		}
		out = append(out, Open(dir, name))
	}
	return out
}

func Open(dir, name string) Folder {
	f := Folder{Name: name, Dir: filepath.Join(dir, name)}
	if err := ValidateName(name); err != nil {
		f.Err = err
		return f
	}
	f.ModelFile, f.LabelFile = resolveFiles(f.Dir)
	switch {
	case !regularFile(f.ModelPath()):
		f.Err = fmt.Errorf("missing %s", f.ModelFile)
	case !regularFile(f.LabelPath()):
		f.Err = fmt.Errorf("missing %s", f.LabelFile)
	default:
		p, err := ResolveProfile(dir, name, f.LabelFile)
		if err != nil {
			f.Err = fmt.Errorf("no inference profile: %w; add a tagger.json", err)
		}
		f.Profile = p
	}
	return f
}

// resolveFiles takes the default names, else a lone file of each kind,
// and falls back to the defaults so a missing file is named.
func resolveFiles(dir string) (string, string) {
	onnx, labels := listFiles(dir)
	modelFile := DefaultModelFile
	if !slices.Contains(onnx, DefaultModelFile) && len(onnx) == 1 {
		modelFile = onnx[0]
	}
	tagsFile := DefaultTagsFile
	switch {
	case slices.Contains(labels, DefaultTagsFile):
	case slices.Contains(labels, DefaultTextTagsFile):
		tagsFile = DefaultTextTagsFile
	case len(labels) == 1:
		tagsFile = labels[0]
	}
	return modelFile, tagsFile
}

func isSidecar(name string) bool {
	switch name {
	case "tagger.json", "dispatch.json", modelManifest, stockProfile, stockDispatch:
		return true
	}
	return false
}

func listFiles(dir string) (onnx, labels []string) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, nil
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		n := e.Name()
		switch strings.ToLower(filepath.Ext(n)) {
		case ".onnx":
			onnx = append(onnx, n)
		case ".csv", ".txt":
			labels = append(labels, n)
		case ".json":
			if !isSidecar(n) {
				labels = append(labels, n)
			}
		}
	}
	return onnx, labels
}

// InputSide is the frame side the folder's model takes: the profile's, or
// the ONNX file's when the profile leaves it at 0.
func (f Folder) InputSide() (int, error) {
	if f.Profile.InputSize > 0 {
		return f.Profile.InputSize, nil
	}
	fi, err := os.Stat(f.ModelPath())
	if err != nil {
		return 0, err
	}
	key := sideKey{f.ModelPath(), fi.Size(), fi.ModTime(), f.Profile.Layout}
	if size, ok := sides.Load(key); ok {
		return size.(int), nil
	}
	dims, err := InputDims(f.ModelPath())
	if err != nil {
		return 0, fmt.Errorf("inspect %s: %w", f.ModelFile, err)
	}
	size := InputSize(dims, f.Profile.Layout)
	if size <= 0 {
		return 0, fmt.Errorf("cannot infer the input size from dimensions %v", dims)
	}
	sides.Store(key, size)
	return size, nil
}

var sides sync.Map

type sideKey struct {
	path    string
	size    int64
	modTime time.Time
	layout  string
}

func (f Folder) Labels(modelsDir string) (worker.Labels, error) {
	if f.Err != nil {
		return worker.Labels{}, f.Err
	}
	labels, err := ReadLabels(f.LabelPath(), f.Profile)
	if err != nil {
		return worker.Labels{}, fmt.Errorf("read %s: %w", f.LabelFile, err)
	}
	out := worker.Labels{
		Model:          f.Name,
		CategoryScheme: f.Profile.CategoryScheme,
		CategoryMap:    f.Profile.CategoryMap,
		Labels:         labels,
		Mappings:       StockMappings(modelsDir, f.Name),
	}
	out.ETag = etagOf(out)
	return out, nil
}

func etagOf(l worker.Labels) string {
	h := sha256.New()
	_ = json.NewEncoder(h).Encode(l)
	return `"` + hex.EncodeToString(h.Sum(nil)[:12]) + `"`
}

func (f Folder) Describe(modelsDir, digest string) worker.Model {
	m := worker.Model{Name: f.Name, Kind: worker.KindTagger, ModelFile: f.ModelFile, LabelFile: f.LabelFile}
	if f.Err != nil {
		m.Reason = f.Err.Error()
		return m
	}
	size, err := f.InputSide()
	if err != nil {
		m.Reason = err.Error()
		return m
	}
	labels, err := f.Labels(modelsDir)
	if err != nil {
		m.Reason = err.Error()
		return m
	}
	m.Input = worker.Input{Size: size, Fill: Fill(f.Profile)}
	m.Labels = len(labels.Labels)
	m.LabelsETag = labels.ETag
	m.Profile, _ = json.Marshal(f.Profile)
	m.Sidecar = SidecarProfileFields(modelsDir, f.Name)
	m.Digest = digest
	if digest == "" {
		m.Reason = "reading the weights"
		return m
	}
	m.Ready = true
	return m
}

// Weights are the files whose bytes make the model: the ONNX file and the
// external data files beside it, named after it.
func (f Folder) Weights() []string {
	out := []string{f.ModelPath()}
	entries, _ := os.ReadDir(f.Dir)
	for _, e := range entries {
		if n := e.Name(); !e.IsDir() && n != f.ModelFile && strings.HasPrefix(n, f.ModelFile) {
			out = append(out, filepath.Join(f.Dir, n))
		}
	}
	return out
}

func hashFile(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer func() { _ = file.Close() }()
	h := sha256.New()
	if _, err := io.Copy(h, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
