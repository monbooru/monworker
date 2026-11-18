package models

import (
	"encoding/json"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"

	"github.com/monbooru/monworker/internal/registry"
	"github.com/monbooru/monworker/internal/shared/logx"
	"github.com/monbooru/monworker/internal/shared/worker"
)

const (
	catalogSchemaVersion  = 1
	dispatchSchemaVersion = 1

	modelManifest = "model.json"
	stockProfile  = "profile_default.json"
	stockDispatch = "dispatch_default.json"
)

type modelDoc struct {
	Version int `json:"version"`
	worker.CatalogEntry
}

func Builtin(name string) bool { return slices.Contains(registry.Taggers, name) }

// stockFile reads one of a tagger's stock files: the embedded copy for a
// tagger the catalog ships, whatever its folder holds, else the folder's.
func stockFile(modelsDir, name, file string) ([]byte, string, error) {
	if Builtin(name) {
		data, err := registry.FS.ReadFile(path.Join(registry.TaggersDir, name, file))
		return data, "embedded " + name, err
	}
	p := filepath.Join(modelsDir, name, file)
	data, err := os.ReadFile(p)
	return data, p, err
}

func loadModelDoc(modelsDir, name string) (worker.CatalogEntry, bool) {
	data, source, err := stockFile(modelsDir, name, modelManifest)
	if err != nil {
		return worker.CatalogEntry{}, false
	}
	var doc modelDoc
	if err := json.Unmarshal(data, &doc); err != nil {
		logx.Warnf("models: %s: parse failed: %v", source, err)
		return worker.CatalogEntry{}, false
	}
	if doc.Version != catalogSchemaVersion {
		logx.Warnf("models: %s: schema version %d unsupported (want %d)", source, doc.Version, catalogSchemaVersion)
		return worker.CatalogEntry{}, false
	}
	doc.Name = name
	return doc.CatalogEntry, true
}

// Catalog lists the models the worker knows how to fetch: the embedded
// ones, then any folder carrying a model.json. Each entry's Command
// fetches what its folder still misses.
func Catalog(modelsDir string) []worker.CatalogEntry {
	var out []worker.CatalogEntry
	add := func(e worker.CatalogEntry) {
		e.Command = missing(e, filepath.Join(modelsDir, e.Name)).hostCommand()
		out = append(out, e)
	}
	for _, name := range registry.Taggers {
		if e, ok := loadModelDoc(modelsDir, name); ok {
			add(e)
		}
	}
	if dirs, err := os.ReadDir(modelsDir); err == nil {
		for _, d := range dirs {
			name := d.Name()
			if Builtin(name) || ValidateName(name) != nil {
				continue
			}
			if e, ok := loadModelDoc(modelsDir, name); ok {
				add(e)
			}
		}
	}
	return out
}

type catalogEntry worker.CatalogEntry

func missing(c worker.CatalogEntry, dir string) catalogEntry {
	out := catalogEntry(c)
	out.Files = nil
	for _, f := range c.Files {
		if !regularFile(filepath.Join(dir, f.Filename)) && !regularFile(filepath.Join(dir, downloadName(f.URL))) {
			out.Files = append(out.Files, f)
		}
	}
	return out
}

func downloadName(raw string) string {
	u, err := url.Parse(raw)
	if err != nil {
		return ""
	}
	return path.Base(u.Path)
}

func regularFile(p string) bool {
	info, err := os.Stat(p)
	return err == nil && info.Mode().IsRegular()
}

// hostCommand's paths are relative to the models directory. $HF_TOKEN sits
// in double quotes so the shell running the snippet expands it.
func (c catalogEntry) hostCommand() string {
	if len(c.Files) == 0 {
		return ""
	}
	auth := ""
	if c.Gated {
		auth = `-H "Authorization: Bearer $HF_TOKEN" `
	}
	steps := []string{"mkdir -p " + shellSingleQuote(c.Name)}
	for _, f := range c.Files {
		steps = append(steps, "curl -L "+auth+"-o "+shellSingleQuote(c.Name+"/"+f.Filename)+" "+shellSingleQuote(f.URL))
	}
	return strings.Join(steps, " && \\\n")
}

func shellSingleQuote(s string) string { return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'" }

type dispatchDoc struct {
	Version int              `json:"version"`
	Rules   []worker.Mapping `json:"rules"`
}

func StockMappings(modelsDir, name string) []worker.Mapping {
	data, source, err := stockFile(modelsDir, name, stockDispatch)
	if err != nil {
		return nil
	}
	var doc dispatchDoc
	if err := json.Unmarshal(data, &doc); err != nil {
		logx.Warnf("models: %s dispatch: parse failed: %v", source, err)
		return nil
	}
	if doc.Version != dispatchSchemaVersion {
		logx.Warnf("models: %s dispatch: schema version %d unsupported (want %d)", source, doc.Version, dispatchSchemaVersion)
		return nil
	}
	return doc.Rules
}
