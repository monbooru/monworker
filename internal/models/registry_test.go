package models

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/monbooru/monworker/internal/registry"
)

func TestReadLabelsCamieJSON_Fixture(t *testing.T) {
	got, err := ReadLabels("testdata/camie_v2_min.json", Profile{LabelFormat: "json", LabelJSON: &camieLabelJSON})
	if err != nil {
		t.Fatal(err)
	}
	want := []struct{ name, cat string }{
		{"1girl", "general"}, {"hatsune_miku", "character"}, {"wlop", "artist"},
		{"neon_genesis_evangelion", "copyright"}, {"explicit", "rating"},
	}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i, w := range want {
		if got[i].Name != w.name || got[i].Category != w.cat {
			t.Errorf("label %d = %+v, want %s in %s", i, got[i], w.name, w.cat)
		}
	}
}

func TestResolveProfile_CamieRegistryFolder(t *testing.T) {
	tmp := t.TempDir()
	installRegistryTagger(t, tmp, "camie-v2")
	got := mustResolveProfile(t, tmp, "camie-v2", "camie-tagger-v2-metadata.json")
	want := Profile{
		Name: "camie-v2", Layout: "nchw", Channels: "rgb", Normalize: "imagenet", Pad: "mean_color_aspect",
		Activation: "logits", LabelFormat: "json", LabelJSON: &camieLabelJSON, CategoryScheme: "name_string", OutputIndex: 1,
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("camie-v2 profile drift:\n got %+v\nwant %+v", got, want)
	}
}

func registryTaggers(t *testing.T) []string {
	t.Helper()
	entries, err := os.ReadDir(filepath.Join("..", "registry", registry.TaggersDir))
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, e := range entries {
		if e.IsDir() {
			out = append(out, e.Name())
		}
	}
	return out
}

func TestEveryRegistryTaggerLoads(t *testing.T) {
	tmp := t.TempDir()
	for _, name := range registryTaggers(t) {
		installRegistryTagger(t, tmp, name)
	}
	byName := map[string]bool{}
	for _, e := range Catalog(tmp) {
		byName[e.Name] = true
		if e.Description == "" || len(e.Files) == 0 {
			t.Errorf("%s: entry lacks a description or files: %+v", e.Name, e)
		}
		label := ""
		for _, f := range e.Files {
			switch strings.ToLower(filepath.Ext(f.Filename)) {
			case ".csv", ".txt":
				label = f.Filename
			case ".json":
				if !strings.HasSuffix(f.Filename, "_default.json") {
					label = f.Filename
				}
			}
		}
		if label == "" {
			t.Errorf("%s: no label file among its files", e.Name)
		} else if _, err := ResolveProfile(tmp, e.Name, label); err != nil {
			t.Errorf("%s: %v", e.Name, err)
		}
		if _, err := os.Stat(filepath.Join(tmp, e.Name, stockDispatch)); err == nil && len(StockMappings(tmp, e.Name)) == 0 {
			t.Errorf("%s: dispatch_default.json holds no rule the worker reads", e.Name)
		}
	}
	for _, name := range registryTaggers(t) {
		if !byName[name] {
			t.Errorf("%s: no catalog entry from its model.json", name)
		}
	}
}

func TestTheCatalogCarriesItsDefaults(t *testing.T) {
	byName := map[string]float64{}
	chars := map[string]float64{}
	for _, e := range Catalog(t.TempDir()) {
		byName[e.Name] = e.DefaultThreshold
		chars[e.Name] = e.DefaultThresholds["character"]
	}
	if byName["wd-swinv2"] != 0.35 || chars["wd-swinv2"] != 0.50 || byName["joytag"] != 0.40 || byName["animetimm-eva02"] != 0.40 || chars["animetimm-eva02"] != 0.61 {
		t.Errorf("defaults = %v, character %v", byName, chars)
	}
}

func TestStockMappingsKeepTheShippedRules(t *testing.T) {
	for _, c := range []struct{ tagger, source, cat, name string }{
		{"joytag", "hatsune_miku", "character", ""},
		{"joytag", "photo_(medium)", "medium", "photo"},
		{"wd-swinv2", "dakimakura_(medium)", "medium", "dakimakura"},
		{"wd-swinv2", "monochrome", "medium", ""},
		{"wd-swinv2", "comic", "medium", ""},
	} {
		found := false
		for _, m := range StockMappings("", c.tagger) {
			if m.Source == c.source {
				found = m.Category == c.cat && m.Name == c.name
			}
		}
		if !found {
			t.Errorf("%s/%s: no rule to %s %q", c.tagger, c.source, c.cat, c.name)
		}
	}
	tmp := t.TempDir()
	installRegistryTagger(t, tmp, "camie-v2")
	rules := map[string]string{}
	for _, m := range StockMappings(tmp, "camie-v2") {
		rules[m.Source] = m.Category + "/" + m.Name
	}
	for src, want := range map[string]string{"rating_general": "rating/general", "rating_explicit": "rating/explicit", "year_2018": "year/2018", "greyscale": "medium/"} {
		if rules[src] != want {
			t.Errorf("camie-v2 %s = %q, want %q", src, rules[src], want)
		}
	}
}
