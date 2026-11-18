package models

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/monbooru/monworker/internal/shared/worker"
)

func writeLabelFile(t *testing.T, name, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	mustWriteFile(t, p, []byte(body), 0o644)
	return p
}

func TestReadLabelsKeepsTheNamesAsTheFileHasThem(t *testing.T) {
	csv := writeLabelFile(t, "tags.csv", "tag_id,name,category\n1,1girl,0\n2,hatsune miku,4\n3,bkub,style\n4,solo\n")
	got, err := ReadLabels(csv, Profile{LabelFormat: "csv"})
	if err != nil {
		t.Fatal(err)
	}
	want := []worker.Label{{Name: "1girl"}, {Name: "hatsune miku", CategoryID: 4}, {Name: "bkub", Category: "style"}, {Name: "solo"}}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("csv = %+v, want %+v", got, want)
	}
	txt := writeLabelFile(t, "tags.txt", "solo\n\nblue sky\n")
	if got, err := ReadLabels(txt, Profile{LabelFormat: "txt"}); err != nil || fmt.Sprint(got) != fmt.Sprint([]worker.Label{{Name: "solo"}, {Name: "blue sky"}}) {
		t.Errorf("txt = %+v, %v", got, err)
	}
}

func TestReadLabelsLeavesAHoleNamelessAndCountsIt(t *testing.T) {
	p := writeLabelFile(t, "tags.json", `{"m": {"0": "a", "1": "b", "3": "d"}, "c": {"a": "general", "d": "character"}}`)
	got, err := ReadLabels(p, Profile{LabelFormat: "json", LabelJSON: &LabelJSON{Shape: "index_map", Path: "m", CategoryPath: "c"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 4 || got[2].Name != "" || got[3].Category != "character" {
		t.Errorf("labels = %+v", got)
	}
}

func TestCatalogOffersTheEmbeddedTaggersWithWhatEachMisses(t *testing.T) {
	dir := t.TempDir()
	mustMkdirAll(t, filepath.Join(dir, "wd-swinv2"), 0o755)
	mustWriteFile(t, filepath.Join(dir, "wd-swinv2", "model.onnx"), []byte("x"), 0o644)
	entries := Catalog(dir)
	byName := map[string]worker.CatalogEntry{}
	for _, e := range entries {
		byName[e.Name] = e
	}
	wd, ok := byName["wd-swinv2"]
	if !ok || len(byName) != 4 {
		t.Fatalf("catalog = %v", entries)
	}
	if strings.Contains(wd.Command, "model.onnx'") || !strings.Contains(wd.Command, "'wd-swinv2/tags.csv'") {
		t.Errorf("command fetches what is there or misses what is not:\n%s", wd.Command)
	}
	if !strings.HasPrefix(byName["joytag"].Command, "mkdir -p 'joytag' && ") {
		t.Errorf("joytag command = %q", byName["joytag"].Command)
	}
}

func installModel(t *testing.T, dir, name, labels, sidecar string) Folder {
	t.Helper()
	d := filepath.Join(dir, name)
	mustMkdirAll(t, d, 0o755)
	mustWriteFile(t, filepath.Join(d, "model.onnx"), []byte("weights"), 0o644)
	mustWriteFile(t, filepath.Join(d, "tags.csv"), []byte(labels), 0o644)
	if sidecar != "" {
		mustWriteFile(t, filepath.Join(d, "tagger.json"), []byte(sidecar), 0o644)
	}
	return Open(dir, name)
}

func TestDescribeReadsTheSizeTheProfileGives(t *testing.T) {
	dir := t.TempDir()
	f := installModel(t, dir, "small", "id,name,category\n0,a,0\n1,b,4\n", `{"version": 1, "profile": {"input_size": 64}}`)
	m := f.Describe(dir, "d1")
	if !m.Ready || m.Input.Size != 64 || m.Input.Fill != [3]uint8{255, 255, 255} || m.Labels != 2 || m.Digest != "d1" || m.LabelsETag == "" {
		t.Errorf("model = %+v", m)
	}
	if m := f.Describe(dir, ""); m.Ready || m.Reason == "" {
		t.Errorf("without a digest yet: %+v", m)
	}
	broken := installModel(t, dir, "broken", "id,name,category\n0,a,0\n", "")
	if m := broken.Describe(dir, "d2"); m.Ready || !strings.Contains(m.Reason, "model.onnx") {
		t.Errorf("an unreadable model: %+v", m)
	}
}

func TestScanSkipsFoldersWithoutModelFilesAndFlagsBadNames(t *testing.T) {
	dir := t.TempDir()
	installModel(t, dir, "good", "id,name,category\n0,a,0\n", "")
	mustMkdirAll(t, filepath.Join(dir, "empty"), 0o755)
	mustMkdirAll(t, filepath.Join(dir, "bad name"), 0o755)
	mustWriteFile(t, filepath.Join(dir, "bad name", "model.onnx"), nil, 0o644)
	folders := Scan(dir)
	if len(folders) != 2 || folders[0].Name != "bad name" || folders[0].Err == nil || folders[1].Name != "good" || folders[1].Err != nil {
		t.Errorf("scan = %+v", folders)
	}
}

func TestWeightsTakeTheExternalDataBesideTheModel(t *testing.T) {
	dir := t.TempDir()
	f := installModel(t, dir, "big", "id,name,category\n0,a,0\n", "")
	mustWriteFile(t, filepath.Join(f.Dir, "model.onnx.data"), []byte("more"), 0o644)
	if w := f.Weights(); len(w) != 2 || filepath.Base(w[1]) != "model.onnx.data" {
		t.Errorf("weights = %v", w)
	}
}

func digestOf(t *testing.T, d *Digests, f Folder) string {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if s := d.Of(f); s != "" {
			return s
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("the digest never came")
	return ""
}

func TestDigestFollowsTheBytesAndSurvivesARestart(t *testing.T) {
	dir := t.TempDir()
	f := installModel(t, dir, "m", "id,name,category\n0,a,0\n", "")
	cache := filepath.Join(t.TempDir(), "digests.json")
	first := digestOf(t, NewDigests(cache), f)
	if _, err := os.Stat(cache); err != nil {
		t.Fatalf("the sums were not kept: %v", err)
	}
	again := NewDigests(cache)
	if got := again.Of(f); got != first {
		t.Errorf("after a restart the digest is %q, want %q at once", got, first)
	}
	later := time.Now().Add(time.Minute)
	mustWriteFile(t, f.ModelPath(), []byte("other weights"), 0o644)
	if err := os.Chtimes(f.ModelPath(), later, later); err != nil {
		t.Fatal(err)
	}
	if got := digestOf(t, again, f); got == first {
		t.Error("new weights kept the old digest")
	}
}
