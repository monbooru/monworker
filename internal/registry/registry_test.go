package registry_test

import (
	"io/fs"
	"path"
	"slices"
	"testing"

	"github.com/monbooru/monworker/internal/registry"
)

func TestBuiltinsAreEmbedded(t *testing.T) {
	for _, name := range registry.Taggers {
		for _, file := range []string{"model.json", "profile_default.json", "dispatch_default.json"} {
			if _, err := fs.ReadFile(registry.FS, path.Join(registry.TaggersDir, name, file)); err != nil {
				t.Errorf("%s/%s: %v", name, file, err)
			}
		}
	}
	entries, err := fs.ReadDir(registry.FS, registry.TaggersDir)
	if err != nil {
		t.Fatal(err)
	}
	var embedded []string
	for _, e := range entries {
		embedded = append(embedded, e.Name())
	}
	if want := slices.Sorted(slices.Values(registry.Taggers)); !slices.Equal(embedded, want) {
		t.Errorf("embedded taggers %v, registry.Taggers %v", embedded, want)
	}
}
