package infer

import (
	"os"
	"path/filepath"
	"runtime"

	"github.com/monbooru/monworker/internal/shared/desktop"
)

// sharedLibPath tries the executable's directory first so a bundled
// library wins over the system one.
func sharedLibPath() string {
	if p := os.Getenv("ORT_LIB_PATH"); p != "" {
		return p
	}
	name := "libonnxruntime.so"
	var candidates []string
	if runtime.GOOS == "windows" {
		name = "onnxruntime.dll"
	}
	if dir := desktop.ExeDir(); dir != "" {
		candidates = append(candidates, filepath.Join(dir, name))
	}
	if runtime.GOOS == "windows" {
		// The working directory is a Windows loader convention.
		if wd, err := os.Getwd(); err == nil {
			candidates = append(candidates, filepath.Join(wd, name))
		}
	} else {
		candidates = append(candidates, "/usr/lib/"+name, "/usr/local/lib/"+name)
	}
	candidates = append(candidates, name)
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return name
}

// MissingRuntimeLibrary says what is wrong with the library sharedLibPath
// found, "" when it is there. Only an ORT_LIB_PATH that is not there is
// sure to fail: a bare name still goes to the system's loader.
func MissingRuntimeLibrary() string {
	path := sharedLibPath()
	if _, err := os.Stat(path); err == nil {
		return ""
	}
	name := filepath.Base(path)
	if path != name {
		return "missing " + name + " at " + path + "; no model will run"
	}
	beside := "beside the executable"
	if dir := desktop.ExeDir(); dir != "" {
		beside += " (" + dir + ")"
	}
	if runtime.GOOS == "windows" {
		return name + " is neither " + beside + " nor in the working directory; the DLL search order is tried at the first call"
	}
	return name + " is neither " + beside + " nor in /usr/lib or /usr/local/lib; the system's library path is tried at the first call"
}
