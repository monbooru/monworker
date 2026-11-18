package models

import (
	"fmt"
	"regexp"
)

var nameRe = regexp.MustCompile(`^[A-Za-z0-9_-]+$`)

// ValidateName keeps a folder name from reaching outside the models
// directory: it is a path segment on the wire and on disk.
func ValidateName(name string) error {
	if !nameRe.MatchString(name) {
		return fmt.Errorf("model name %q must match [A-Za-z0-9_-]+", name)
	}
	return nil
}
