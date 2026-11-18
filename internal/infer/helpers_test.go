package infer

import "github.com/monbooru/monworker/internal/models"

var (
	wd14Profile, _   = models.ResolveProfile("", "wd-swinv2", "tags.csv")
	joytagProfile, _ = models.ResolveProfile("", "joytag", "tags.txt")
)
