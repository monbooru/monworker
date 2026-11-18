package registry

import "embed"

//go:embed plugins/taggers/wd-swinv2 plugins/taggers/animetimm-eva02 plugins/taggers/joytag plugins/taggers/pixai-v1
var FS embed.FS

const TaggersDir = "plugins/taggers"

var Taggers = []string{"wd-swinv2", "animetimm-eva02", "joytag", "pixai-v1"}
