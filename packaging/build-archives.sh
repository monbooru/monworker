#!/bin/sh
# Builds the archive for GOOS/GOARCH into dist/, with the runtime that
# fetch-tools.sh left in tools/ and a .sha256 beside it.
set -eu

version=$(tr -d '[:space:]' < VERSION.md)
os=${GOOS:?GOOS must name the target}
arch=${GOARCH:?GOARCH must name the target}
exe=monworker
[ "$os" = windows ] && exe=monworker.exe

name="monworker_${version#v}_${os}_${arch}"
dir="stage/$name"
rm -rf "$dir" && mkdir -p "$dir" dist
CGO_ENABLED=1 go build -trimpath -ldflags="-s -w -X main.version=$version" -o "$dir/$exe" ./cmd/monworker
cp -r tools/* "$dir/"
cp LICENSE README.md "$dir/"
mkdir -p "$dir/models"

if [ "$os" = windows ]; then
  (cd stage && zip -qr "../dist/$name.zip" "$name")
  archive="dist/$name.zip"
else
  tar -czf "dist/$name.tar.gz" -C stage "$name"
  archive="dist/$name.tar.gz"
fi
(cd dist && sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256")
ls -la dist
