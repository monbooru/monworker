#!/bin/sh
# Downloads ONNX Runtime (and DirectML on Windows) from NuGet for one
# target into tools/; GOOS, GOARCH and the pins.env values come from the
# environment.
set -eu

case "${GOOS}/${GOARCH}" in
  linux/amd64)   rid=linux-x64;   pkg=microsoft.ml.onnxruntime;          ort_sha=$ORT_SHA256 ;;
  linux/arm64)   rid=linux-arm64; pkg=microsoft.ml.onnxruntime;          ort_sha=$ORT_SHA256 ;;
  windows/amd64) rid=win-x64;     pkg=microsoft.ml.onnxruntime.directml; ort_sha=$ORT_DML_SHA256 ;;
  *) echo "no runtime published for ${GOOS}/${GOARCH}" >&2; exit 1 ;;
esac

verify() {
  echo "$2  $1" | sha256sum -c -
}

nuget=https://api.nuget.org/v3-flatcontainer
rm -rf tools /tmp/ort && mkdir -p tools/licenses/onnxruntime
curl -fsSL -o /tmp/ort.zip "$nuget/$pkg/$ORT_VERSION/$pkg.$ORT_VERSION.nupkg"
verify /tmp/ort.zip "$ort_sha"
unzip -q /tmp/ort.zip -d /tmp/ort
ort_dir=/tmp/ort
if [ "$GOOS" = windows ]; then
  cp "$ort_dir/runtimes/$rid/native/onnxruntime.dll" tools/
  # Its telemetry provider and the Microsoft telemetry group it joins get
  # GUIDs no collector listens to; DisableTelemetryEvents alone is undone
  # whenever a trace session enables the provider.
  perl -0777 -pi -e '
    $p = s/\xff\xb1\x26\x3a\x84\x74\x84\x74\x74\x84\x15\x26\x1f\x42\x61\x4d/\x7d\xad\x43\x9e\xcd\x66\xe8\x42\xbe\x3a\xef\xcc\x91\x38\x54\x16/g;
    $g = s/\x1a\x73\x50\x4f\xcf\x89\x82\x47\xb3\xe0\xdc\xe8\xc9\x04\x76\xba/\x88\x4e\x1a\x79\x86\x12\x35\x46\x9e\x92\xac\x9d\x64\x43\x31\xfc/g;
    die "onnxruntime.dll: telemetry GUIDs not found once each\n" unless $p == 1 && $g == 1;
  ' tools/onnxruntime.dll
  verify tools/onnxruntime.dll "$ORT_DML_DLL_SHA256"
  rm -rf /tmp/dml
  curl -fsSL -o /tmp/dml.zip "$nuget/microsoft.ai.directml/${DIRECTML_VERSION}/microsoft.ai.directml.${DIRECTML_VERSION}.nupkg"
  verify /tmp/dml.zip "$DIRECTML_SHA256"
  unzip -q /tmp/dml.zip -d /tmp/dml
  cp /tmp/dml/bin/x64-win/DirectML.dll tools/
  mkdir -p tools/licenses/directml
  cp /tmp/dml/LICENSE.txt /tmp/dml/LICENSE-CODE.txt /tmp/dml/ThirdPartyNotices.txt tools/licenses/directml/
else
  cp "$ort_dir/runtimes/$rid/native/libonnxruntime.so" tools/
fi
cp "$ort_dir/LICENSE" "$ort_dir/ThirdPartyNotices.txt" tools/licenses/onnxruntime/
ls -la tools
