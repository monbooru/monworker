FROM golang:1.27.4-trixie AS build
ARG VERSION=dev
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=1 go build -trimpath -ldflags="-s -w -X main.version=${VERSION}" -o /monworker ./cmd/monworker

FROM debian:trixie-slim AS assets
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl unzip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=gcr.io/distroless/cc-debian13:nonroot /etc/passwd /etc/group /idsrc/
RUN echo 'monworker:x:1000:1000:monworker:/home/monworker:/sbin/nologin' >> /idsrc/passwd && \
    echo 'monworker:x:1000:' >> /idsrc/group && \
    install -d -o 1000 -g 1000 /idsrc/home

ARG ORT_VERSION
ARG ORT_SHA256
RUN case "$(dpkg --print-architecture)" in \
      amd64) RID=linux-x64 ;; \
      arm64) RID=linux-arm64 ;; \
      *) echo "Unsupported arch" && exit 1 ;; \
    esac && \
    curl -fsSL "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/${ORT_VERSION}/microsoft.ml.onnxruntime.${ORT_VERSION}.nupkg" \
      -o /tmp/ort.zip && \
    echo "${ORT_SHA256}  /tmp/ort.zip" | sha256sum -c - && \
    unzip -q /tmp/ort.zip -d /tmp/ort && \
    mkdir -p /opt/ort /opt/licenses/onnxruntime && \
    cp /tmp/ort/runtimes/${RID}/native/libonnxruntime.so /opt/ort/libonnxruntime.so && \
    cp /tmp/ort/LICENSE /tmp/ort/ThirdPartyNotices.txt /opt/licenses/onnxruntime/

FROM gcr.io/distroless/cc-debian13:nonroot

COPY --from=assets /idsrc/passwd /etc/passwd
COPY --from=assets /idsrc/group  /etc/group
COPY --from=assets --chown=1000:1000 /idsrc/home /home/monworker
COPY --from=assets /opt/ort/libonnxruntime.so /usr/lib/libonnxruntime.so
COPY --from=assets /opt/licenses              /usr/share/doc/
COPY --from=build  /monworker                 /usr/local/bin/monworker

ENV MONWORKER_LISTEN=0.0.0.0:8457 \
    MONWORKER_MODELS=/models \
    MONWORKER_CONFIG=/config/monworker.toml \
    HOME=/home/monworker

EXPOSE 8457

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["/usr/local/bin/monworker", "healthcheck"]

USER 1000:1000

ENTRYPOINT ["/usr/local/bin/monworker"]
