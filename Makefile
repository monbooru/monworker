.PHONY: build lint test test-race test-models

LDFLAGS := -ldflags="-X main.version=$(shell cat VERSION.md)"

build:
	CGO_ENABLED=1 go build $(LDFLAGS) ./cmd/monworker

lint:
	golangci-lint run

test:
	go test -timeout 30m ./...

test-race:
	go test -race -timeout 30m ./...

MODELS_ENV = $(if $(MODELS),MONWORKER_TEST_MODELS=$(MODELS)) $(if $(MODELS_EVAL),MONWORKER_TEST_EVAL=$(MODELS_EVAL))

test-models:
	$(MODELS_ENV) go test -timeout 30m ./...
