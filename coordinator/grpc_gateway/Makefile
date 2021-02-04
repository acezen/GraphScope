ifeq (,$(shell go env GOBIN))
GOBIN=$(shell go env GOPATH)/bin
else
GOBIN=$(shell go env GOBIN)
endif

PROTOBUFS := $(wildcard ../../proto/*.proto)

all: gateway

gen:
	mkdir -p gen/go
	PATH=$PATH:$(GOBIN)
	protoc -I ../../ \
		--go_out ./gen/go \
		--go_opt paths=source_relative \
		--go-grpc_out ./gen/go \
		--go-grpc_opt paths=source_relative \
		$(PROTOBUFS)
	protoc -I ../../ \
		--grpc-gateway_out ./gen/go \
		--grpc-gateway_opt logtostderr=true \
		--grpc-gateway_opt paths=source_relative \
		--grpc-gateway_opt generate_unbound_methods=true \
		$(PROTOBUFS)
.PHONY: gen

openapi:
	mkdir -p gen/openapiv2
	PATH=$PATH:$(GOBIN)
	protoc -I ../.. \
		--openapiv2_out ./gen/openapiv2 \
		--openapiv2_opt logtostderr=true \
		--openapiv2_opt generate_unbound_methods=true \
		$(PROTOBUFS)
.PHONY: openapi

install-tools:
	go install \
		github.com/grpc-ecosystem/grpc-gateway/v2/protoc-gen-grpc-gateway \
		github.com/grpc-ecosystem/grpc-gateway/v2/protoc-gen-openapiv2 \
		google.golang.org/protobuf/cmd/protoc-gen-go \
		google.golang.org/grpc/cmd/protoc-gen-go-grpc
.PHONY: install-tools

gateway: gen main.go
	go fmt ./main.go
	go build -o bin/gateway main.go
