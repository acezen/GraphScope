package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"net/http"

	"github.com/golang/glog"
	"github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
	"google.golang.org/grpc"

	gw "github.com/alibaba/GraphScope/coordinator/grpc_gateway/gen/go/proto"
)

func run(grpcServerEndpoint string, gatewayAddress string) error {
	ctx := context.Background()
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Register gRPC server endpoint
	// Note: Make sure the gRPC server is running properly and accessible
	mux := runtime.NewServeMux()
	opts := []grpc.DialOption{
		grpc.WithInsecure(),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(math.MaxInt32),
			grpc.MaxCallSendMsgSize(math.MaxInt32),
		),
	}
	err := gw.RegisterCoordinatorServiceHandlerFromEndpoint(ctx, mux, grpcServerEndpoint, opts)
	if err != nil {
		return err
	}

	fmt.Printf("graphscope grpc-gateway: proxy coordinator at %v to %v ...\n", grpcServerEndpoint, gatewayAddress)
	// Start HTTP server (and proxy calls to gRPC server endpoint)
	return http.ListenAndServe(gatewayAddress, mux)
}

func main() {
	var gatewayAddress string
	var grpcServerEndpoint string
	flag.StringVar(&grpcServerEndpoint, "grpc-server-endpoint", "localhost:63800", "gRPC server endpoint")
	flag.StringVar(&gatewayAddress, "gateway-address", ":8081", "The address that the service gateway binds to")
	flag.Parse()
	defer glog.Flush()

	if err := run(grpcServerEndpoint, gatewayAddress); err != nil {
		glog.Fatal(err)
	}
}
