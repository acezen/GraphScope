#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import functools
import json
import logging
import sys
import threading
import time

import grpc
import requests
from google.protobuf import json_format

from graphscope.config import GSConfig as gs_config
from graphscope.framework.errors import ConnectionError
from graphscope.framework.errors import GRPCError
from graphscope.framework.errors import check_grpc_response
from graphscope.proto import coordinator_service_pb2_grpc
from graphscope.proto import error_codes_pb2
from graphscope.proto import message_pb2

logger = logging.getLogger("graphscope")


def catch_grpc_error(fn):
    """Print error info from a :class:`grpc.RpcError`."""

    @functools.wraps(fn)
    def with_grpc_catch(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except grpc.RpcError as exc:
            if isinstance(exc, grpc.Call):
                # pylint: disable=no-member
                raise GRPCError(
                    "rpc %s: failed with error code %s, details: %s"
                    % (fn.__name__, exc.code(), exc.details())
                ) from exc
            else:
                raise GRPCError(
                    "rpc %s failed: status %s" % (str(fn.__name__), exc)
                ) from exc

    return with_grpc_catch


def suppress_grpc_error(fn):
    """Suppress the GRPC error."""

    @functools.wraps(fn)
    def with_grpc_catch(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except grpc.RpcError as exc:
            if isinstance(exc, grpc.Call):
                logger.warning(
                    "Grpc call '%s' failed: %s: %s",
                    fn.__name__,
                    exc.code(),
                    exc.details(),
                )
        except Exception as exc:  # noqa: F841
            logger.warning("RPC call '%s' failed: %s", fn.__name__, exc)

    return with_grpc_catch


def catch_grpc_gateway_error(fn):
    """Print error info from a :class:`requests.exceptions.RequestException`."""

    @functools.wraps(fn)
    def with_grpc_gateway_catch(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except requests.exceptions.RequestException as exc:
            if isinstance(exc, grpc.Call):
                # pylint: disable=no-member
                raise GRPCError(
                    "requests call %s: failed with error %s" % (fn.__name__, exc)
                ) from exc
            else:
                raise GRPCError(
                    "requests call %s failed: status %s" % (fn.__name__, exc)
                ) from exc

    return with_grpc_gateway_catch


def suppress_grpc_gateway_error(fn):
    """Suppress the GRPC error."""

    @functools.wraps(fn)
    def with_grpc_gateway_catch(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except requests.exceptions.RequestException as exc:
            if isinstance(exc, requests.exceptions.HTTPError):
                logger.warning(
                    "requests call '%s' failed: %s",
                    fn.__name__,
                    exc,
                )
        except Exception as exc:  # noqa: F841
            logger.warning("requests call '%s' failed: %s", fn.__name__, exc)

    return with_grpc_gateway_catch


class GRPCClient(object):
    def __init__(self, endpoint):
        """Connect to GRAPE engine at the given :code:`endpoint`."""
        # create the gRPC stub
        options = [
            ("grpc.max_send_message_length", 2147483647),
            ("grpc.max_receive_message_length", 2147483647),
        ]
        self._channel = grpc.insecure_channel(endpoint, options=options)
        self._stub = coordinator_service_pb2_grpc.CoordinatorServiceStub(self._channel)
        self._session_id = None
        self._logs_fetching_thread = None

    def waiting_service_ready(self, timeout_seconds=60, enable_k8s=True):
        begin_time = time.time()
        # Do not drop this line, which is for handling KeyboardInterrupt.
        response = None
        while True:
            try:
                response = self.send_heartbeat()
            except Exception as e:
                print(e)
                # grpc disconnect is expect
                response = None
            finally:
                if response is not None:
                    # connnect to coordinator, fetch log
                    if enable_k8s:
                        self.fetch_logs()
                    if response.status.code == error_codes_pb2.OK:
                        logger.info("GraphScope coordinator service connected.")
                        break
                time.sleep(1)
                if time.time() - begin_time >= timeout_seconds:
                    if response is None:
                        msg = "grpc connnect failed."
                    else:
                        msg = response.status.error_msg
                    raise ConnectionError("Connect coordinator timeout, {}".format(msg))

    def connect(self):
        request = message_pb2.ConnectSessionRequest()
        response = self._connect_session_impl(request)

        self._session_id = response.session_id
        return (
            response.session_id,
            json.loads(response.engine_config),
            response.pod_name_list,
        )

    @property
    def session_id(self):
        return self._session_id

    def __str__(self):
        return "%s" % self._session_id

    def __repr__(self):
        return str(self)

    def run(self, dag_def):
        request = message_pb2.RunStepRequest(
            session_id=self._session_id, dag_def=dag_def
        )
        return self._run_step_impl(request)

    def fetch_logs(self):
        if self._logs_fetching_thread is None:
            self._logs_fetching_thread = threading.Thread(
                target=self._fetch_logs_impl, args=()
            )
            self._logs_fetching_thread.daemon = True
            self._logs_fetching_thread.start()

    @catch_grpc_error
    def send_heartbeat(self):
        request = message_pb2.HeartBeatRequest()
        return self._send_heartbeat_impl(request)

    @catch_grpc_error
    def create_interactive_engine(
        self,
        object_id,
        schema_path,
        gremlin_server_cpu,
        gremlin_server_mem,
        engine_params,
    ):
        request = message_pb2.CreateInteractiveRequest(
            object_id=object_id,
            schema_path=schema_path,
            gremlin_server_cpu=gremlin_server_cpu,
            gremlin_server_mem=gremlin_server_mem,
            engine_params=engine_params,
        )
        return self._create_interactive_engine_impl(request)

    def close_interactive_engine(self, object_id):
        request = message_pb2.CloseInteractiveRequest(object_id=object_id)
        return self._close_interactive_engine_impl(request)

    def create_learning_engine(self, object_id, handle, config):
        request = message_pb2.CreateLearningInstanceRequest(
            object_id=object_id,
            handle=handle,
            config=config,
        )
        response = self._create_learning_engine_impl(request)
        return response.endpoints

    def close_learning_engine(self, object_id):
        request = message_pb2.CloseLearningInstanceRequest(object_id=object_id)
        return self._close_learning_engine_impl(request)

    def close(self):
        if self._session_id:
            request = message_pb2.CloseSessionRequest(session_id=self._session_id)
            self._close_session_impl(request)
            self._session_id = None
        if self._logs_fetching_thread:
            self._logs_fetching_thread.join(timeout=5)

    @catch_grpc_error
    def _connect_session_impl(self, request):
        response = self._stub.ConnectSession(request)
        return check_grpc_response(response)

    @catch_grpc_error
    def _run_step_impl(self, request):
        response = self._stub.RunStep(request)
        return check_grpc_response(response)

    @suppress_grpc_error
    def _fetch_logs_impl(self):
        request = message_pb2.FetchLogsRequest(session_id=self._session_id)
        responses = self._stub.FetchLogs(request)
        for resp in responses:
            resp = check_grpc_response(resp)
            message = resp.message.rstrip()
            if message:
                logger.info(message, extra={"simple": True})

    @catch_grpc_error
    def _send_heartbeat_impl(self, request):
        return check_grpc_response(self._stub.HeartBeat(request))

    @catch_grpc_error
    def _create_interactive_engine_impl(self, request):
        response = self._stub.CreateInteractiveInstance(request)
        return check_grpc_response(response)

    @catch_grpc_error
    def _close_interactive_engine_impl(self, request):
        response = self._stub.CloseInteractiveInstance(request)
        return check_grpc_response(response)

    @catch_grpc_error
    def _create_learning_engine_impl(self, request):
        response = self._stub.CreateLearningInstance(request)
        return check_grpc_response(response)

    @catch_grpc_error
    def _close_learning_engine_impl(self, request):
        response = self._stub.CloseLearningInstance(request)
        return check_grpc_response(response)

    @catch_grpc_error
    def _close_session_impl(self, request):
        response = self._stub.CloseSession(request)
        return check_grpc_response(response)


class GRPCGatewayClient(GRPCClient):
    def __init__(self, endpoint):
        if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
            raise ValueError("GRPCGatewayClient expects a HTTP/HTTPS endpoint")
        self._endpoint = endpoint
        self._session_id = None
        self._logs_fetching_thread = None

    def _run_gateway_stub(self, path, request, response):
        json_payload = json_format.MessageToDict(request)
        resp = requests.post(
            "%s/gs.rpc.CoordinatorService/%s" % (self._endpoint, path),
            json=json_payload,
        )
        if resp.status_code != 200:
            raise GRPCError(
                "requests call '%s': failed with error %s, message is %s"
                % (path, resp.status_code, resp.text)
            )
        body = resp.json()
        if "status" not in body:
            code = body.get("code", "")
            message = body.get("message", "")
            details = body.get("details", "")
            raise GRPCError(
                "requests call '%s': failed with error %s, message is %s, details is %s"
                % (path, code, message, details)
            )
        return json_format.ParseDict(body, response)

    @catch_grpc_gateway_error
    def _connect_session_impl(self, request):
        response = message_pb2.ConnectSessionResponse()
        response = self._run_gateway_stub("ConnectSession", request, response)
        return check_grpc_response(response)

    @catch_grpc_gateway_error
    def _run_step_impl(self, request):
        tag = str(time.time())
        request.tag = tag
        response = message_pb2.RunStepResponse()
        while True:
            response = self._run_gateway_stub("RunStep", request, response)
            if response.status.code == error_codes_pb2.UNKNOWN:
                message = response.status.error_msg
                if not (message == 'pending' or message == 'running'):
                    break
            else:
                break
            time.sleep(3)
        return check_grpc_response(response)

    @suppress_grpc_gateway_error
    def _fetch_logs_impl(self):
        logger.warning("GRPC gateway call doesn't support fetch logs.")

    @catch_grpc_gateway_error
    def _send_heartbeat_impl(self, request):
        response = message_pb2.HeartBeatResponse()
        response = self._run_gateway_stub("HeartBeat", request, response)
        return check_grpc_response(response)

    @catch_grpc_gateway_error
    def _create_interactive_engine_impl(self, request):
        response = message_pb2.CreateInteractiveResponse()
        response = self._run_gateway_stub(
            "CreateInteractiveInstance", request, response
        )
        return check_grpc_response(response)

    @catch_grpc_gateway_error
    def _close_interactive_engine_impl(self, request):
        response = message_pb2.CloseInteractiveResponse()
        response = self._run_gateway_stub("CloseInteractiveInstance", request, response)
        return check_grpc_response(response)

    @catch_grpc_gateway_error
    def _create_learning_engine_impl(self, request):
        response = message_pb2.CreateLearningInstanceResponse()
        response = self._run_gateway_stub("CreateLearningInstance", request, response)
        return check_grpc_response(response)

    @catch_grpc_gateway_error
    def _close_learning_engine_impl(self, request):
        response = message_pb2.CloseLearningInstanceResponse()
        response = self._run_gateway_stub("CloseLearningInstance", request, response)
        return check_grpc_response(response)

    @catch_grpc_gateway_error
    def _close_session_impl(self, request):
        response = message_pb2.CloseSessionResponse()
        response = self._run_gateway_stub("CloseSession", request, response)
        return check_grpc_response(response)
