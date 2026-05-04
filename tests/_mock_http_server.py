"""Subprocess-based mock HTTP server for testing Rust HTTP clients.

This module is designed to be run as a subprocess to provide true process
isolation when testing Rust HTTP clients (like reqwest) from Python tests.

Usage:
    python -m tests._mock_http_server <port> <json_response>
    python -m tests._mock_http_server <port> <json_routes_dict>

The server:
- Listens on 127.0.0.1:<port>
- Prints "READY:<port>" to stdout when ready
- For simple mode: handles one POST request, prints "BODY:<json>" to stdout
- For routes mode: handles multiple requests based on path matching
- Responds with the provided JSON response
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer


class CaptureHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures request body and returns configured response."""

    def _handle_request(self, method: str):
        try:
            self._handle_request_inner(method)
        except Exception:  # noqa: BLE001
            # If anything throws inside the handler the response never
            # reaches the client and reqwest reports a confusing
            # "connection closed before message completed" instead of the
            # actual root cause. Surface the full traceback to stderr so
            # the test driver can capture it.
            import sys as _sys
            import traceback as _traceback

            print(
                f"MOCK HANDLER EXCEPTION ({method} {self.path}):",
                file=_sys.stderr,
                flush=True,
            )
            _traceback.print_exc(file=_sys.stderr)
            _sys.stderr.flush()
            raise

    def _handle_request_inner(self, method: str):
        """Common handler for all HTTP methods."""
        # Read request body if present
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b""

        path = self.path
        routes = getattr(self.server, "routes", None)

        if routes:
            # Routes mode: prefer method-keyed routes ("POST /tags/"), then
            # fall back to path-only routes ("/tags/"). Insertion order wins
            # within each group.
            response_data = None
            for route_key, route_response in routes.items():
                if " " in route_key:
                    route_method, route_path = route_key.split(" ", 1)
                    if method == route_method and path.startswith(route_path):
                        response_data = route_response
                        break
            if response_data is None:
                for route_key, route_response in routes.items():
                    if " " in route_key:
                        continue
                    if path.startswith(route_key):
                        response_data = route_response
                        break

            if response_data is None:
                # Default response for unmatched routes
                response_data = '""'

            # Print request info for debugging / structured capture. Body
            # is decoded as latin-1 (always lossless) since PUTs of binary
            # frame data won't be valid UTF-8.
            body_str = body.decode("latin-1") if body else ""
            print(f"REQUEST:{method}:{path}:{body_str}", flush=True)
        else:
            # Simple mode: single response for all requests
            response_data = self.server.response_data
            print(f"BODY:{body.decode()}", flush=True)

        # Send response.
        # `Connection: close` tells reqwest's pool not to reuse this socket
        # for the next request. Default Python BaseHTTPRequestHandler is
        # HTTP/1.0, so each connection only handles one request anyway —
        # but reqwest doesn't know that and will try to pipeline. Without
        # this header the next pooled request sees a half-closed socket
        # and fails ("PUT bytes failed" / "connection closed before message
        # completed"), most reproducibly on Windows.
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response_data))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(response_data.encode())

    def do_GET(self):
        self._handle_request("GET")

    def do_POST(self):
        self._handle_request("POST")

    def do_PUT(self):
        self._handle_request("PUT")

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


class MultiRequestServer(ThreadingHTTPServer):
    """Threaded HTTP server that handles concurrent connections.

    Tokio uploads from multiple streams arrive in tight bursts; serving
    them serially (or with a small listen backlog) caused silent connection
    drops under load. Using ThreadingHTTPServer + a generous backlog lets
    the mock keep up.
    """

    request_queue_size = 256

    def __init__(self, *args, max_requests: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_requests = max_requests
        self._count_lock = threading.Lock()
        self.request_count = 0
        self._done = threading.Event()

    def handle_requests(self):
        """Serve until max_requests have been handled (or forever if the
        server is told to shut down).
        """
        thread = threading.Thread(target=self.serve_forever, daemon=True)
        thread.start()
        self._done.wait()
        self.shutdown()
        thread.join(timeout=2)

    def process_request_thread(self, request, client_address):  # noqa: D401
        """Override to count handled requests and signal completion."""
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._count_lock:
                self.request_count += 1
                if self.request_count >= self.max_requests:
                    self._done.set()


def run_server(port: int, response_data: str) -> None:
    """Run the mock HTTP server."""
    # Try to parse as routes dict
    try:
        routes = json.loads(response_data)
        if isinstance(routes, dict) and routes.get("__routes__"):
            # Routes mode
            del routes["__routes__"]
            max_requests = routes.pop("__max_requests__", 10)
            server = MultiRequestServer(
                ("127.0.0.1", port), CaptureHandler, max_requests=max_requests
            )
            server.routes = routes
            print(f"READY:{port}", flush=True)
            server.handle_requests()
            return
    except (json.JSONDecodeError, TypeError):
        pass

    # Simple mode: single request with static response
    server = HTTPServer(("127.0.0.1", port), CaptureHandler)
    server.response_data = response_data
    print(f"READY:{port}", flush=True)
    server.handle_request()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <port> <json_response>", file=sys.stderr)
        sys.exit(1)

    port = int(sys.argv[1])
    response_data = sys.argv[2]
    run_server(port, response_data)
