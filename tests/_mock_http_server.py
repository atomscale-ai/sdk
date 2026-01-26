"""Subprocess-based mock HTTP server for testing Rust HTTP clients.

This module is designed to be run as a subprocess to provide true process
isolation when testing Rust HTTP clients (like reqwest) from Python tests.

Usage:
    python -m tests._mock_http_server <port> <json_response>

The server:
- Listens on 127.0.0.1:<port>
- Prints "READY:<port>" to stdout when ready
- Handles one POST request, prints "BODY:<json>" to stdout
- Responds with the provided JSON response
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class CaptureHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures request body and returns configured response."""

    def do_POST(self):
        # Read and output request body
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        print(f"BODY:{body.decode()}", flush=True)

        # Send response
        response = self.server.response_data
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response))
        self.end_headers()
        self.wfile.write(response.encode())

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def run_server(port: int, response_data: str) -> None:
    """Run the mock HTTP server."""
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
