"""
Simple local web server for testing the MCA FAQ Chatbot HTML frontend.
Uses only Python's standard library - zero external packages needed.

Usage:
    python server.py
Then open:
    http://localhost:8000
"""

import http.server
import socketserver
import os
import webbrowser

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)


def start_server():
    os.chdir(DIRECTORY)
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}/index.html"
        print("=" * 65)
        print(f"🎓 MCA FAQ Chatbot Local Web Server running at: {url}")
        print("Press Ctrl+C to stop the server.")
        print("=" * 65)
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server. Goodbye!")


if __name__ == "__main__":
    start_server()
