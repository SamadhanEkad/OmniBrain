import argparse
import re
import sys
import time
from html.parser import HTMLParser
from urllib.parse import urlsplit

import requests


class BundleParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            src = dict(attrs).get("src")
            if src:
                self.scripts.append(src)


def normalize_url(value):
    value = value.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise argparse.ArgumentTypeError("URL must be an absolute http:// or https:// URL")
    if parsed.path:
        raise argparse.ArgumentTypeError("URL must not include a path")
    return f"{parsed.scheme}://{parsed.netloc}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Check an OmniBrain Render deployment")
    parser.add_argument("--backend", required=True, type=normalize_url)
    parser.add_argument("--frontend", required=True, type=normalize_url)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    failures = []

    print(f"Backend: {args.backend}")
    print(f"Frontend: {args.frontend}")

    health = None
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{args.backend}/health", timeout=10)
            print(f"GET /health -> HTTP {response.status_code}")
            if response.ok:
                try:
                    health = response.json()
                    print(health)
                except requests.JSONDecodeError:
                    print(response.text[:500])
                break
        except requests.RequestException as exc:
            print(f"GET /health network error: {type(exc).__name__}")
        time.sleep(5)
    if health is None:
        failures.append("GET /health did not become ready")
        print("FAIL: backend health")

    try:
        ready = requests.get(f"{args.backend}/ready", timeout=15)
        print(f"GET /ready -> HTTP {ready.status_code}")
        print(ready.text[:1000])
        if not ready.ok:
            failures.append("Fix backend readiness configuration and see /ready")
        elif ready.headers.get("content-type", "").startswith("application/json"):
            ready_json = ready.json()
            if ready_json.get("status") != "ready":
                failures.append("Fix not_ready checks shown above")
    except requests.RequestException as exc:
        failures.append("Could not reach /ready")
        print(f"/ready network error: {type(exc).__name__}")

    preflight = requests.options(
        f"{args.backend}/api/documents/answer",
        headers={
            "Origin": args.frontend,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
        timeout=15,
    )
    allowed_origin = preflight.headers.get("Access-Control-Allow-Origin")
    cors_ok = allowed_origin == args.frontend
    print(f"CORS preflight -> {'PASS' if cors_ok else 'FAIL'}")
    if not cors_ok:
        failures.append(
            f"CORS_ORIGINS on the backend must be exactly {args.frontend} "
            "(https, no trailing slash)"
        )

    try:
        frontend = requests.get(args.frontend, timeout=15)
        frontend_ok = frontend.status_code == 200
        print(f"GET frontend -> HTTP {frontend.status_code}: {'PASS' if frontend_ok else 'FAIL'}")
        if not frontend_ok:
            failures.append("Fix the frontend deployment or static-site URL")
        bundle_text = ""
        parser = BundleParser()
        parser.feed(frontend.text)
        for src in parser.scripts:
            bundle_url = src if re.match(r"^https?://", src) else f"{args.frontend}/{src.lstrip('/')}"
            bundle = requests.get(bundle_url, timeout=15)
            bundle.raise_for_status()
            bundle_text += bundle.text
        contains_backend = args.backend in bundle_text
        contains_localhost = "127.0.0.1:8000" in bundle_text
        print(
            "Main JS bundle -> "
            f"{'PASS' if contains_backend and not contains_localhost else 'FAIL'} "
            f"(backend URL {'present' if contains_backend else 'missing'}, "
            f"localhost {'present' if contains_localhost else 'absent'})"
        )
        if not contains_backend or contains_localhost:
            failures.append(
                "Redeploy the frontend after setting VITE_API_URL because it is baked in at build time"
            )
    except requests.RequestException as exc:
        failures.append("Could not download the frontend or main JS bundle")
        print(f"Frontend bundle error: {type(exc).__name__}")

    print("\nSummary")
    if failures:
        for failure in failures:
            print(f"- FAIL: {failure}")
        return 1
    print("- PASS: OmniBrain deployment checks completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
