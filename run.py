import os
import sys
from pathlib import Path

# Helper to load local .env variables at startup (avoids external dependencies)
def load_env():
    for path in [Path(".env"), Path(__file__).resolve().parent / ".env", Path(__file__).resolve().parent.parent / ".env"]:
        if path.exists() and path.is_file():
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        os.environ.setdefault(k, v)
            break

# Load environment configurations
load_env()

# 1. Environment variables & Debug check
DEBUG_MODE = os.getenv("DEBUG", "False").lower() in ("true", "1", "t", "yes")

# If NOT in debug mode, verify critical environment variables
CRITICAL_ENV_VARS = ["DATABASE_URL", "AI_RECOMMENDER_API_KEY"]
missing_vars = [var for var in CRITICAL_ENV_VARS if not os.getenv(var)]

if missing_vars and not DEBUG_MODE:
    sys.stderr.write(f"❌ Production Start Blocked: Missing critical environment variables: {', '.join(missing_vars)}\n")
    sys.stderr.write("Please set these variables in the environment or set DEBUG=True for local development.\n")
    sys.exit(1)

# Default development fallback values if in debug mode
if DEBUG_MODE:
    os.environ.setdefault("DATABASE_URL", "sqlite:///data/edupro.db")
    os.environ.setdefault("AI_RECOMMENDER_API_KEY", "dev_mock_key_12345")

# 2. Monkeypatch Streamlit's Starlette middleware to inject security headers
try:
    import streamlit.web.server.starlette.starlette_app as starlette_app
    from starlette.middleware import Middleware
    class SecurityHeadersSender:
        def __init__(self, send):
            self.send = send

        async def __call__(self, message):
            if message["type"] == "http.response.start":
                # Shallow copy to avoid modifying original Uvicorn dictionaries
                message = dict(message)
                headers = list(message.get("headers", []))

                # Security headers list (must be bytes keys/values)
                override_headers = [
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"deny"),
                    (b"content-security-policy", (
                        b"default-src * 'unsafe-inline' 'unsafe-eval'; "
                        b"connect-src * ws: wss:; "
                        b"style-src * 'unsafe-inline'; "
                        b"img-src * data: blob:; "
                        b"font-src * data:;"
                    ))
                ]

                # Conditionally enforce HSTS only in production
                if not DEBUG_MODE:
                    override_headers.append((b"strict-transport-security", b"max-age=31536000; includeSubDomains"))

                new_headers = []
                override_keys = {name for name, _ in override_headers}

                # Copy existing headers, omitting overridden ones
                for h_name, h_val in headers:
                    if h_name.lower() not in override_keys:
                        new_headers.append((h_name, h_val))

                # Append security headers
                new_headers.extend(override_headers)
                message["headers"] = new_headers

            await self.send(message)

    class SecurityHeadersMiddleware:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                await self.app(scope, receive, send)
                return

            # Wrap the send channel with the dedicated class to avoid closure reference leaks
            await self.app(scope, receive, SecurityHeadersSender(send))

    # Wrap the original create_streamlit_middleware function
    original_create_middleware = starlette_app.create_streamlit_middleware
    
    def secure_create_streamlit_middleware():
        middleware_list = original_create_middleware()
        middleware_list.insert(0, Middleware(SecurityHeadersMiddleware))
        return middleware_list

    starlette_app.create_streamlit_middleware = secure_create_streamlit_middleware
except Exception as patch_err:
    sys.stderr.write(f"Warning: Failed to inject security headers middleware: {patch_err}\n")

# 3. Launch Streamlit app
from streamlit.web.cli import main

if __name__ == "__main__":
    port = os.getenv("PORT", "8501")
    sys.argv = ["streamlit", "run", "app/streamlit_app.py", "--server.port", port]
    sys.exit(main())
