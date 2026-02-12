"""
Context-Aware Emoji Prediction - Full Stack Launcher
Run with: python main.py
Serves both the FastAPI backend and React frontend on a single port.
"""
import os
import sys
import subprocess
import importlib.util
import socket

# Fix Windows terminal encoding for emoji/unicode
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
DIST_DIR = os.path.join(FRONTEND_DIR, "dist")

PORT = 8000


def free_port(port):
    """Kill any process using the given port."""
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True, text=True, shell=True
            )
            for line in result.stdout.strip().split("\n"):
                if f":{port}" in line and "LISTENING" in line:
                    pid = line.strip().split()[-1]
                    try:
                        subprocess.run(
                            ["taskkill", "/F", "/PID", pid],
                            capture_output=True, shell=True
                        )
                        print(f"   [OK] Freed port {port} (killed PID {pid})")
                    except Exception:
                        pass
        else:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True
            )
            if result.stdout.strip():
                for pid in result.stdout.strip().split("\n"):
                    subprocess.run(["kill", "-9", pid], capture_output=True)
                print(f"   [OK] Freed port {port}")
    except Exception:
        pass


def is_port_in_use(port):
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def build_frontend():
    """Build the React frontend if dist doesn't exist."""
    if not os.path.exists(os.path.join(DIST_DIR, "index.html")):
        print("\n   Building frontend...")
        try:
            subprocess.run(
                ["npm", "run", "build"],
                cwd=FRONTEND_DIR,
                check=True,
                shell=True,
            )
            print("   [OK] Frontend built successfully!\n")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("   [FAIL] Frontend build failed.")
            print("   Run 'cd frontend && npm install && npm run build' first.")
            sys.exit(1)
    else:
        print("   [OK] Frontend build found.")


def load_backend_app():
    """Load the backend FastAPI app using importlib to avoid circular imports."""
    sys.path.insert(0, BACKEND_DIR)

    spec = importlib.util.spec_from_file_location(
        "backend_main",
        os.path.join(BACKEND_DIR, "main.py")
    )
    backend_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend_module)

    return backend_module.app


def mount_frontend(app):
    """Mount the React frontend static files on the FastAPI app."""
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    assets_dir = os.path.join(DIST_DIR, "assets")
    if os.path.exists(assets_dir):
        app.mount(
            "/assets",
            StaticFiles(directory=assets_dir),
            name="frontend-assets",
        )

    @app.get("/vite.svg")
    async def serve_vite_svg():
        return FileResponse(os.path.join(DIST_DIR, "vite.svg"))

    # Catch-all: serve index.html for SPA client-side routing
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = os.path.join(DIST_DIR, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(DIST_DIR, "index.html"))


def main():
    print("")
    print("=" * 50)
    print("  Context-Aware Emoji Prediction")
    print("  Full Stack Application")
    print("=" * 50)
    print("")

    # Step 0: Free the port if in use
    if is_port_in_use(PORT):
        print(f"[0/3] Port {PORT} is in use, freeing it...")
        free_port(PORT)
        import time
        time.sleep(2)

    # Step 1: Check frontend build
    print("[1/3] Checking frontend build...")
    build_frontend()

    # Step 2: Load backend
    print("[2/3] Loading backend & NLP pipeline...")
    app = load_backend_app()

    # Step 3: Mount frontend
    print("[3/3] Mounting frontend...")
    mount_frontend(app)
    print("   [OK] Frontend mounted.\n")

    print("=" * 50)
    print("  Ready! Open in your browser:")
    print("")
    print(f"  App:      http://localhost:{PORT}")
    print(f"  API Docs: http://localhost:{PORT}/docs")
    print("=" * 50)
    print("")

    import uvicorn
    uvicorn.run(app, host="localhost", port=PORT)


if __name__ == "__main__":
    main()
