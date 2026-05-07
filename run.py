"""
Launcher script: Starts both FastAPI backend and Streamlit frontend.
"""

import subprocess
import sys
import time
import signal
import os

FASTAPI_PORT = 8008
STREAMLIT_PORT = 8501



def main():
    """Launch both FastAPI and Streamlit servers."""
    print("=" * 60)
    print("  Multimodal Clinical Decision Support System")
    print("=" * 60)
    print()

    processes = []

    # Start FastAPI
    print(f"[1/2] Starting FastAPI backend on port {FASTAPI_PORT}...")
    fastapi_cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", str(FASTAPI_PORT),
        "--reload",
    ]
    fastapi_proc = subprocess.Popen(
        fastapi_cmd,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    processes.append(fastapi_proc)
    time.sleep(2)

    # Start Streamlit
    print(f"[2/2] Starting Streamlit frontend on port {STREAMLIT_PORT}...")
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run",
        "streamlit_app/app.py",
        "--server.port", str(STREAMLIT_PORT),
        "--server.headless", "true",
        "--theme.base", "dark",
    ]
    streamlit_proc = subprocess.Popen(
        streamlit_cmd,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    processes.append(streamlit_proc)

    print()
    print(f"  FastAPI:   http://localhost:{FASTAPI_PORT}")
    print(f"  API Docs:  http://localhost:{FASTAPI_PORT}/docs")
    print(f"  Streamlit: http://localhost:{STREAMLIT_PORT}")
    print()
    print("  Press Ctrl+C to stop all services")
    print("=" * 60)

    def signal_handler(sig, frame):
        print("\nShutting down...")
        for p in processes:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()


if __name__ == "__main__":
    main()
