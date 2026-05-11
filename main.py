"""ASGI entrypoint compatibility module.

Some hosts default to `main:app` when launching Uvicorn.
Re-export the FastAPI app from server.py so both `main:app` and `server:app` work.
"""

from server import app

