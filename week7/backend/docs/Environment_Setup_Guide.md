# Environment Setup Guide: FastAPI Backend

This guide outlines the steps required to configure, install, and run the production-ready FastAPI backend for the AI Fashion Design Assistant.

---

## 1. Prerequisites
- **Python 3.11.x** (Verify your version using `python --version`)
- **pip** package installer

---

## 2. Virtual Environment Setup

Create and activate a isolated python virtual environment within the repository root:

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Dependency Installation

Navigate to the `week7/` directory and install all required python libraries:

```bash
cd week7/
pip install -r requirements.txt
```

---

## 4. Configuration & Environment Variables

The backend uses Pydantic V2 to automatically parse and type-validate configuration inputs from environment variables.

1. Navigate to the configs directory:
   ```bash
   cd backend/configs/
   ```
2. Copy the template `.env.example` file to `.env`:
   - On Windows: `copy .env.example .env`
   - On macOS/Linux: `cp .env.example .env`
3. Edit `.env` to configure settings:
   - **`SECURITY__JWT_SECRET_KEY`**: Set a cryptographically secure 32-character key for token signing.
   - **`MODEL__GLOBAL_MOCK`**: Set to `true` (default) for offline/CPU testing, or `false` to utilize real GPU model pipelines.

---

## 5. Running the Web Server

Start the FastAPI application using **Uvicorn** from the `week7/` directory:

```bash
# Execute from week7/ folder:
python -m uvicorn week7.backend.main:app --reload --host 127.0.0.1 --port 8000
```

- **`--reload`**: Automatically restarts the server upon file modifications (highly recommended for local development).
- **`--host`**: Defines the IP binding address.
- **`--port`**: Defines the target TCP port.

---

## 6. Accessing Interactive API Documentation

Once the server launches, you can view and test endpoints interactively in your browser:

- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) (Interactive OpenAPI exploration)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc) (Clean, structured documentation layout)
