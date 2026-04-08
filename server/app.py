import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, JSONResponse
from openenv.core.env_server import create_fastapi_app

try:
    from .health_triage_env_environment import HealthTriageEnvironment
    from ..models import HealthAction, HealthObservation
except ImportError:
    from server.health_triage_env_environment import HealthTriageEnvironment
    from models import HealthAction, HealthObservation

app = create_fastapi_app(
    HealthTriageEnvironment, 
    HealthAction, 
    HealthObservation
)

# 1. Fix the root 404
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

# 2. Fix the "/web" 404 (This stops the logs from complaining)
@app.get("/web", include_in_schema=False)
async def web_proxy():
    return JSONResponse(content={"status": "Live", "message": "API is running. Use /docs for UI."})

def main():
    # HF usually listens on 7860, but 8000 is fine if the Dockerfile matches
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()