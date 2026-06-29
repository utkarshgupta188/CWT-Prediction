import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from crypto_prediction.routes import health, market, predict
from crypto_prediction.database.repository import init_db
from crypto_prediction.utils.logger import setup_logger
from crypto_prediction.schemas.config import settings

logger = setup_logger(settings.LOG_LEVEL)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Database
    logger.info("Initializing database on startup...")
    await init_db()
    yield
    logger.info("Shutting down backend app...")

app = FastAPI(
    title="Crypto Prediction Research System API",
    description="A multi-agent system using Hermes Agent, Kronos and Kelly Criterion for prediction research.",
    version="1.0.0",
    lifespan=lifespan
)

# Register routes
app.include_router(health.router, tags=["Health"])
app.include_router(market.router, tags=["Markets"])
app.include_router(predict.router, tags=["Prediction"])

from fastapi.responses import HTMLResponse
from pathlib import Path

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_ui():
    template_path = Path(__file__).parent / "templates" / "index.html"
    if not template_path.exists():
        return HTMLResponse("<h2>UI Template Not Found</h2>", status_code=404)
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}...")
    uvicorn.run(
        "crypto_prediction.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False
    )
