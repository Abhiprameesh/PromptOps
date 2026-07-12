from fastapi import FastAPI

from app.api.inference import router as inference_router

app = FastAPI(
    title="PromptOps API",
    version="1.0.0",
    description="LLM Regression Detection Platform"
)

app.include_router(inference_router)


@app.get("/")
async def root():
    return {"message": "PromptOps is running"}