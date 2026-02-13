from fastapi import FastAPI
from app.api.stt.stt_router import router

app = FastAPI()

app.include_router(router)

@app.get("/")
def main():
    return {"status": "ok"} 