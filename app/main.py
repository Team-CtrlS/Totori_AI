from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api import stt_router, story_router, quiz_router, reading_router
from app.core import redis_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    await redis_client.connect()
    yield
    await redis_client.close()

app = FastAPI(lifespan=lifespan)

app.include_router(stt_router.router)
app.include_router(story_router.router)
app.include_router(quiz_router.router)
app.include_router(reading_router.router)

@app.get("/")
def main():
    return {"status": "ok"} 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)