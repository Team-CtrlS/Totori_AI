from fastapi import FastAPI
from app.api import refiner_router, stt_router

app = FastAPI()

app.include_router(stt_router.router)
app.include_router(refiner_router.router)

@app.get("/")
def main():
    return {"status": "ok"} 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)