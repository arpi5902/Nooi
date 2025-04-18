from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import time
import threading
from chatbot import get_bot_response
from chatbot import log_chat

app = FastAPI()

# Serve static assets
app.mount("/static", StaticFiles(directory="templates/static"), name="static")
app.mount("/music", StaticFiles(directory="."), name="music")

# Homepage
@app.get("/")
async def index():
    return FileResponse("templates/index.html")

# ✅ Correct POST endpoint for chatbot
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    message = data.get("message", "")
    mode = data.get("mode", None)

    if not message:
        return JSONResponse({"reply": "⚠️ Empty message."})

    response = get_bot_response(message, mode)
    return JSONResponse({"reply": response})

@app.post("/rate")
async def rate(request: Request):
    data = await request.json()
    user = data.get("user", "web_user")
    query = data.get("query", "")
    rating = data.get("rating", 0)
    log_chat(user, query, "", rating=rating)  # log only rating + query
    return JSONResponse({"status": "ok"})

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
