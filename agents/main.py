# agent_server/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import discovery_router, generation_router

app = FastAPI(
    title="Sentinel Agent Server",
    description="A standalone server providing generative AI agents.",
    version="1.0.0"
)
origins = [
    "http://localhost",
    "http://localhost:3000", 
    "http://localhost:5173", 
    "*" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

app.include_router(discovery_router)
app.include_router(generation_router, prefix="/agents")

@app.get("/")
def root():
    return {"message": "Agent Server is running. See /docs for available agents."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)