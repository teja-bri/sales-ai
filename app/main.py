from fastapi import FastAPI

app = FastAPI(
    title="Sales AI",
    version="0.1.0",
)


@app.get("/")
def home():
    return {"message": "Sales AI is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}