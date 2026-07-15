from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import shutil
import uuid
import gc
import threading
import time
from typing import List, Dict
import logging 
import argparse

from .models import llm_registry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


loaded_models: Dict[str, object] = {}
model_management_lock = threading.Lock() 

def unload_all_models():
    """Safely unload all currently loaded models to free memory."""
    global loaded_models
    if not loaded_models:
        return
    
    model_names = list(loaded_models.keys())
    logging.info(f"Preparing to unload models: {model_names}...")
    
    loaded_models.clear()
    gc.collect()

    if TORCH_AVAILABLE and torch.xpu.is_available():
        logging.info("Cleaning up XPU cache...")
        torch.xpu.empty_cache()
    elif TORCH_AVAILABLE and torch.cuda.is_available():
        logging.info("Cleaning up CUDA cache...")
        torch.cuda.empty_cache()
    
    logging.info(f"Models {model_names} unloaded, memory freed.")

def load_model(llm_type: str, model_path: str):
    """Load the specified model and unload all old models before loading."""
    global loaded_models
    logging.info(f"--- Received load model request: {llm_type} ---")
    
    # Unload old models
    unload_all_models()
    
    try:
        logging.info(f"Creating model instance '{llm_type}' from registry...")
        
        model = llm_registry[llm_type](model_path=model_path)
        loaded_models[llm_type] = model
        logging.info(f"--- Model '{llm_type}' loaded successfully. ---")
    except Exception as e:
        # Using exc_info=True will automatically record complete error stack trace information, perfect for debugging
        logging.error(f"Failed to load model '{llm_type}'. Error: {e}", exc_info=True)
        loaded_models.clear()
        raise e

app = FastAPI(title="LLM Inference Server")

UPLOADS_DIR = "server_uploads1"
os.makedirs(UPLOADS_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "LLM Inference Server is running."}

@app.post("/loading/")
async def handle_load_model(
    llm_type: str = Form(...),
    ):
    request_id = str(uuid.uuid4())[:8]  
    with model_management_lock:
        if llm_type not in loaded_models:
            logging.info(f"[{request_id}] Requested model '{llm_type}' not loaded, loading now...")
            try:
                load_model(llm_type)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

@app.post("/inference/")
async def handle_inference(
    llm_type: str = Form(...),
    prompt: str = Form(...),
    video_files: List[UploadFile] = File(...)
):
    request_id = str(uuid.uuid4())[:8]  # Use a short ID for easy tracking
    logging.info(f"--- [Request ID: {request_id}] Received new inference request ---")

    # 1. Model loading logic
    with model_management_lock:
        if llm_type not in loaded_models:
            logging.info(f"[{request_id}] Requested model '{llm_type}' not loaded, loading now...")
            try:
                load_model(llm_type, app.model_path)  # Pass the model_path from app state
            except Exception as e:
                # Log was already recorded in load_model, only return API error here
                raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
    
    llm = loaded_models.get(llm_type)
    if not llm:
        raise HTTPException(status_code=500, detail=f"Model '{llm_type}' still cannot be retrieved after loading, please check server logs.")
    
    # 2. File processing
    temp_dir = os.path.join(UPLOADS_DIR, request_id)
    os.makedirs(temp_dir)
    image_paths = []
    
    logging.info(f"[{request_id}] Received {len(video_files)} files, saving to temporary directory: {temp_dir}")
    
    try:
        for file in video_files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            image_paths.append(file_path)
        
        image_paths.sort()
        logging.info(f"[{request_id}] All files saved.")

        # 3. Execute inference
        logging.info(f"[{request_id}] Starting inference on {len(image_paths)} images...")
        
        start_time = time.time()
        result = llm.inference(image_paths, prompt)
        end_time = time.time()
        
        logging.info(f"[{request_id}] Inference completed, duration: {end_time - start_time:.2f} seconds.")
        logging.info(f"[{request_id}] Inference completed, returning: {result}")
        
        return JSONResponse(content={"result": result})

    except Exception as e:
        logging.error(f"[{request_id}] Error occurred during inference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error occurred during inference: {e}")
    finally:
        # 4. Clean up temporary files
        if os.path.exists(temp_dir):
            logging.info(f"[{request_id}] Cleaning up temporary files and directory: {temp_dir}")
            shutil.rmtree(temp_dir)
        logging.info(f"--- [Request ID: {request_id}] Request processing completed ---")


def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM Inference Server")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the RynnBrain model"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the server on"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to run the server on"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    app.model_path = args.model_path  # Store model_path in app state so it can be accessed by endpoints
    logging.info(f"Starting LLM Inference Server with model path: {args.model_path}")
    uvicorn.run(app, host=args.host, port=args.port)