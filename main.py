from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import numpy as np
import uuid
import os
import logging
from datetime import datetime
import traceback
from explainer_fallback import SimpleExplainer

# Initialize FastAPI
app = FastAPI(
    title="Explainable AI API",
    description="API for generating ML model explanations using SHAP, LIME, and other techniques",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for uploaded files and results
storage = {
    "models": {},
    "datasets": {},
    "results": {},
    "tasks": {}
}

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup"""
    logger.info("Explainable AI API starting up")
    os.makedirs("temp_uploads", exist_ok=True)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Explainable AI API shutting down")
    import shutil
    if os.path.exists("temp_uploads"):
        try:
            shutil.rmtree("temp_uploads")
        except Exception as e:
            logger.warning(f"Could not clean temp directory: {e}")

@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "Explainable AI API is running",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    """Upload a machine learning model file"""
    try:
        if not file.filename.endswith(('.pkl', '.joblib')):
            raise HTTPException(status_code=400, detail="Only .pkl and .joblib files supported")
        
        content = await file.read()
        if len(content) > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=400, detail="File too large. Max 50MB.")
        
        model_id = str(uuid.uuid4())
        temp_path = f"temp_uploads/model_{model_id}_{file.filename}"
        
        with open(temp_path, "wb") as buffer:
            buffer.write(content)
        
        # Validate model
        try:
            model = joblib.load(temp_path)
            if not hasattr(model, 'predict'):
                raise ValueError("Model must have a 'predict' method")
        except Exception as e:
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail=f"Invalid model: {str(e)}")
        
        storage["models"][model_id] = {
            "filename": file.filename,
            "path": temp_path,
            "uploaded_at": datetime.now().isoformat(),
            "size": len(content)
        }
        
        logger.info(f"Model uploaded: {model_id} - {file.filename}")
        return {"model_id": model_id, "filename": file.filename, "size": len(content)}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset file"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files supported")
        
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File too large. Max 10MB.")
        
        dataset_id = str(uuid.uuid4())
        temp_path = f"temp_uploads/dataset_{dataset_id}_{file.filename}"
        
        with open(temp_path, "wb") as buffer:
            buffer.write(content)
        
        # Validate dataset
        try:
            df = pd.read_csv(temp_path)
            if df.empty:
                raise ValueError("Dataset is empty")
        except Exception as e:
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail=f"Invalid dataset: {str(e)}")
        
        storage["datasets"][dataset_id] = {
            "filename": file.filename,
            "path": temp_path,
            "uploaded_at": datetime.now().isoformat(),
            "size": len(content),
            "shape": df.shape,
            "columns": df.columns.tolist()
        }
        
        logger.info(f"Dataset uploaded: {dataset_id} - {file.filename}")
        return {
            "dataset_id": dataset_id, 
            "filename": file.filename, 
            "size": len(content),
            "shape": df.shape,
            "columns": df.columns.tolist()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def generate_explanations_task(task_id: str, model_id: str, dataset_id: str):
    """Background task to generate explanations"""
    try:
        storage["tasks"][task_id]["status"] = "processing"
        storage["tasks"][task_id]["progress"] = "Loading model and dataset..."
        
        # Load model and dataset
        model_info = storage["models"][model_id]
        dataset_info = storage["datasets"][dataset_id]
        
        model = joblib.load(model_info["path"])
        df = pd.read_csv(dataset_info["path"])
        
        # Limit dataset size for memory efficiency
        if len(df) > 5000:
            df = df.head(5000)
            logger.info("Dataset truncated to 5000 rows")
        
        # Initialize explainer
        explainer = SimpleExplainer()
        
        storage["tasks"][task_id]["progress"] = "Generating predictions..."
        
        # Prepare data and generate predictions
        model_feature_names = explainer.get_model_feature_names(model)
        df_clean = explainer.clean_data(df)
        
        if model_feature_names:
            df_clean = explainer.align_features(df_clean, model_feature_names)
        
        predictions = explainer.safe_predict(model, df_clean)
        
        storage["tasks"][task_id]["progress"] = "Generating explanations..."
        
        # Generate explanations
        explanations, business_explanation, fidelity_scores = explainer.generate_explanations(
            model, df, predictions
        )
        
        # Prepare response
        result = {
            "model_info": {
                "filename": model_info["filename"],
                "type": explainer.detect_model_type(model, predictions)
            },
            "dataset_info": {
                "filename": dataset_info["filename"],
                "shape": dataset_info["shape"],
                "processed_samples": len(predictions)
            },
            "predictions": {
                "count": len(predictions),
                "sample": predictions[:10].tolist(),
                "statistics": {
                    "mean": float(np.mean(predictions)),
                    "std": float(np.std(predictions)),
                    "min": float(np.min(predictions)),
                    "max": float(np.max(predictions))
                }
            },
            "explanations": explanations,
            "business_explanation": business_explanation,
            "fidelity_scores": fidelity_scores,
            "generated_at": datetime.now().isoformat()
        }
        
        storage["results"][task_id] = result
        storage["tasks"][task_id]["status"] = "completed"
        storage["tasks"][task_id]["progress"] = "Explanations generated successfully"
        
        logger.info(f"Explanations generated successfully for task: {task_id}")
        
    except Exception as e:
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        
        storage["tasks"][task_id]["status"] = "failed"
        storage["tasks"][task_id]["error"] = error_msg
        storage["tasks"][task_id]["traceback"] = error_traceback
        
        logger.error(f"Explanation generation failed for task {task_id}: {error_msg}")

@app.post("/generate-explanations")
async def generate_explanations(model_id: str, dataset_id: str, background_tasks: BackgroundTasks):
    """Generate explanations for uploaded model and dataset"""
    try:
        if model_id not in storage["models"]:
            raise HTTPException(status_code=404, detail="Model not found")
        
        if dataset_id not in storage["datasets"]:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        task_id = str(uuid.uuid4())
        
        storage["tasks"][task_id] = {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "status": "queued",
            "progress": "Task queued for processing",
            "created_at": datetime.now().isoformat()
        }
        
        background_tasks.add_task(generate_explanations_task, task_id, model_id, dataset_id)
        
        logger.info(f"Explanation task queued: {task_id}")
        return {"task_id": task_id, "message": "Explanation generation started", "status": "queued"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start explanation task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Task creation failed: {str(e)}")

@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of an explanation generation task"""
    if task_id not in storage["tasks"]:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = storage["tasks"][task_id]
    
    response = {
        "task_id": task_id,
        "status": task_info["status"],
        "progress": task_info["progress"],
        "created_at": task_info["created_at"]
    }
    
    if task_info["status"] == "failed":
        response["error"] = task_info.get("error", "Unknown error")
    
    return response

@app.get("/results/{task_id}")
async def get_results(task_id: str):
    """Get the results of a completed explanation task"""
    if task_id not in storage["tasks"]:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = storage["tasks"][task_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Task not completed. Status: {task_info['status']}"
        )
    
    if task_id not in storage["results"]:
        raise HTTPException(status_code=404, detail="Results not found")
    
    return storage["results"][task_id]

@app.get("/list-models")
async def list_models():
    """List all uploaded models"""
    models = []
    for model_id, info in storage["models"].items():
        models.append({
            "model_id": model_id,
            "filename": info["filename"],
            "uploaded_at": info["uploaded_at"],
            "size": info["size"]
        })
    return {"models": models}

@app.get("/list-datasets")
async def list_datasets():
    """List all uploaded datasets"""
    datasets = []
    for dataset_id, info in storage["datasets"].items():
        datasets.append({
            "dataset_id": dataset_id,
            "filename": info["filename"],
            "uploaded_at": info["uploaded_at"],
            "size": info["size"],
            "shape": info["shape"]
        })
    return {"datasets": datasets}

@app.delete("/model/{model_id}")
async def delete_model(model_id: str):
    """Delete an uploaded model"""
    if model_id not in storage["models"]:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = storage["models"][model_id]
    if os.path.exists(model_info["path"]):
        os.remove(model_info["path"])
    
    del storage["models"][model_id]
    logger.info(f"Model deleted: {model_id}")
    return {"message": "Model deleted successfully"}

@app.delete("/dataset/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete an uploaded dataset"""
    if dataset_id not in storage["datasets"]:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset_info = storage["datasets"][dataset_id]
    if os.path.exists(dataset_info["path"]):
        os.remove(dataset_info["path"])
    
    del storage["datasets"][dataset_id]
    logger.info(f"Dataset deleted: {dataset_id}")
    return {"message": "Dataset deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)