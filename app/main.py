from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil
import uuid
import cv2
import numpy as np
import tempfile
from typing import Optional, Literal
import logging
import base64
from pathlib import Path

# Import our modules
from app.face_processor import FaceProcessor
from app.langchain_utils import AnonymizerLangChain

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Face Anonymizer")

# Setup templates and static files directories
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Ensure directories exist for model files
os.makedirs("models", exist_ok=True)

# Check for OpenCV face detection model or download it
model_file = "models/opencv_face_detector_uint8.pb"
config_file = "models/opencv_face_detector.pbtxt"

# Download model if not exists (placeholder - in production would download real models)
if not os.path.exists(model_file) or not os.path.exists(config_file):
    logger.info("OpenCV DNN face detection model not found, will use Haar cascade fallback")

# Create face processor and LangChain utilities
face_processor = FaceProcessor(blur_method="gaussian", blur_factor=30)
anonymizer_chain = AnonymizerLangChain()

# Create temp directory for file uploads
temp_dir = tempfile.gettempdir()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process_frame")
async def process_frame(image: UploadFile = File(...), 
                        blur_method: Literal["gaussian", "pixelate"] = Form("gaussian"),
                        blur_factor: int = Form(30)):
    """
    Process a single frame uploaded from the webcam
    
    Args:
        image: Base64 encoded image from the webcam
        blur_method: Method of blur to apply
        blur_factor: Intensity of blur effect
        
    Returns:
        Processed image with faces anonymized
    """
    try:
        # Update processor settings
        face_processor.set_blur_method(blur_method)
        face_processor.set_blur_factor(blur_factor)
        
        # Read and decode image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Process the frame to detect and anonymize faces
        processed_frame, face_count = face_processor.process_frame(img)
        
        # Encode the processed frame as base64 JPEG
        _, encoded_img = cv2.imencode('.jpg', processed_frame)
        encoded_img_str = base64.b64encode(encoded_img).decode('utf-8')
        
        # Return the processed image and metadata
        return {
            "processed_image": f"data:image/jpeg;base64,{encoded_img_str}",
            "face_count": face_count
        }
    
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing frame: {str(e)}")

@app.post("/upload_video")
async def upload_video(video: UploadFile = File(...),
                       blur_method: Literal["gaussian", "pixelate"] = Form("gaussian"),
                       blur_factor: int = Form(30)):
    """
    Upload and process a video file
    
    Args:
        video: Video file to process
        blur_method: Method of anonymization
        blur_factor: Intensity of anonymization effect
        
    Returns:
        Path to processed video and processing stats
    """
    try:
        # Save uploaded video to temp file
        temp_file = os.path.join(temp_dir, f"upload_{uuid.uuid4().hex}_{video.filename}")
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Process the video using LangChain
        result = anonymizer_chain.process_uploaded_video(
            video_path=temp_file,
            blur_method=blur_method,
            blur_factor=blur_factor
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Return the output video path for download
        return {
            "output_path": result["output_path"],
            "download_url": f"/download?file={os.path.basename(result['output_path'])}",
            "total_frames": result["total_frames"],
            "total_faces": result["total_faces_detected"],
            "avg_faces_per_frame": round(result["avg_faces_per_frame"], 2)
        }
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.post("/upload_image")
async def upload_image(image: UploadFile = File(...),
                       blur_method: Literal["gaussian", "pixelate"] = Form("gaussian"),
                       blur_factor: int = Form(30)):
    """
    Upload and process an image file
    
    Args:
        image: Image file to process
        blur_method: Method of anonymization
        blur_factor: Intensity of anonymization effect
        
    Returns:
        Path to processed image and processing stats
    """
    try:
        # Save uploaded image to temp file
        temp_file = os.path.join(temp_dir, f"upload_{uuid.uuid4().hex}_{image.filename}")
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Process the image using LangChain
        result = anonymizer_chain.process_uploaded_image(
            image_path=temp_file,
            blur_method=blur_method,
            blur_factor=blur_factor
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Return the output image path for download
        return {
            "output_path": result["output_path"],
            "download_url": f"/download?file={os.path.basename(result['output_path'])}",
            "faces_detected": result["faces_detected"]
        }
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/download")
async def download_file(file: str):
    """
    Download a processed file
    
    Args:
        file: Filename to download (basename only for security)
        
    Returns:
        FileResponse for download
    """
    try:
        file_path = os.path.join(temp_dir, file)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # For security, verify this is a file we created
        if not file.startswith("anonymized_"):
            raise HTTPException(status_code=403, detail="Access denied")
        
        return FileResponse(
            path=file_path, 
            filename=file.split('_', 2)[-1],  # Remove the prefix
            media_type="application/octet-stream"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

# Startup event to initialize models if needed
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Face Anonymizer App")
    
    # Check if we need to download models
    if not os.path.exists(model_file) or not os.path.exists(config_file):
        # In a real app, we'd download the models here
        logger.info("Face detection model not found, will use fallback method")

# Return 404 for undefined routes
@app.exception_handler(404)
async def custom_404_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": f"Route {request.url.path} not found"}
    )

if __name__ == "__main__":
    import uvicorn
    # Start server
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)