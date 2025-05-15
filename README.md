# Face Anonymizer: Real-time face detection and blurring web app

## Key Components:

### Backend (FastAPI):

Face detection using OpenCV and a pre-trained model
Image processing for face blurring/pixelation
API endpoints for processing uploaded videos
LangChain integration for additional functionality


### Frontend:

Real-time webcam capture using JavaScript
Display of processed video with blurred/pixelated faces
Controls for starting/stopping video and toggling blur methods
Upload functionality for pre-recorded videos


### Dependencies:

FastAPI
OpenCV
LangChain
uvicorn (ASGI server)
python-multipart (for file uploads)