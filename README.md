# Face Anonymizer App

A real-time face detection and anonymization web application built with FastAPI, OpenCV, and LangChain.

## Features

- Real-time webcam face detection and anonymization
- Support for both Gaussian blur and pixelation effects
- Upload and process static images and videos
- Adjustable blur intensity and processing frame rate
- Face detection statistics and processing logs
- LangChain integration for enhanced functionality

## Requirements

- Python 3.8+
- OpenCV
- FastAPI
- LangChain
- See `requirements.txt` for full dependencies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/drama-404/face-anonymizer.git
cd face-anonymizer
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```

2. Open your browser and navigate to:
```
http://localhost:8000
```

3. Use the web interface to:
   - Start real-time webcam anonymization
   - Upload images or videos for processing
   - Adjust anonymization settings
   - View processing statistics

## Project Structure

```
face-anonymizer/
├── app/
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   ├── face_processor.py
│   ├── langchain_utils.py
│   └── main.py
├── templates/
│   └── index.html
├── requirements.txt
└── README.md
```

## API Endpoints

- `GET /`: Main web interface
- `POST /process_frame`: Process a single webcam frame
- `POST /upload_image`: Process an uploaded image
- `POST /upload_video`: Process an uploaded video
- `GET /download`: Download processed files

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for computer vision capabilities
- FastAPI for the web framework
- LangChain for enhanced functionality
- Bootstrap for the UI components