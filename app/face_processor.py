import cv2
import numpy as np
from typing import Tuple, List, Union, Literal


class FaceProcessor:
    def __init__(self, 
                 blur_method: Literal["gaussian", "pixelate"] = "gaussian",
                 blur_factor: int = 30, 
                 detection_confidence: float = 0.5):
        """
        Initialize face processor with detection model and blur parameters
        
        Args:
            blur_method: Method used for face anonymization ('gaussian' or 'pixelate')
            blur_factor: Intensity of blur effect (higher = more blur)
            detection_confidence: Confidence threshold for face detection (0-1)
        """
        self.blur_method = blur_method
        self.blur_factor = blur_factor
        self.detection_confidence = detection_confidence
        
        # Load pre-trained face detection model from OpenCV DNN module
        # Using a pre-trained model from the OpenCV face detector
        model_file = "models/opencv_face_detector_uint8.pb"
        config_file = "models/opencv_face_detector.pbtxt"
        
        try:
            self.face_net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
            self.model_loaded = True
        except:
            # Fallback to Haar cascade if DNN model files aren't available
            print("DNN face detection model not found, falling back to Haar cascade")
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.model_loaded = False
        
        self.face_count = 0
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the given frame
        
        Args:
            frame: Input image frame as numpy array
            
        Returns:
            List of face rectangles as (x, y, width, height)
        """
        height, width = frame.shape[:2]
        faces = []
        
        if self.model_loaded:
            # Using DNN-based detector (more accurate)
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
            self.face_net.setInput(blob)
            detections = self.face_net.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > self.detection_confidence:
                    x1 = int(detections[0, 0, i, 3] * width)
                    y1 = int(detections[0, 0, i, 4] * height)
                    x2 = int(detections[0, 0, i, 5] * width)
                    y2 = int(detections[0, 0, i, 6] * height)
                    
                    # Convert to (x, y, width, height) format
                    faces.append((x1, y1, x2-x1, y2-y1))
        else:
            # Fallback to Haar cascade detector
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            for (x, y, w, h) in detected_faces:
                faces.append((x, y, w, h))
        
        self.face_count = len(faces)
        return faces
    
    def apply_gaussian_blur(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Apply Gaussian blur to a face region
        
        Args:
            frame: Input image frame
            face: Face rectangle (x, y, width, height)
            
        Returns:
            Frame with blurred face
        """
        x, y, w, h = face
        # Create a safe region within image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        # Extract face region and apply blur
        face_roi = frame[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_roi, (self.blur_factor, self.blur_factor), 0)
        
        # Replace original face with blurred face
        frame[y:y+h, x:x+w] = blurred_face
        
        return frame
    
    def apply_pixelation(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Apply pixelation effect to a face region
        
        Args:
            frame: Input image frame
            face: Face rectangle (x, y, width, height)
            
        Returns:
            Frame with pixelated face
        """
        x, y, w, h = face
        # Create a safe region within image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        # Extract face region
        face_roi = frame[y:y+h, x:x+w]
        
        # Pixelate by scaling down and up
        pixelation_factor = max(1, self.blur_factor // 10)  # Convert blur factor to pixelation scale
        
        small = cv2.resize(face_roi, (w // pixelation_factor, h // pixelation_factor), 
                          interpolation=cv2.INTER_LINEAR)
        pixelated_face = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Replace original face with pixelated face
        frame[y:y+h, x:x+w] = pixelated_face
        
        return frame
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Process a frame to detect and blur/pixelate faces
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple of (processed frame, number of faces detected)
        """
        # Detect faces
        faces = self.detect_faces(frame)
        
        # Apply blur/pixelation to each detected face
        for face in faces:
            if self.blur_method == "gaussian":
                frame = self.apply_gaussian_blur(frame, face)
            else:  # pixelate
                frame = self.apply_pixelation(frame, face)
                
        return frame, self.face_count
    
    def set_blur_method(self, method: Literal["gaussian", "pixelate"]):
        """
        Set the blur method
        
        Args:
            method: 'gaussian' or 'pixelate'
        """
        if method in ["gaussian", "pixelate"]:
            self.blur_method = method
    
    def set_blur_factor(self, factor: int):
        """
        Set the blur factor
        
        Args:
            factor: Intensity of blur effect
        """
        self.blur_factor = max(1, factor)