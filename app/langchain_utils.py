from langchain.agents import Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import BaseLLM
from typing import Dict, List, Any, Optional
import cv2
import numpy as np
import os
import uuid
import tempfile
from datetime import datetime

class AnonymizerLangChain:
    """
    LangChain integration for the Face Anonymizer app.
    Provides tools for processing static images/videos when webcam capture isn't available.
    """
    
    def __init__(self, llm: Optional[BaseLLM] = None):
        """
        Initialize the LangChain integration
        
        Args:
            llm: Optional LangChain language model for enhanced functionality
        """
        self.llm = llm
        self.temp_dir = tempfile.gettempdir()
        
        # Create tools that can be used with LangChain agents
        self.tools = self._create_tools()
        
    def _create_tools(self) -> List[Tool]:
        """Create LangChain tools for face anonymization operations"""
        
        tools = [
            Tool(
                name="process_uploaded_video",
                description="Process an uploaded video file to anonymize faces",
                func=self.process_uploaded_video
            ),
            Tool(
                name="process_uploaded_image",
                description="Process an uploaded image file to anonymize faces",
                func=self.process_uploaded_image
            )
        ]
        
        return tools
    
    def process_uploaded_video(self, 
                              video_path: str, 
                              blur_method: str = "gaussian", 
                              blur_factor: int = 30) -> Dict[str, Any]:
        """
        Process an uploaded video file to detect and anonymize faces
        
        Args:
            video_path: Path to uploaded video file
            blur_method: Method of anonymization ('gaussian' or 'pixelate')
            blur_factor: Intensity of anonymization effect
            
        Returns:
            Dictionary with output video path and processing stats
        """
        from app.face_processor import FaceProcessor
        
        # Create face processor
        processor = FaceProcessor(
            blur_method=blur_method,
            blur_factor=blur_factor
        )
        
        try:
            # Open video capture
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": "Could not open video file"}
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Generate output filename
            output_path = os.path.join(
                self.temp_dir, 
                f"anonymized_{uuid.uuid4().hex}_{os.path.basename(video_path)}"
            )
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Process each frame
            total_faces = 0
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process the frame to detect and anonymize faces
                processed_frame, face_count = processor.process_frame(frame)
                total_faces += face_count
                frame_count += 1
                
                # Write the processed frame to output video
                out.write(processed_frame)
                
                # Print progress update (every 10% of frames)
                if frame_count % max(1, total_frames // 10) == 0:
                    print(f"Processed {frame_count}/{total_frames} frames")
            
            # Release resources
            cap.release()
            out.release()
            
            # Return results
            return {
                "output_path": output_path,
                "total_frames": frame_count,
                "total_faces_detected": total_faces,
                "avg_faces_per_frame": total_faces / max(1, frame_count),
                "blur_method": blur_method
            }
        
        except Exception as e:
            return {"error": f"Error processing video: {str(e)}"}
    
    def process_uploaded_image(self, 
                              image_path: str, 
                              blur_method: str = "gaussian", 
                              blur_factor: int = 30) -> Dict[str, Any]:
        """
        Process an uploaded image file to detect and anonymize faces
        
        Args:
            image_path: Path to uploaded image file
            blur_method: Method of anonymization ('gaussian' or 'pixelate')
            blur_factor: Intensity of anonymization effect
            
        Returns:
            Dictionary with output image path and processing stats
        """
        from app.face_processor import FaceProcessor
        
        # Create face processor
        processor = FaceProcessor(
            blur_method=blur_method,
            blur_factor=blur_factor
        )
        
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not open image file"}
            
            # Process the image to detect and anonymize faces
            processed_image, face_count = processor.process_frame(image)
            
            # Generate output filename
            output_path = os.path.join(
                self.temp_dir, 
                f"anonymized_{uuid.uuid4().hex}_{os.path.basename(image_path)}"
            )
            
            # Save processed image
            cv2.imwrite(output_path, processed_image)
            
            # Return results
            return {
                "output_path": output_path,
                "faces_detected": face_count,
                "blur_method": blur_method,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        except Exception as e:
            return {"error": f"Error processing image: {str(e)}"}
    
    def create_processing_chain(self, prompt_template: str) -> LLMChain:
        """
        Create a LangChain for processing decisions
        
        Args:
            prompt_template: Template for the LLM prompt
            
        Returns:
            LLMChain for processing
        """
        if self.llm is None:
            raise ValueError("LLM must be provided to create a processing chain")
        
        prompt = PromptTemplate(
            input_variables=["input"],
            template=prompt_template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain