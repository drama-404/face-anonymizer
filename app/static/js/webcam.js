/**
 * Face Anonymizer App - Webcam and Processing Logic
 * Handles webcam capture, video processing, and file uploads
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const webcamVideo = document.getElementById('webcamVideo');
    const outputCanvas = document.getElementById('outputCanvas');
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const faceCountElement = document.getElementById('faceCount');
    const processingFpsElement = document.getElementById('processingFps');
    const processingLogElement = document.getElementById('processingLog');
    const blurMethodRadios = document.querySelectorAll('input[name="blurMethod"]');
    const blurFactorSlider = document.getElementById('blurFactor');
    const blurValueElement = document.getElementById('blurValue');
    const frameRateSlider = document.getElementById('frameRate');
    const frameRateValueElement = document.getElementById('frameRateValue');
    const imageUploadForm = document.getElementById('imageUploadForm');
    const videoUploadForm = document.getElementById('videoUploadForm');
    const imageResultElement = document.getElementById('imageResult');
    const videoResultElement = document.getElementById('videoResult');
    const imageStatsElement = document.getElementById('imageStats');
    const videoStatsElement = document.getElementById('videoStats');
    const imageDownloadElement = document.getElementById('imageDownload');
    const videoDownloadElement = document.getElementById('videoDownload');
    const processedImageElement = document.getElementById('processedImage');
    
    // App State
    let streaming = false;
    let stream = null;
    let processingTimer = null;
    let lastFrameTime = 0;
    let frameCount = 0;
    let framesLastSecond = 0;
    let fpsUpdateTimer = null;
    
    // Settings
    let blurMethod = 'gaussian';
    let blurFactor = 30;
    let frameRate = 10; // Frames per second for processing
    
    // Canvas context for drawing
    const ctx = outputCanvas.getContext('2d');
    
    /**
     * Start webcam capture
     */
    async function startWebcam() {
        try {
            // Request webcam access
            stream = await navigator.mediaDevices.getUserMedia({
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                },
                audio: false
            });
            
            // Connect stream to video element
            webcamVideo.srcObject = stream;
            
            // Wait for video to start playing
            await new Promise(resolve => {
                webcamVideo.onloadedmetadata = () => {
                    webcamVideo.play();
                    resolve();
                };
            });
            
            // Show output canvas and begin processing
            outputCanvas.style.display = 'block';
            webcamVideo.style.display = 'none';
            
            // Update UI state
            startButton.disabled = true;
            stopButton.disabled = false;
            streaming = true;
            
            // Start processing frames
            startProcessing();
            
            // Log successful webcam start
            addToProcessingLog('Webcam started successfully');
            
        } catch (error) {
            console.error('Error accessing webcam:', error);
            addToProcessingLog('Error: ' + error.message);
            alert('Could not access webcam. Please ensure you have a webcam connected and have granted permission to use it.');
        }
    }
    
    /**
     * Stop webcam capture and processing
     */
    function stopWebcam() {
        if (stream) {
            // Stop all tracks in the stream
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
        
        // Reset video element
        webcamVideo.srcObject = null;
        
        // Stop processing timer
        if (processingTimer) {
            clearInterval(processingTimer);
            processingTimer = null;
        }
        
        // Stop FPS update timer
        if (fpsUpdateTimer) {
            clearInterval(fpsUpdateTimer);
            fpsUpdateTimer = null;
        }
        
        // Update UI state
        startButton.disabled = false;
        stopButton.disabled = true;
        streaming = false;
        
        // Hide canvas, show video element (for next start)
        outputCanvas.style.display = 'none';
        webcamVideo.style.display = 'block';
        
        // Log webcam stop
        addToProcessingLog('Webcam stopped');
    }
    
    /**
     * Start frame processing at specified frame rate
     */
    function startProcessing() {
        // Calculate interval in ms based on frame rate
        const interval = 1000 / frameRate;
        
        // Set up FPS counter updater
        fpsUpdateTimer = setInterval(() => {
            processingFpsElement.textContent = framesLastSecond;
            framesLastSecond = 0;
        }, 1000);
        
        // Start processing timer
        processingTimer = setInterval(processFrame, interval);
    }
    
    /**
     * Process a single frame from the webcam
     */
    async function processFrame() {
        if (!streaming) return;
        
        try {
            // Draw current video frame to canvas
            ctx.drawImage(webcamVideo, 0, 0, outputCanvas.width, outputCanvas.height);
            
            // Get frame data as blob
            const blob = await new Promise(resolve => {
                outputCanvas.toBlob(resolve, 'image/jpeg', 0.8);
            });
            
            // Create form data with frame and settings
            const formData = new FormData();
            formData.append('image', blob, 'frame.jpg');
            formData.append('blur_method', blurMethod);
            formData.append('blur_factor', blurFactor);
            
            // Process frame with backend API
            const response = await fetch('/process_frame', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Error processing frame');
            }
            
            const result = await response.json();
            
            // Update face count
            faceCountElement.textContent = result.face_count;
            
            // Display processed frame
            const img = new Image();
            img.onload = () => {
                ctx.drawImage(img, 0, 0, outputCanvas.width, outputCanvas.height);
                
                // Update FPS counter
                frameCount++;
                framesLastSecond++;
                
                // Log occasional frame processing
                if (frameCount % 30 === 0) {
                    addToProcessingLog(`Processed frame ${frameCount}: ${result.face_count} faces detected`);
                }
            };
            img.src = result.processed_image;
            
        } catch (error) {
            console.error('Error in frame processing:', error);
            addToProcessingLog('Processing error: ' + error.message);
        }
    }
    
    /**
     * Upload and process an image file
     * @param {FormData} formData Form data with image file and settings
     */
    async function uploadAndProcessImage(formData) {
        try {
            // Show loading state
            document.querySelector('#image-upload').classList.add('loading');
            
            // Send to backend for processing
            const response = await fetch('/upload_image', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Error processing image');
            }
            
            const result = await response.json();
            
            // Update UI with results
            imageStatsElement.textContent = `Processed image with ${result.faces_detected} faces detected`;
            imageDownloadElement.href = result.download_url;
            
            // Fetch and display the processed image
            processedImageElement.src = result.download_url;
            imageResultElement.style.display = 'block';
            
            // Log success
            addToProcessingLog(`Processed uploaded image: ${result.faces_detected} faces anonymized`);
            
        } catch (error) {
            console.error('Error uploading image:', error);
            addToProcessingLog('Error uploading image: ' + error.message);
            alert('Error processing image: ' + error.message);
        } finally {
            // Hide loading state
            document.querySelector('#image-upload').classList.remove('loading');
        }
    }
    
    /**
     * Upload and process a video file
     * @param {FormData} formData Form data with video file and settings
     */
    async function uploadAndProcessVideo(formData) {
        try {
            // Show progress bar and loading state
            const progressBar = document.querySelector('.progress');
            const progressBarInner = document.querySelector('.progress-bar');
            progressBar.style.display = 'block';
            document.querySelector('#video-upload').classList.add('loading');
            
            // Set up progress monitoring with fetch
            const response = await fetch('/upload_video', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Error processing video');
            }
            
            const result = await response.json();
            
            // Update UI with results
            videoStatsElement.textContent = `Processed video: ${result.total_frames} frames, ${result.total_faces} faces detected (avg: ${result.avg_faces_per_frame} faces/frame)`;
            videoDownloadElement.href = result.download_url;
            videoResultElement.style.display = 'block';
            
            // Log success
            addToProcessingLog(`Processed uploaded video: ${result.total_frames} frames, ${result.total_faces} total faces`);
            
        } catch (error) {
            console.error('Error uploading video:', error);
            addToProcessingLog('Error uploading video: ' + error.message);
            alert('Error processing video: ' + error.message);
        } finally {
            // Hide loading state and progress bar
            document.querySelector('#video-upload').classList.remove('loading');
            document.querySelector('.progress').style.display = 'none';
        }
    }
    
    /**
     * Add a message to the processing log
     * @param {string} message Log message to add
     */
    function addToProcessingLog(message) {
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.textContent = `[${timestamp}] ${message}`;
        processingLogElement.appendChild(logEntry);
        processingLogElement.scrollTop = processingLogElement.scrollHeight;
        
        // Limit log size
        while (processingLogElement.children.length > 100) {
            processingLogElement.removeChild(processingLogElement.firstChild);
        }
    }
    
    // Event Listeners
    
    // Start webcam button
    startButton.addEventListener('click', startWebcam);
    
    // Stop webcam button
    stopButton.addEventListener('click', stopWebcam);
    
    // Blur method selection
    blurMethodRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            blurMethod = e.target.value;
            addToProcessingLog(`Changed blur method to: ${blurMethod}`);
        });
    });
    
    // Blur factor slider
    blurFactorSlider.addEventListener('input', (e) => {
        blurFactor = parseInt(e.target.value);
        blurValueElement.textContent = blurFactor;
    });
    
    // Processing frame rate slider
    frameRateSlider.addEventListener('input', (e) => {
        frameRate = parseInt(e.target.value);
        frameRateValueElement.textContent = frameRate;
        
        // Update processing interval if active
        if (processingTimer) {
            clearInterval(processingTimer);
            processingTimer = setInterval(processFrame, 1000 / frameRate);
            addToProcessingLog(`Changed processing frame rate to: ${frameRate} fps`);
        }
    });
    
    // Image upload form
    imageUploadForm.addEventListener('submit', (e) => {
        e.preventDefault();
        
        const imageFile = document.getElementById('imageFile').files[0];
        if (!imageFile) {
            alert('Please select an image file');
            return;
        }
        
        const formData = new FormData();
        formData.append('image', imageFile);
        formData.append('blur_method', blurMethod);
        formData.append('blur_factor', blurFactor);
        
        uploadAndProcessImage(formData);
    });
    
    // Video upload form
    videoUploadForm.addEventListener('submit', (e) => {
        e.preventDefault();
        
        const videoFile = document.getElementById('videoFile').files[0];
        if (!videoFile) {
            alert('Please select a video file');
            return;
        }
        
        const formData = new FormData();
        formData.append('video', videoFile);
        formData.append('blur_method', blurMethod);
        formData.append('blur_factor', blurFactor);
        
        uploadAndProcessVideo(formData);
    });
    
    // Add initial log message
    addToProcessingLog('Face Anonymizer initialized and ready');
});