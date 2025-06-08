let video = document.getElementById('videoInput');
let canvasOutput = document.getElementById('canvasOutput');
let startButton = document.getElementById('startButton');
let stopButton = document.getElementById('stopButton');
let zoomSlider = document.getElementById('zoomSlider');
let resistorValue = document.getElementById('resistorValue');
let stream = null;
let processing = false;
let currentZoom = 1;

// Color bands and their corresponding values
const colorBands = {
    'black': 0,
    'brown': 1,
    'red': 2,
    'orange': 3,
    'yellow': 4,
    'green': 5,
    'blue': 6,
    'violet': 7,
    'grey': 8,
    'white': 9
};

function onOpenCvReady() {
    console.log('OpenCV.js is ready');
    startButton.disabled = false;
}

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'environment',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        });
        video.srcObject = stream;
        startButton.disabled = true;
        stopButton.disabled = false;
        processing = true;
        
        // Wait for video to be ready
        video.onloadedmetadata = () => {
            canvasOutput.width = video.videoWidth;
            canvasOutput.height = video.videoHeight;
            processVideo();
        };
    } catch (err) {
        console.error('Error accessing camera:', err);
        alert('Error accessing camera. Please make sure you have granted camera permissions.');
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        startButton.disabled = false;
        stopButton.disabled = true;
        processing = false;
    }
}

function processVideo() {
    if (!processing) return;

    let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    let dst = new cv.Mat();
    let cap = new cv.VideoCapture(video);

    function processFrame() {
        if (!processing) {
            src.delete();
            dst.delete();
            return;
        }

        cap.read(src);
        
        // Apply zoom
        let zoom = parseFloat(zoomSlider.value);
        if (zoom !== currentZoom) {
            currentZoom = zoom;
            let center = new cv.Point(src.cols / 2, src.rows / 2);
            let matrix = cv.getRotationMatrix2D(center, 0, zoom);
            cv.warpAffine(src, dst, matrix, new cv.Size(src.cols, src.rows));
        } else {
            src.copyTo(dst);
        }

        // Process the frame for resistor detection
        processResistor(dst);

        // Display the processed frame
        cv.imshow('canvasOutput', dst);

        // Schedule the next frame
        requestAnimationFrame(processFrame);
    }

    processFrame();
}

function processResistor(frame) {
    try {
        // Convert to grayscale
        let gray = new cv.Mat();
        cv.cvtColor(frame, gray, cv.COLOR_RGBA2GRAY);

        // Apply Gaussian blur
        let blurred = new cv.Mat();
        cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);

        // Edge detection with adjusted thresholds
        let edges = new cv.Mat();
        cv.Canny(blurred, edges, 30, 100);

        // Find contours
        let contours = new cv.MatVector();
        let hierarchy = new cv.Mat();
        cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

        let maxArea = 0;
        let bestContour = null;

        // Find the largest contour that could be a resistor
        for (let i = 0; i < contours.size(); i++) {
            let contour = contours.get(i);
            let area = cv.contourArea(contour);
            
            // Adjust these thresholds based on your needs
            if (area > 1000 && area < 50000) {
                if (area > maxArea) {
                    maxArea = area;
                    bestContour = contour;
                }
            }
        }

        if (bestContour) {
            let rect = cv.boundingRect(bestContour);
            
            // Draw rectangle around potential resistor
            cv.rectangle(frame, 
                new cv.Point(rect.x, rect.y),
                new cv.Point(rect.x + rect.width, rect.y + rect.height),
                new cv.Scalar(0, 255, 0, 255), 2);

            // Extract the region of interest (ROI)
            let roi = frame.roi(rect);
            
            // Convert to HSV for better color detection
            let hsv = new cv.Mat();
            cv.cvtColor(roi, hsv, cv.COLOR_RGBA2HSV);
            
            // Detect color bands
            let bands = detectColorBands(hsv);
            
            // Calculate resistor value
            if (bands.length >= 3) {
                let value = calculateResistorValue(bands);
                resistorValue.textContent = value;
            } else {
                resistorValue.textContent = 'No resistor detected';
            }
            
            // Clean up ROI processing
            roi.delete();
            hsv.delete();
        } else {
            resistorValue.textContent = 'No resistor detected';
        }

        // Clean up
        gray.delete();
        blurred.delete();
        edges.delete();
        contours.delete();
        hierarchy.delete();
    } catch (err) {
        console.error('Error processing frame:', err);
        resistorValue.textContent = 'Error processing image';
    }
}

function detectColorBands(hsv) {
    let bands = [];
    let width = hsv.cols;
    let height = hsv.rows;
    
    // Sample points along the resistor
    let samplePoints = 10;
    let step = width / (samplePoints + 1);
    
    for (let i = 1; i <= samplePoints; i++) {
        let x = Math.floor(i * step);
        let color = getDominantColor(hsv, x, height/2);
        if (color) {
            bands.push(color);
        }
    }
    
    return bands;
}

function getDominantColor(hsv, x, y) {
    try {
        // Sample a small region around the point
        let region = hsv.roi(new cv.Rect(x-5, y-5, 10, 10));
        let mean = new cv.Mat();
        cv.mean(region, mean);
        
        // Convert HSV to color name
        let h = mean.data[0];
        let s = mean.data[1];
        let v = mean.data[2];
        
        let color = hsvToColorName(h, s, v);
        
        region.delete();
        mean.delete();
        
        return color;
    } catch (err) {
        console.error('Error getting dominant color:', err);
        return null;
    }
}

function hsvToColorName(h, s, v) {
    // HSV ranges for different colors
    if (v < 50) return 'black';
    if (v > 200 && s < 50) return 'white';
    
    if (h >= 0 && h < 30) return 'red';
    if (h >= 30 && h < 60) return 'orange';
    if (h >= 60 && h < 90) return 'yellow';
    if (h >= 90 && h < 150) return 'green';
    if (h >= 150 && h < 210) return 'blue';
    if (h >= 210 && h < 270) return 'violet';
    if (h >= 270 && h < 330) return 'red';
    if (h >= 330 && h < 360) return 'red';
    
    return null;
}

function calculateResistorValue(bands) {
    if (bands.length < 3) return 'Invalid';
    
    let value = '';
    for (let i = 0; i < bands.length - 1; i++) {
        if (colorBands[bands[i]] !== undefined) {
            value += colorBands[bands[i]];
        }
    }
    
    // Add multiplier (last band)
    let multiplier = Math.pow(10, colorBands[bands[bands.length - 1]] || 0);
    let finalValue = parseInt(value) * multiplier;
    
    // Format the value with appropriate units
    if (finalValue >= 1000000) {
        return (finalValue / 1000000) + 'MΩ';
    } else if (finalValue >= 1000) {
        return (finalValue / 1000) + 'kΩ';
    } else {
        return finalValue + 'Ω';
    }
}

// Event listeners
startButton.addEventListener('click', startCamera);
stopButton.addEventListener('click', stopCamera);
zoomSlider.addEventListener('input', () => {
    // Zoom changes will be applied in the next frame
});

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    stopCamera();
}); 