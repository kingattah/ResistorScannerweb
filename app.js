let video = document.getElementById('videoInput');
let canvasOutput = document.getElementById('canvasOutput');
let startButton = document.getElementById('startButton');
let stopButton = document.getElementById('stopButton');
let zoomSlider = document.getElementById('zoomSlider');
let resistorValue = document.getElementById('resistorValue');
let stream = null;
let processing = false;

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
        processVideo();
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
        if (zoom !== 1) {
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
    // Convert to grayscale
    let gray = new cv.Mat();
    cv.cvtColor(frame, gray, cv.COLOR_RGBA2GRAY);

    // Apply Gaussian blur
    let blurred = new cv.Mat();
    cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);

    // Edge detection
    let edges = new cv.Mat();
    cv.Canny(blurred, edges, 50, 150);

    // Find contours
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    // Process contours to find potential resistors
    for (let i = 0; i < contours.size(); i++) {
        let contour = contours.get(i);
        let area = cv.contourArea(contour);
        
        // Filter based on area to find potential resistors
        if (area > 1000) {
            let rect = cv.boundingRect(contour);
            // Draw rectangle around potential resistor
            cv.rectangle(frame, new cv.Point(rect.x, rect.y),
                new cv.Point(rect.x + rect.width, rect.y + rect.height),
                new cv.Scalar(0, 255, 0, 255), 2);
            
            // TODO: Implement color band detection and value calculation
            // This would involve analyzing the colors within the detected rectangle
        }
    }

    // Clean up
    gray.delete();
    blurred.delete();
    edges.delete();
    contours.delete();
    hierarchy.delete();
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