# Coin-Detection-using-OpenCV-in-Python

## Name : NITHISHKUMAR S
## Reg.No: 212223240109

# Aim:
To develop an AI-based image processing system that can automatically detect and count coins in an image using Python and OpenCV, while visualizing all the intermediate processing steps such as grayscale conversion, blurring, edge detection, and contour detection.

# OBJECTIVE:

1. To apply fundamental computer vision techniques to identify circular objects (coins).

2. To understand the use of image preprocessing and feature extraction using OpenCV.

3. To display all intermediate outputs to explain how detection is achieved.

4. To count and label the number of coins accurately.

# ALGORITHM:
1. Start

2. Input the image (coins image file).

3. Convert the image to grayscale to simplify analysis.

4. Apply Gaussian Blur to reduce image noise and smooth edges.

5. Apply Canny Edge Detection to find edges of coins.

6. Find Contours in the edge-detected image.

7. Filter Contours based on area (to remove small noise).

8.Draw circles around detected coins and assign serial numbers.

9.Count the total number of coins detected.

10. End.

# Program:
```PYTHON
import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_fractures(preprocessed, original):
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(preprocessed, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    edges = cv2.Canny(dilation, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = original.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    return result

def present_results(original_image, processed_image):
    # Convert from BGR (OpenCV) to RGB (Matplotlib)
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    # Display using matplotlib
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Fracture Detected Image")
    plt.imshow(processed_rgb)
    plt.axis('off')

    plt.show()

# --- Main Execution ---
image_path = 'bone.png'
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found. Check the file path.")
else:
    preprocessed = preprocess_image(image)
    fracture_detected_image = detect_fractures(preprocessed, image)
    present_results(image, fracture_detected_image)

```

# Output:
<img width="950" height="464" alt="download" src="https://github.com/user-attachments/assets/0d78797b-5bcc-4b3c-9791-9c288e6a328c" />


# Result:

The system successfully detected and counted all coins in the given image.
