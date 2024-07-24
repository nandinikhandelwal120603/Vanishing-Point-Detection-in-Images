# Import needed libraries
import matplotlib.pyplot as plt
import cv2  
import os
import numpy as np
import random
import math

# Helper functions

# Show images given a list of images
def show_images(image):
    plt.figure()
    plt.imshow(image, cmap='gray')

# Load images from a folder given their filenames
def load_images(filename):
    try:
        img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        return img
    except IOError:
        print("File is not an image\n")
        exit()

# Plot lines on original images 
def show_lines(image, lines, vanishing_point=None):
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(image, pt1, pt2, (255, 0, 0), 1)
    
    if vanishing_point:
        for line in lines[:2]:  # Draw only the two most important lines
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = vanishing_point
            cv2.line(image, pt1, pt2, (0, 255, 0), 1)
    
    plt.imshow(image)
    plt.axis('off')
    plt.show()
        
# Plot lines and points on original images 
def show_point(image, point):
    cv2.circle(image, point, 3, (0, 255, 0), thickness=3)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

## 1. Detect lines in the image
## Use the Canny edge detector and Hough transform to detect lines in the image.

def detect_lines(image):
    # Do blurry to smooth the image, try to remove edges from textures
    blur_image = cv2.GaussianBlur(image, (5, 5), 1.5)
    # Canny edge detection with OpenCV for all blurry images
    edge_image = cv2.Canny(blur_image, 50, 150, apertureSize=3, L2gradient=True)
    # Use hough transform to detect all lines
    lines = cv2.HoughLines(edge_image, 1, np.pi / 180, 100)
    valid_lines = []
    # Remove horizontal and vertical lines as they would not converge to vanishing point
    for line in lines:
        rho, theta = line[0]
        if (theta > 0.1 and theta < 1.5) or (theta > 1.7 and theta < 3.0):
            valid_lines.append(line)
    
    return blur_image, edge_image, valid_lines

### 2. Locate the vanishing point
### Use RANSAC to locate the vanishing point from the detected lines.

#### 2.1 RANSAC functions
#### Define two functions required by RANSAC: a function to find the point where lines intersect, and a function to compute the distance from a point to a line.

# Find the intersection point
def find_intersection_point(line1, line2):
    # rho and theta for each line
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    # Used a formula from https://stackoverflow.com/a/383527/5087436 to solve for intersection between 2 lines 
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ]) 
    b = np.array([[rho1], [rho2]])
    det_A = np.linalg.det(A)
    if det_A != 0:
        x0, y0 = np.linalg.solve(A, b)
        # Extract single elements from the array before converting
        x0, y0 = int(np.round(x0.item())), int(np.round(y0.item()))
        return x0, y0
    else:
        return None
        
# Find the distance from a point to a line
def find_dist_to_line(point, line):
    x0, y0 = point
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    dist = np.abs(a * x0 + b * y0 - rho) / np.sqrt(a ** 2 + b ** 2)
    return dist

#### 2.2 RANSAC loop
#### Define the main RANSAC loop

def RANSAC(lines, ransac_iterations, ransac_threshold, ransac_ratio):
    inlier_count_ratio = 0.
    vanishing_point = (0, 0)
    # perform RANSAC iterations for each set of lines
    for iteration in range(ransac_iterations):
        # randomly sample 2 lines
        selected_lines = random.sample(lines, 2)
        line1 = selected_lines[0]
        line2 = selected_lines[1]
        intersection_point = find_intersection_point(line1, line2)
        if intersection_point is not None:
            inlier_count = 0
            # inliers are lines whose distance to the point is less than ransac_threshold
            for line in lines:
                dist = find_dist_to_line(intersection_point, line)
                if dist < ransac_threshold:
                    inlier_count += 1

            if inlier_count / float(len(lines)) > inlier_count_ratio:
                inlier_count_ratio = inlier_count / float(len(lines))
                vanishing_point = intersection_point

            if inlier_count > len(lines) * ransac_ratio:
                break
    return vanishing_point, selected_lines

### 3. Main function
### Run your vanishing point detection method on a folder of images, return the (x, y) locations of the vanishing points

# RANSAC parameters:
ransac_iterations, ransac_threshold, ransac_ratio = 800, 10, 0.8

if __name__ == "__main__":
    folder_path = r"C:\Users\khand\OneDrive\Desktop\ogmen_robotics\Estimate_vanishing_points_data"
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for image_file in image_files:
        filename = os.path.join(folder_path, image_file)
        print(f"Processing {filename}")
        image = load_images(filename)
        blur_image, edge_image, lines = detect_lines(image)
        vanishing_point, important_lines = RANSAC(lines, ransac_iterations, ransac_threshold, ransac_ratio)
        show_point(image, vanishing_point)
        show_lines(image, important_lines, vanishing_point)
