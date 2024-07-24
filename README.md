# Vanishing-Point-Detection-in-Images
The goal was to identify straight lines in images and determine the point where these lines converge, known as the vanishing point
---

### Report on Vanishing Point Detection in Images

**Introduction:**

In this project, I implemented an algorithm to detect vanishing points in images using the Hough Line Transform. The goal was to identify straight lines in images and determine the point where these lines converge, known as the vanishing point. This report outlines my approach, results, and challenges faced during the implementation.

---

**Approach:**

1. **Image Reading:**
   - Developed a function to read images from either a single file or a directory. The function checks if the provided path is a file or a folder and reads the images accordingly.

2. **Image Preprocessing:**
   - Converted images to grayscale to enhance feature detection.
   - Applied Gaussian blur to reduce noise and avoid false positives during edge detection.

3. **Edge Detection:**
   - Used the Canny edge detector to generate edge images. The thresholds for edge detection were iteratively adjusted to balance between detecting significant edges and avoiding noise.

4. **Line Detection:**
   - Applied the Probabilistic Hough Line Transform to detect lines in the edge images. The parameters for this transform, including threshold, min_line_length, and max_line_gap, were fine-tuned to improve the accuracy of line detection.

5. **Line Filtering:**
   - Filtered the detected lines based on their angles to reject lines that were close to horizontal or vertical, as they do not contribute significantly to vanishing point detection.

6. **Vanishing Point Calculation:**
   - Calculated the intersection points of all pairs of lines.
   - Computed the distance from each intersection point to all other lines and selected the point with the minimum sum of squared distances as the vanishing point.

---

**Results:**

- Successfully detected vanishing points in several test images.
- Visualized the detected lines and vanishing points by drawing them on the images.
- Saved the processed images with the detected lines and vanishing points marked for easy verification.

---

**Challenges Faced:**

1. **Parameter Tuning:**
   - Finding the correct values for the Canny edge detection thresholds, Hough Line Transform threshold, min_line_length, and max_line_gap was challenging. It required multiple iterations to balance between detecting enough lines and avoiding noise.
   - Automated parameter tuning using grid search or random search could help streamline this process in the future.

2. **Image Quality:**
   - Blurry or noisy images posed a significant challenge for edge detection and line detection. Preprocessing steps like denoising and adaptive thresholding could be explored to enhance image quality before edge detection.

3. **Filtering Lines:**
   - Filtering lines based on their angles was essential to avoid including lines that were too close to horizontal or vertical. However, setting the reject degree threshold required careful consideration to avoid discarding important lines.

4. **Error Calculation:**
   - Calculating the error for each intersection point involved handling special cases like vertical lines. Ensuring accurate distance computation for all types of lines was crucial for determining the correct vanishing point.

---

**Conclusion:**

The project successfully demonstrated the approach to vanishing point detection using the Hough Line Transform. Despite the challenges faced, the method proved effective for several test images. Future work could focus on automating parameter tuning and enhancing image preprocessing techniques to improve robustness and accuracy.

---
---

**Future Work:**

- Enhance preprocessing steps to handle noisy and blurry images more effectively.
- Implement automated parameter tuning for edge detection and line detection.
- Explore advanced techniques like deep learning for more robust and accurate vanishing point detection.

**References:**

- OpenCV Documentation: [Hough Line Transform](https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html)
- Canny Edge Detection: [Wikipedia](https://en.wikipedia.org/wiki/Canny_edge_detector)

