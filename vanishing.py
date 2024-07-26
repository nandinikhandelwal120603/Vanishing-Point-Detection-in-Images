
import os
import cv2
import math
import numpy as np

#Threshold by which lines will be rejected wrt the horizontal

REJECT_DEGREE_TH = 4.0 #Also for rejecting lines based on the angles . lines with angles close to 0 or 90 degrees will be filtered out 


#this function reads images from either a single file or a directory of images.
def ReadImage(InputImagePath):
    Images = []  # Input Images will be stored in this list.
    ImageNames = []  # Names of input images will be stored in this list.

    # Checking if path is of file or folder.
    if os.path.isfile(InputImagePath):  # If path is of file.
        InputImage = cv2.imread(InputImagePath)  # Reading the image.

        # Checking if image is read.
        if InputImage is None:
            print("Image not read. Provide a correct path")
            exit()

        Images.append(InputImage)  # Storing the image.
        ImageNames.append(os.path.basename(InputImagePath))  # Storing the image's name.

    # If path is of a folder containing images.
    elif os.path.isdir(InputImagePath):
        # Getting all image's names present inside the folder.
        for ImageName in os.listdir(InputImagePath):
            # Reading images one by one.
            InputImage = cv2.imread(os.path.join(InputImagePath, ImageName))

            if InputImage is not None:
                Images.append(InputImage)  # Storing images.
                ImageNames.append(ImageName)  # Storing image's names.
        
    # If it is neither file nor folder (Invalid Path).
    else: # Error Handling: If the path is invalid or the image can't be read, it prints an error and exits.
        print("\nEnter valid Image Path.\n")
        exit()

    return Images, ImageNames

# this function : helps us to find straight lines prenet in the imge that on extending will approx merge at vasnishing point 
def FilterLines(Lines):
    FinalLines = []
    
    for Line in Lines:
        [[x1, y1, x2, y2]] = Line

        # Calculate slope and intercept safely
        if x1 != x2:
            m = (y2 - y1) / (x2 - x1)
        else:
            m = float('inf')  # Use infinity for vertical lines
        
        if m == float('inf'):
            c = x1  # For vertical lines, use x as the 'intercept'
        else:
            c = y1 - m * x1

        theta = math.degrees(math.atan(m)) if m != float('inf') else 90

        # Rejecting lines of slope near to 0 degree or 90 degree and storing others
        if REJECT_DEGREE_TH <= abs(theta) <= (90 - REJECT_DEGREE_TH):
            l = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)  # length of the line
            FinalLines.append([x1, y1, x2, y2, m, c, l])

    # Removing extra lines
    if len(FinalLines) > 15:
        FinalLines = sorted(FinalLines, key=lambda x: x[-1], reverse=True)
        FinalLines = FinalLines[:15]
    
    return FinalLines


# finding lines using Probablistic hough line transform 
# hough line transfoom : The Probabilistic Hough Transform works by randomly selecting a subset of the edge pixels in the image 
# and then fitting lines to those pixels. 
# This process is repeated multiple times, 
# with the hope that each iteration will detect additional pixels that belong to the same line.

def GetLines(Image):
    # step1 : Converting to grayscale --> features better detected 
    GrayImage = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    # setp2: blurring image to reduce noise --> s that are not requiredand may lead to false positives
    BlurGrayImage = cv2.GaussianBlur(GrayImage, (5, 5), 1)
    # step3 : generating Edge image --> simple yet effective method of finding the edges 
    EdgeImage = cv2.Canny(BlurGrayImage, 40, 255)

    
    '''
      edge_image (numpy.ndarray): A binary edge image (1 channel).
      rho (int, optional): The resolution of the r parameter in pixels. Defaults to 1.
      theta (float, optional): The resolution of the theta parameter in radians. Defaults to np.pi/180 (1 degree).
      threshold (int, optional): The minimum number of intersections to consider a line. Defaults to 100.
      min_line_length (int, optional): The minimum length (number of points) for a valid line. Defaults to 50.
      max_line_gap (int, optional): The maximum allowed gap between points in a line. Defaults to 10.

  Returns:
      list: A list of detected lines, where each line is represented as a tuple (x1, y1, x2, y2).
    '''
    
    
    # Finding Lines in the image
    Lines = cv2.HoughLinesP(EdgeImage, 1, np.pi / 180, 50, 10, 15)

    # Check if lines found and exit if not.
    #
    if Lines is None:
        print("Not enough lines found in the image for Vanishing Point detection.")
        exit(0)
    
    # Filtering Lines wrt angle
    #We can filter lines based on their length in one step during the filtering stage.
    
    FilteredLines = FilterLines(Lines)

    return FilteredLines

def GetVanishingPoint(Lines):
    
    '''
    Intersection Calculation: Find the intersection points of all pairs of lines.
    Distance Calculation: Compute the distance from each intersection point to all other lines.
    Error Minimization: Sum the squared distances and select the point with the minimum sum as the vanishing point.
    '''
    VanishingPoint = None
    MinError = float('inf')

    for i in range(len(Lines)):
        for j in range(i+1, len(Lines)):
            m1, c1 = Lines[i][4], Lines[i][5]
            m2, c2 = Lines[j][4], Lines[j][5]

            if m1 != m2:
                x0 = (c1 - c2) / (m2 - m1)
                y0 = m1 * x0 + c1

                err = 0
                for k in range(len(Lines)):
                    m, c = Lines[k][4], Lines[k][5]
                    if m != float('inf'):
                        m_ = -1 / m
                        c_ = y0 - m_ * x0
                        x_ = (c - c_) / (m_ - m)
                        y_ = m_ * x_ + c_
                        l = math.sqrt((y_ - y0)**2 + (x_ - x0)**2)
                        err += l**2
                    else:
                        # Handle vertical lines
                        if c == x0:
                            l = abs(y0 - y0)
                            err += l**2

                err = math.sqrt(err)

                if MinError > err:
                    MinError = err
                    VanishingPoint = [x0, y0]
                
    return VanishingPoint

if __name__ == "__main__":
    # update the path to the folder containing the images
    Images, ImageNames = ReadImage("C:\\Users\\khand\\OneDrive\\Desktop\\ogmen_robotics\\Estimate_vanishing_points_data")
    
    for i in range(len(Images)):
        Image = Images[i]

        # getting the lines from the image
        Lines = GetLines(Image)

        # get vanishing point
        VanishingPoint = GetVanishingPoint(Lines)

        # checking if vanishing point found
        if VanishingPoint is None:
            print("Vanishing Point not found. Possible reason is that not enough lines are found in the image for determination of vanishing point.")
            continue

        # drawing lines and vanishing point
        for Line in Lines:
            cv2.line(Image, (Line[0], Line[1]), (Line[2], Line[3]), (0, 255, 0), 2)
        cv2.circle(Image, (int(VanishingPoint[0]), int(VanishingPoint[1])), 10, (0, 0, 255), -1)
        
        # Save the final image
        output_image_path = os.path.join("C:\\Users\\khand\\OneDrive\\Desktop\\ogmen_robotics", f"OutputImage_{i}.jpg")
        cv2.imwrite(output_image_path, Image)
        print(f"Image saved as {output_image_path}")

#-------------------------------------------------------------------------------------------------------------------------
