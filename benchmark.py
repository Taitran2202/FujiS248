import cv2
import numpy as np
import timeit

# Generate a random 320x320 image
image = np.random.randint(0, 256, (320, 320), dtype=np.uint8)
output_image = np.zeros_like(image)
# Find contours and measure the time it takes
def find_contours_benchmark():
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours and measure the time it takes
def draw_contours_benchmark():
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(output_image, contours, -1, (255, 0, 0), 2)

# Benchmark the findContours function
find_contours_time = timeit.timeit(find_contours_benchmark, number=10)

# Benchmark the drawContours function
draw_contours_time = timeit.timeit(draw_contours_benchmark, number=10)
cv2.imshow("result",output_image)
cv2.waitKey(-1)
print("findContours Time: {:.5f} seconds".format(find_contours_time))
print("drawContours Time: {:.5f} seconds".format(draw_contours_time))