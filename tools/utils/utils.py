import math
import cv2
import numpy as np


def CropImage(image,box):
    x,y,w,h,angle =box
    # Calculate the rotation matrix
    M = cv2.getRotationMatrix2D((x, y), -angle, 1)

    # Apply the rotation to the image
    rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Calculate the coordinates of the top-left corner of the rotated rectangle
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)

    # Crop the image
    cropped_image = rotated_image[y1:y1+int(h), x1:x1+int(w)]
    return cropped_image

def rotate_rectangle(image, xc, yc, w, h, angle_rad, color=(255, 0, 0), thickness=1):
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        x1 = (xc - w/2 * cos_a - h/2 * sin_a)
        y1 = (yc - w/2 * sin_a + h/2 * cos_a)
        x2 = (xc + w/2 * cos_a - h/2 * sin_a)
        y2 = (yc + w/2 * sin_a + h/2 * cos_a)
        x3 = (xc + w/2 * cos_a + h/2 * sin_a)
        y3 = (yc + w/2 * sin_a - h/2 * cos_a)
        x4 = (xc - w/2 * cos_a + h/2 * sin_a)
        y4 = (yc - w/2 * sin_a - h/2 * cos_a)
        try:
            for x1, y1, x2, y2, x3, y3, x4, y4 in zip(x1, y1, x2, y2, x3, y3, x4, y4):
                image = cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=thickness)
                image = cv2.line(image, (int(x2), int(y2)), (int(x3), int(y3)), color, thickness=thickness)
                image = cv2.line(image, (int(x3), int(y3)), (int(x4), int(y4)), color, thickness=thickness)
                image = cv2.line(image, (int(x4), int(y4)), (int(x1), int(y1)), color, thickness=thickness)
        except:
            image = cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=thickness)
            image = cv2.line(image, (int(x2), int(y2)), (int(x3), int(y3)), color, thickness=thickness)
            image = cv2.line(image, (int(x3), int(y3)), (int(x4), int(y4)), color, thickness=thickness)
            image = cv2.line(image, (int(x4), int(y4)), (int(x1), int(y1)), color, thickness=thickness)
        return image
def convert_format(box):
        xc = (box[:, 2] - box[:, 0]) / 2.0 + box[:, 0]
        yc = (box[:, 3] - box[:, 1]) / 2.0 + box[:, 1]
        w = (box[:, 2] - box[:, 0])
        h = (box[:, 3] - box[:, 1])
        angle = box[:, 4]
        return xc, yc, w, h, angle
def center2box(xywha):
    #convert centerX-centerY-width-height-angle format to 4-point polygon format
    x,y,width,height,angle=xywha
    rot_M = cv2.getRotationMatrix2D((x,y),angle,1)
    new_box = np.array([
        (x-width/2, y-height/2),
        (x+width/2, y-height/2),
        (x+width/2, y+height/2),
        (x-width/2, y+height/2)
    ],dtype=np.float32)
    bb_rotated = np.vstack((new_box.T,np.array((1,1,1,1))))
    bb_rotated = np.dot(rot_M,bb_rotated).astype(np.float32)
    return bb_rotated.T

def box2center(points):
    # Calculate the center point of the rectangle
    center_x = sum([p[0] for p in points]) / 4
    center_y = sum([p[1] for p in points]) / 4
    
    # Calculate the width and height of the rectangle
    width = math.sqrt((points[0][0] - points[1][0])**2 + (points[0][1] - points[1][1])**2)
    height = math.sqrt((points[1][0] - points[2][0])**2 + (points[1][1] - points[2][1])**2)
    
    # Calculate the angle of the rectangle
    angle = math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0])
    
    return center_x, center_y, width, height, -np.rad2deg(angle)