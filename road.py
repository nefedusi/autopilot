import cv2
import numpy as np
import math

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=4):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is None or len(lines) == 0:
        return None, None
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    x_min = np.min(lines[:,:,0])
    x_max = np.max(lines[:,:,0])
    max_dist = (x_max - x_min)/4
    left_lines = []
    right_lines = []

    lines = lines[0]
    for line in lines:
        # print "HEY", line[0]
        if line[1] > line[3]:
            p1 = (line[0], line[1])
            line[0] = line[2]
            line[1] = line[3]
            line[2] = p1[0]
            line[3] = p1[1]
    draw_lines(line_img, [lines])
    return line_img, lines


def find_road_and_get_angle(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_black = cv2.inRange(gray_frame, (0, 0, 0), (255, 255, 120))
    # return mask_black
    kernel_size = 5
    gauss_black = mask_black
    #gauss_black = cv2.GaussianBlur(mask_black, (kernel_size, kernel_size), 0)
    imshape = gauss_black.shape
    lower_left = [0, imshape[0]]
    lower_right = [imshape[1], imshape[0]]
    top_left = [imshape[1]/2 - imshape[1]/2, imshape[0]/2 + 0]
    top_right = [imshape[1]/2 + imshape[1]/2, imshape[0]/2 + 0]
    # print lower_left, lower_right
    # print top_left, top_right
    vertices = [np.array([lower_left,top_left,top_right,lower_right], dtype=np.int32)]

    gauss_black = cv2.dilate(gauss_black, None, iterations = 7)
    gauss_black = cv2.erode(gauss_black, None, iterations = 7)
    gauss_black_roi = region_of_interest(gauss_black, vertices)
    #return gauss_black_roi, None
    # gauss_black_roi = cv2.bitwise_not(gauss_black_roi)
    indices = np.transpose(np.nonzero(gauss_black_roi))
    # print indices
    mass_center = np.average(indices, axis = 0).astype(np.int32)
    mass_center = np.roll(mass_center, 1)
    # print "MASS", mass_center
    low_threshold = 0
    high_threshold = 50
    #return gauss_black, None
    canny_edges = cv2.Canny(gauss_black, low_threshold, high_threshold)
    roi_image = region_of_interest(canny_edges, vertices)
    #return roi_image, None
    # return roi_image
    rho = 1
    theta = np.pi/180
    threshold = 10
    min_line_len = 30/4
    max_line_gap = 10

    line_image, lines = hough_lines(roi_image, rho, theta, threshold, min_line_len, max_line_gap)
   # print "lines", lines
    if lines is None:
        return None, None
    # print type(lines)
    #lines = lines[0]
    line_lengths = [math.sqrt((line[2] - line[0])*(line[2] - line[0]) + (line[3] - line[1])*(line[3] - line[1])) for line in lines]
    #print "weights", line_lengths
    average_line = np.average(lines, axis = 0, weights = line_lengths)
    average_line = average_line.astype(np.int32)
    #print "avg", average_line
    #average_line = average_line[0]
    average_line_bot = (average_line[0], average_line[1])
    average_line_dir = [average_line[2] - average_line[0], average_line[3] - average_line[1]]

    # average_line[0] -= average_line_bot[0]
    # average_line[1] -= average_line_bot[1]
    # average_line[2] -= average_line_bot[0]
    # average_line[3] -= average_line_bot[1]
    draw_lines(line_image, [[average_line]], color=[0, 255, 0])
    # average_point = [(average_line[0] + average_line[2])/2, (average_line[1] + average_line[3])/2]
    average_point = [average_line[0], average_line[1]]
    # average_point_center = list(average_point)
    # average_point_center[0] = imshape[1]/2
    average_point_center = mass_center
    average_point_dir = [average_point_center[0] - average_point[0], average_point_center[1] - average_point[1]]
    average_line[0] += average_point_dir[0]
    average_line[1] += average_point_dir[1]
    average_line[2] += average_point_dir[0]
    average_line[3] += average_point_dir[1]
    average_line_dir = [average_line[2] - average_line[0], average_line[3] - average_line[1]]
    mass_center_dir = [imshape[1]/2 - mass_center[0], imshape[0] - mass_center[1]]
    #print mass_center_dir, average_line_dir
    average_vector = [average_line_dir[0] + mass_center_dir[0]/2, average_line_dir[1] + mass_center_dir[1]/2]
    print "Vect", average_vector
    # average_vector = [average_point_dir[0] + average_line_dir[0], average_point_dir[1] + average_line_dir[1]]
    if average_vector[1] == 0:
        return frame, 0
    angle = math.atan(average_vector[0]/float(average_vector[1]))
    # print average_vector, angle
    angle = angle/(math.pi/2)
    # print "point", average_point_center
    cv2.line(line_image, (average_line[0], average_line[1]), (average_line[2], average_line[3]), color = (255, 255, 255), thickness = 4)
    # cv2.line(line_image, (imshape[1]/2, imshape[0]), (mass_center_dir[0], mass_center_dir[1]), color = (255, 255, 255), thickness = 4)
    
    # cv2.line(line_image, tuple(average_point), tuple(average_point_center), color = (255, 255, 255), thickness = 4)
    weightened_image = cv2.addWeighted(frame, 0.8, line_image, 1., 0)
    return weightened_image, angle
