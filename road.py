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

def draw_lines(img, lines, color=[255, 0, 0], thickness=4, dir=False):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            if dir is True:
                cv2.circle(img, (x2, y2), 2, color=[0, 0, 255], thickness = 2)


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
    iw = img.shape[1]/2
    # lines = lines.reshape((lines.shape[0], lines.shape[2]))
    # lines = lines[0]
    for line in lines:
        # print "HEY", line[0]
        if abs(line[1] - line[3]) > abs(line[2] - line[0]):
            if line[1] > line[3]:
                p1 = (line[0], line[1])
                line[0] = line[2]
                line[1] = line[3]
                line[2] = p1[0]
                line[3] = p1[1]
        else:
            if (line[0] < iw or line[2] < iw) and line[2] > line[0]:
                p1 = (line[0], line[1])
                line[0] = line[2]
                line[1] = line[3]
                line[2] = p1[0]
                line[3] = p1[1]
            elif (line[0] > iw or line[2] > iw) and line[0] > line[2]:
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
    top_left = [imshape[1]/2 - imshape[1]/4, imshape[0]/2 + imshape[0]/15]
    top_right = [imshape[1]/2 + imshape[1]/4, imshape[0]/2 + imshape[0]/15]
    # print lower_left, lower_right
    # print top_left, top_right
    vertices = [np.array([lower_left,top_left,top_right,lower_right], dtype=np.int32)]

    gauss_black = cv2.dilate(gauss_black, None, iterations = 7)
    gauss_black = cv2.erode(gauss_black, None, iterations = 7)
    gauss_black_roi = region_of_interest(gauss_black, vertices)
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
    # return roi_image, None, None
    #return roi_image, None
    # return roi_image
    rho = 1
    theta = np.pi/180
    threshold = 10
    min_line_len = 30/4
    max_line_gap = 10
    #rho = 1
    #theta = np.pi/180
    #threshold = 10
    #min_line_len = 4
    #max_line_gap = 5

    line_image, lines = hough_lines(roi_image, rho, theta, threshold, min_line_len, max_line_gap)
    print "lines", lines.shape
    if lines is None:
        return None, None, lines
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
    # print "Vect", average_vector
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
    return weightened_image, angle, lines


def getDistancePoints(p1, p2):
    return math.sqrt((p2[0] - p1[0])*(p2[0] - p1[0]) + (p2[1] - p1[1])*(p2[1] - p1[1]))


def getDistanceLines(line1, line2):
    p11 = (line1[0], line1[1])
    p12 = (line1[2], line1[3])
    p21 = (line2[0], line2[1])
    p22 = (line2[2], line2[3])
    d1 = getDistancePoints(p11, p21)
    d2 = getDistancePoints(p11, p22)
    d3 = getDistancePoints(p12, p21)
    d4 = getDistancePoints(p12, p22)
    return min((d1, d2, d3, d4))


def swapVector(v):
    v = v[2], v[3], v[0], v[1]
    return v


def findCrossings(frame, lines):
    print("fincCrossing", lines)
    clusters = []
    lines = list(lines)
    while len(lines) > 0:
        line = lines.pop()
        if len(clusters) == 0:
            clusters.append([line])
        lcluster = None
        for cluster in clusters:
            for cline in cluster:
                dist = getDistanceLines(line, cline)
                if dist < 20:
                    lcluster = cluster
                    break
            if lcluster is not None:
                break
        if lcluster is not None:
            lcluster.append(line)
        else:
            clusters.append([line])
    cluster_averages = [np.average(cluster, axis = 0).astype(int) for cluster in clusters]
    average = np.average(cluster_averages, axis = 0)
    average = ((average[0] + average[2])/2, (average[1] + average[3])/2)
    print("Total average", average)
    print(cluster_averages)
    draw_lines(frame, [cluster_averages], color=[0, 0, 255], dir = True)
    directions = []
    for i in range(len(cluster_averages) - 1):
        avg1 = cluster_averages[i]
        dir1 =  (avg1[2] - avg1[0], avg1[3] - avg1[1]) 
        for j in range(i + 1, len(cluster_averages)):
            avg2 = cluster_averages[j]
            avg_avg = (int((avg1[0] + avg2[0])/2), int((avg1[1] + avg2[1])/2), int((avg1[2] + avg2[2])/2), int((avg1[3] + avg2[3])/2))
            avg1s, avg2s = avg1, avg2
            if avg_avg[1] + 10 < average[1] and avg_avg[3] > avg_avg[1]:
                avg1s = swapVector(avg1)
                avg2s = swapVector(avg2)
                avg_avg = swapVector(avg_avg)

            dir1 =  (avg1s[2] - avg1s[0], avg1s[3] - avg1s[1]) 
            dir2 =  (avg2s[2] - avg2s[0], avg2s[3] - avg2s[1])
            dir_avg = (avg_avg[2] - avg_avg[0], avg_avg[3] - avg_avg[1])
            ang1 = math.atan(dir1[1]/float(dir1[0]))
            ang2 = math.atan(dir2[1]/float(dir2[0]))
            ang_avg = math.atan(dir_avg[1]/float(dir_avg[0]))

            # cv2.circle(frame, (avg_avg[2], avg_avg[3]), 2, color=[0, 0, 255], thickness = 2)
            print(dir_avg, dir1, dir2)
            print(ang_avg, ang1, ang2)
            if abs(ang_avg - ang1) < math.pi/4 and abs(ang_avg - ang2) < math.pi/4 and (ang2 - ang1) < math.pi/4:
                # print "WTF", avg_avg
                # avg_avg = swapVector(avg_avg)
                draw_lines(frame, [[avg_avg]], color=[0, 255, 255], dir = True)
                # print "WTF", avg_avg
                directions.append(avg_avg)
            else:
                print(abs(ang_avg - ang1), abs(ang_avg - ang2))
    print("Directions", directions)
    # for avg in cluster_a/verages:
        # cv2.line()
