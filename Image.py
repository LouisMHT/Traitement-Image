import numpy as np
import cv2 as cv
from collections import defaultdict

def distance(coord1, coord2):
    return ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** 0.5

def remove_outliers(coordinates, threshold, min_points=15):
    grouped_coordinates = defaultdict(list)

    for coord in coordinates:
        added = False
        for center_coord, similar_coords in grouped_coordinates.items():
            if any(distance(coord, similar_coord) <= threshold for similar_coord in similar_coords):
                grouped_coordinates[center_coord].append(coord)
                added = True
                break
        if not added:
            grouped_coordinates[tuple(coord)] = [coord]

    if len(grouped_coordinates) > 0:
        max_group = max(grouped_coordinates.values(), key=len)
        if len(max_group) >= min_points:
            cleaned_coordinates = max_group
        else:
            cleaned_coordinates = []
    else:
        cleaned_coordinates = []

    return cleaned_coordinates

def average_coordinates(coordinates):
    x_sum = 0
    y_sum = 0
    for coord in coordinates:
        x_sum += coord[0]
        y_sum += coord[1]
    x_avg = x_sum / len(coordinates)
    y_avg = y_sum / len(coordinates)
    return [x_avg, y_avg]

vid = cv.VideoCapture('Route30s.mp4')
img1 = cv.imread('stop2.png', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('panne.png', cv.IMREAD_GRAYSCALE)
img3 = cv.imread('panneau.png', cv.IMREAD_GRAYSCALE)
img4 = cv.imread('pass.png', cv.IMREAD_GRAYSCALE)
img5 = cv.imread('RP.png', cv.IMREAD_GRAYSCALE)

fps = vid.get(cv.CAP_PROP_FPS)

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
kp3, des3 = sift.detectAndCompute(img3, None)
kp4, des4 = sift.detectAndCompute(img4, None)
kp5, des5 = sift.detectAndCompute(img5, None)
bf = cv.BFMatcher()

last_position = None

while(1):
    ret, frame = vid.read()
    if not ret:
        break

    framegray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    height, width = framegray.shape
    right_third_frame = framegray[:, (2*width)//3:]

    kp_frame, des_frame = sift.detectAndCompute(right_third_frame, None)

    matched_pts1 = []
    matched_pts2 = []
    matched_pts3 = []
    matched_pts4 = []
    matched_pts5 = []

    if des_frame is not None:
        matches1 = bf.knnMatch(des1, des_frame, k=2)
        good1 = []
        for m, n in matches1:
            if m.distance < 0.75 * n.distance:
                good1.append([m])
        matched_pts1 = np.float32([kp_frame[m[0].trainIdx].pt for m in good1]).reshape(-1, 2)

        matches2 = bf.knnMatch(des2, des_frame, k=2)
        good2 = []
        for m, n in matches2:
            if m.distance < 0.75 * n.distance:
                good2.append([m])
        matched_pts2 = np.float32([kp_frame[m[0].trainIdx].pt for m in good2]).reshape(-1, 2)

        matches3 = bf.knnMatch(des3, des_frame, k=2)
        good3 = []
        for m, n in matches3:
            if m.distance < 0.75 * n.distance:
                good3.append([m])
        matched_pts3 = np.float32([kp_frame[m[0].trainIdx].pt for m in good3]).reshape(-1, 2)

        matches4 = bf.knnMatch(des4, des_frame, k=2)
        good4 = []
        for m, n in matches4:
            if m.distance < 0.75 * n.distance:
                good4.append([m])
        matched_pts4 = np.float32([kp_frame[m[0].trainIdx].pt for m in good4]).reshape(-1, 2)

        matches5 = bf.knnMatch(des5, des_frame, k=2)
        good5 = []
        for m, n in matches5:
            if m.distance < 0.75 * n.distance:
                good5.append([m])
        matched_pts5 = np.float32([kp_frame[m[0].trainIdx].pt for m in good5]).reshape(-1, 2)

    matched_pts1[:, 0] += (2 * width) // 3
    matched_pts2[:, 0] += (2 * width) // 3
    matched_pts3[:, 0] += (2 * width) // 3
    matched_pts4[:, 0] += (2 * width) // 3
    matched_pts5[:, 0] += (2 * width) // 3

    cleaned_coordinates1 = remove_outliers(matched_pts1, threshold=50)
    cleaned_coordinates2 = remove_outliers(matched_pts2, threshold=50)
    cleaned_coordinates3 = remove_outliers(matched_pts3, threshold=50)
    cleaned_coordinates4 = remove_outliers(matched_pts4, threshold=50)
    cleaned_coordinates5 = remove_outliers(matched_pts5, threshold=50)

    if len(cleaned_coordinates1) > 0 or len(cleaned_coordinates2) or len(cleaned_coordinates3) or len(cleaned_coordinates4) or len(cleaned_coordinates5) > 0:
        if len(cleaned_coordinates1) > len(cleaned_coordinates2) and len(cleaned_coordinates1) > len(cleaned_coordinates3) and len(cleaned_coordinates1) > len(cleaned_coordinates4) and len(cleaned_coordinates1) > len(cleaned_coordinates5):
            object_type = "STOP"
            cleaned_coordinates = cleaned_coordinates1
        if len(cleaned_coordinates2) > len(cleaned_coordinates1) and len(cleaned_coordinates2) > len(cleaned_coordinates3) and len(cleaned_coordinates2) > len(cleaned_coordinates4) and len(cleaned_coordinates2) > len(cleaned_coordinates5):
            object_type = "30"
            cleaned_coordinates = cleaned_coordinates2
        if len(cleaned_coordinates3) > len(cleaned_coordinates2) and len(cleaned_coordinates3) > len(cleaned_coordinates1) and len(cleaned_coordinates3) > len(cleaned_coordinates4) and len(cleaned_coordinates3) > len(cleaned_coordinates5):
            object_type = "50"
            cleaned_coordinates = cleaned_coordinates3
        if len(cleaned_coordinates4) > len(cleaned_coordinates2) and len(cleaned_coordinates4) > len(cleaned_coordinates3) and len(cleaned_coordinates4) > len(cleaned_coordinates1) and len(cleaned_coordinates4) > len(cleaned_coordinates5):
            object_type = "PIETON"
            cleaned_coordinates = cleaned_coordinates4
        if len(cleaned_coordinates5) > len(cleaned_coordinates2) and len(cleaned_coordinates5) > len(cleaned_coordinates3) and len(cleaned_coordinates5) > len(cleaned_coordinates4) and len(cleaned_coordinates5) > len(cleaned_coordinates1):
            object_type = "RP"
            cleaned_coordinates = cleaned_coordinates5

        average_coord = average_coordinates(cleaned_coordinates)

        if last_position is not None:
            distances = [distance(average_coord, pos) for pos in cleaned_coordinates]
            closest_index = np.argmin(distances)
            closest_position = cleaned_coordinates[closest_index]
            if distance(closest_position, last_position) < 50:
                average_coord = closest_position

        last_position = average_coord

        mean_x = average_coord[0]
        mean_y = average_coord[1]
        object_size = 80
        cv.rectangle(frame, (int(mean_x - object_size / 2), int(mean_y - object_size / 2)),
                     (int(mean_x + object_size / 2), int(mean_y + object_size / 2)), (0, 255, 0), 2)
        cv.putText(frame, object_type, (int(mean_x) - 20, int(mean_y) - 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv.imshow('frame', frame)

    if cv.waitKey(int(1000/fps)) == ord('b'):
        break

vid.release()
cv.destroyAllWindows()
