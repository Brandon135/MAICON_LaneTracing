import cv2
import numpy as np
import time

def detect_lines_and_intersections(image, rho, theta, threshold, min_line_length, max_line_gap, angle_threshold):
    crop_img = image[240:480, 150:490].copy()
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    if lines is None:
        return image, crop_img, [], None, 0
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(crop_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    intersections = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            x1, y1, x2, y2 = lines[i][0]
            x3, y3, x4, y4 = lines[j][0]
            
            angle1 = np.arctan2(y2 - y1, x2 - x1)
            angle2 = np.arctan2(y4 - y3, x4 - x3)
            angle_diff = np.abs(angle1 - angle2) * 180 / np.pi
            
            if angle_diff > angle_threshold and angle_diff < (180 - angle_threshold):
                det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if det != 0:
                    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / det
                    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / det
                    intersections.append((int(px), int(py)))
    
    for point in intersections:
        cv2.circle(crop_img, point, 5, (0, 0, 255), -1)
    
    image[240:480, 150:490] = crop_img
    
    # 선 중앙점 계산
    if len(lines) > 0:
        center_line = np.mean(lines, axis=0)[0]
        center_x = int((center_line[0] + center_line[2]) / 2)
        center_y = int((center_line[1] + center_line[3]) / 2)
        cv2.circle(crop_img, (center_x, center_y), 5, (255, 0, 0), -1)
    else:
        center_x = None
    
    # 교차로까지의 거리 추정 (가장 아래에 있는 교차점의 y 좌표를 사용)
    intersection_distance = 0
    if intersections:
        lowest_intersection = max(intersections, key=lambda p: p[1])
        intersection_distance = crop_img.shape[0] - lowest_intersection[1]
    
    return image, crop_img, intersections, center_x, intersection_distance


def is_at_intersection(intersection_distance, threshold_min=80, threshold_max=100):
    return threshold_min <= intersection_distance <= threshold_max

def control_robot(center_x, frame_width, intersection_distance, turn_count, turning):
    if turning:
        return "LEFT_TURN"
    
    center_threshold = 20
    frame_center = frame_width // 2
    
    if intersection_distance == 0:
        return "FORWARD"
    elif is_at_intersection(intersection_distance):
        return "APPROACH_INTERSECTION"
    elif abs(center_x - frame_center) < center_threshold:
        return "FORWARD"
    elif center_x < frame_center:
        return "LEFT"
    else:
        return "RIGHT"

def main():
    camera = cv2.VideoCapture(0)
    camera.set(3, 640)
    camera.set(4, 480)
    
    rho = 1
    theta = np.pi / 180
    threshold = 75
    min_line_length = 50
    max_line_gap = 10
    angle_threshold = 30
    
    turn_count = 0
    turning = False
    turn_start_time = None
    turn_duration = 2  # 좌회전 지속 시간 (초)
    
    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            break
        
        result, crop_result, intersections, center_x, intersection_distance = detect_lines_and_intersections(
            frame, rho, theta, threshold, min_line_length, max_line_gap, angle_threshold
        )
        
        action = control_robot(center_x, crop_result.shape[1], intersection_distance, turn_count, turning)
        
        if action == "APPROACH_INTERSECTION":
            turn_count += 1
            if turn_count >= 4:
                turning = True
                turn_start_time = time.time()
        else:
            turn_count = 0
        
        if turning:
            if time.time() - turn_start_time > turn_duration:
                turning = False
                turn_count = 0
        
        cv2.putText(result, f"Action: {action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result, f"Intersection Distance: {intersection_distance}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result, f"Turn Count: {turn_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Robot View', result)
        cv2.imshow('Processed View', crop_result)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+'):
            threshold += 5
        elif key == ord('-'):
            threshold = max(5, threshold - 5)
        
        print(f"Current threshold: {threshold}, Action: {action}, Intersection Distance: {intersection_distance}, Turn Count: {turn_count}")
    
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
