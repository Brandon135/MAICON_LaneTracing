import cv2
import numpy as np
import time

def detect_color_spot(crop_img):
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if green_contours:
        largest_contour = max(green_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > 100:
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return True, (cx, cy), area
    
    return False, None, 0

def detect_lines_and_intersections(image, rho, theta, threshold, min_line_length, max_line_gap, angle_threshold):
    crop_img = image[240:480, 150:490].copy()
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    
    color_detected, color_position, area = detect_color_spot(crop_img)
    green_distance = 0
    
    if color_detected:
        green_distance = crop_img.shape[0] - color_position[1]
        cv2.circle(crop_img, color_position, 5, (0, 255, 0), -1)
    
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    if lines is None:
        return image, crop_img, [], None, 0, green_distance, color_detected, color_position, area
    
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
    
    if len(lines) > 0:
        center_line = np.mean(lines, axis=0)[0]
        center_x = int((center_line[0] + center_line[2]) / 2)
        center_y = int((center_line[1] + center_line[3]) / 2)
        cv2.circle(crop_img, (center_x, center_y), 5, (255, 0, 0), -1)
    else:
        center_x = None
    
    intersection_distance = 0
    if intersections:
        lowest_intersection = max(intersections, key=lambda p: p[1])
        intersection_distance = crop_img.shape[0] - lowest_intersection[1]
    
    return image, crop_img, intersections, center_x, intersection_distance, green_distance, color_detected, color_position, area

def is_at_intersection(intersection_distance, threshold_min=60, threshold_max=100):
    return threshold_min <= intersection_distance <= threshold_max

def steer_to_center(center_x, frame_width):
    if center_x is None:
        return 0
    
    frame_center = frame_width // 2
    error = frame_center - center_x
    max_error = frame_width // 4
    
    normalized_error = max(min(error / max_error, 1), -1)
    steering_angle = normalized_error * 30
    
    return steering_angle

def align_with_marker(color_position, frame_width, green_distance):
    marker_x, marker_y = color_position
    frame_center = frame_width // 2
    
    x_error = frame_center - marker_x
    y_error = green_distance
    
    if y_error > 50:
        return "FORWARD", 0
    elif abs(x_error) > 10:
        if x_error > 0:
            return "LEFT", 15
        else:
            return "RIGHT", -15
    else:
        return "STOP", 0

def steer_to_center(center_x, frame_width):
    if center_x is None:
        return 0
    
    frame_center = frame_width // 2
    error = frame_center - center_x
    max_error = frame_width // 4
    
    # 더 부드러운 조향을 위해 비례 제어 적용
    kp = 0.5  # 비례 제어 게인
    normalized_error = max(min(error / max_error, 1), -1)
    steering_angle = normalized_error * 30 * kp
    
    return steering_angle

def control_robot(center_x, frame_width, intersection_distance, turn_count, turning, mission_in_progress, color_detected, color_position, green_distance):
    if color_detected and not mission_in_progress:
        return align_with_marker(color_position, frame_width, green_distance)
        
    if mission_in_progress:
        return "STOP", 0
        
    if turning:
        return "LEFT_TURN", 30
    
    if center_x is None:
        return "STOP", 0
    
    # 중앙 정렬을 위한 임계값 축소
    center_threshold = 15  # 20에서 10으로 변경
    frame_center = frame_width // 2
    
    # 더 정밀한 방향 제어
    if intersection_distance == 0:
        action = "FORWARD"
    elif is_at_intersection(intersection_distance):
        action = "APPROACH_INTERSECTION"
    elif abs(center_x - frame_center) < center_threshold:
        action = "FORWARD"
    elif center_x < frame_center:
        action = "LEFT"
    else:
        action = "RIGHT"
    
    steering_angle = steer_to_center(center_x, frame_width)
    return action, steering_angle
def draw_wheels(result, steering_angle, action):
    # 이미지 하단에 바퀴 표시 위치 설정
    h, w = result.shape[:2]
    wheel_y = h - 50
    left_wheel_x = 50
    right_wheel_x = w - 50
    
    # 바퀴 크기 설정
    wheel_width = 40
    wheel_height = 20
    
    # 스티어링 각도에 따른 바퀴 회전 표시
    left_angle = steering_angle
    right_angle = steering_angle
    
    # 특별한 동작에 따른 바퀴 회전 조정
    if action == "LEFT_TURN":
        left_angle = -30
        right_angle = 30
    elif action == "STOP":
        left_angle = 0
        right_angle = 0
    
    # 왼쪽 바퀴 그리기
    left_pts = np.array([
        [left_wheel_x - wheel_width//2, wheel_y],
        [left_wheel_x + wheel_width//2, wheel_y],
        [left_wheel_x + wheel_width//2, wheel_y + wheel_height],
        [left_wheel_x - wheel_width//2, wheel_y + wheel_height]
    ], np.int32)
    
    # 오른쪽 바퀴 그리기
    right_pts = np.array([
        [right_wheel_x - wheel_width//2, wheel_y],
        [right_wheel_x + wheel_width//2, wheel_y],
        [right_wheel_x + wheel_width//2, wheel_y + wheel_height],
        [right_wheel_x - wheel_width//2, wheel_y + wheel_height]
    ], np.int32)
    
    # 바퀴 회전 표시를 위한 선 그리기
    cv2.line(result, 
             (left_wheel_x, wheel_y + wheel_height//2),
             (left_wheel_x + int(wheel_width//2 * np.sin(np.radians(left_angle))),
              wheel_y + wheel_height//2 - int(wheel_width//2 * np.cos(np.radians(left_angle)))),
             (0, 0, 255), 2)
             
    cv2.line(result,
             (right_wheel_x, wheel_y + wheel_height//2),
             (right_wheel_x + int(wheel_width//2 * np.sin(np.radians(right_angle))),
              wheel_y + wheel_height//2 - int(wheel_width//2 * np.cos(np.radians(right_angle)))),
             (0, 0, 255), 2)
    
    # 바퀴 외곽선 그리기
    cv2.polylines(result, [left_pts], True, (0, 255, 0), 2)
    cv2.polylines(result, [right_pts], True, (0, 255, 0), 2)
    
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
    turn_duration = 2
    
    # 미션 상태 관리를 위한 변수들
    mission_state = {
        'in_progress': False,
        'completed': False,
        'start_time': None,
        'cooldown': False,
        'cooldown_start': None,
        'mission_count': 0
    }
    
    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            break
        
        result, crop_result, intersections, center_x, intersection_distance, green_distance, color_detected, color_position, area = detect_lines_and_intersections(
            frame, rho, theta, threshold, min_line_length, max_line_gap, angle_threshold
        )
        
        current_time = time.time()
        
        # 미션 진행 중일 때
        if mission_state['in_progress']:
            if current_time - mission_state['start_time'] >= 5:
                mission_state['in_progress'] = False
                mission_state['cooldown'] = True
                mission_state['cooldown_start'] = current_time
                mission_state['mission_count'] += 1
                print(f"미션 {mission_state['mission_count']} 완료!")
        
        # 쿨다운 체크 (미션 완료 후 3초 대기)
        if mission_state['cooldown']:
            if current_time - mission_state['cooldown_start'] >= 10:
                mission_state['cooldown'] = False
                print("다음 미션 준비 완료")
        
        # 새로운 미션 시작
        if (color_detected and not mission_state['in_progress'] and 
            not mission_state['cooldown'] and action == "STOP"):
            mission_state['in_progress'] = True
            mission_state['start_time'] = current_time
            print(f"미션 {mission_state['mission_count'] + 1} 시작")
        
        action, steering_angle = control_robot(
            center_x, crop_result.shape[1], intersection_distance,
            turn_count, turning, mission_state['in_progress'],
            color_detected, color_position, green_distance
        )
        
        # 화면 표시
        cv2.putText(result, f"Action: {action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result, f"Green Distance: {green_distance}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result, f"Mission Count: {mission_state['mission_count']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if mission_state['in_progress']:
            remaining_time = 5 - (current_time - mission_state['start_time'])
            cv2.putText(result, f"MISSION {mission_state['mission_count'] + 1} IN PROGRESS: {remaining_time:.1f}s", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif mission_state['cooldown']:
            cooldown_remaining = 10 - (current_time - mission_state['cooldown_start'])
            cv2.putText(result, f"COOLDOWN: {cooldown_remaining:.1f}s", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
        
        draw_wheels(result, steering_angle, action)
        
        # 나머지 화면 표시 코드
        cv2.imshow('Robot View', result)
        cv2.imshow('Processed View', crop_result)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            mission_state = {
                'in_progress': False,
                'completed': False,
                'start_time': None,
                'cooldown': False,
                'cooldown_start': None,
                'mission_count': 0
            }
        
        print(f"Action: {action}, Green Distance: {green_distance}, "
              f"Mission Count: {mission_state['mission_count']}")
    
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()