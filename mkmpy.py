
import cv2
import numpy as np
import time
import asyncio
# 초기 HSV 범위 설정
lower_green = np.array([67, 23, 101])
upper_green = np.array([80, 255, 168])


def nothing(x):
    pass

def create_trackbars():
    cv2.namedWindow("HSV Trackbars")
    cv2.createTrackbar("L-H", "HSV Trackbars", lower_green[0], 179, nothing)
    cv2.createTrackbar("L-S", "HSV Trackbars", lower_green[1], 255, nothing)
    cv2.createTrackbar("L-V", "HSV Trackbars", lower_green[2], 255, nothing)
    cv2.createTrackbar("U-H", "HSV Trackbars", upper_green[0], 179, nothing)
    cv2.createTrackbar("U-S", "HSV Trackbars", upper_green[1], 255, nothing)
    cv2.createTrackbar("U-V", "HSV Trackbars", upper_green[2], 255, nothing)

def get_hsv_values():
    l_h = cv2.getTrackbarPos("L-H", "HSV Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "HSV Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "HSV Trackbars")
    u_h = cv2.getTrackbarPos("U-H", "HSV Trackbars")
    u_s = cv2.getTrackbarPos("U-S", "HSV Trackbars")
    u_v = cv2.getTrackbarPos("U-V", "HSV Trackbars")
    return np.array([l_h, l_s, l_v]), np.array([u_h, u_s, u_v])
#Lane, HSV Detection
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

    return image, crop_img, intersections, center_x, intersection_distance

async def detect_green_color(image, lower_green, upper_green, center_width=100, center_height=100):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    height, width = mask.shape
    center_x = width // 2
    center_y = height // 2
    start_x = center_x - center_width // 2
    start_y = center_y - center_height // 2
    end_x = start_x + center_width
    end_y = start_y + center_height
    center_area = mask[start_y:end_y, start_x:end_x]
    green_detected = np.sum(center_area) > 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_point = None
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            green_point = (cx, cy)
    return green_detected, green_point, mask

def set_motor_power(motor, value):
    # Placeholder function for setting motor power
    # Implement actual motor control logic here
    #print(f"Setting {motor} motor power to {value}")
    pass
def stop():
    # Placeholder function for stopping motors
    print("Stopping motors")

def turn_90_degrees(direction):
    turn_duration = 1.0  # Adjust this value based on your robot's turning speed
    turn_power = 100  # Full power for turning

    if direction.lower() == 'left':
        set_motor_power("MOTOR_LEFT", 0.3*turn_power)
        set_motor_power("MOTOR_RIGHT", turn_power)
    elif direction.lower() == 'right':
        set_motor_power("MOTOR_LEFT", turn_power)
        set_motor_power("MOTOR_RIGHT", 0.3*turn_power)
    else:
        print("Invalid direction. Use 'left' or 'right'.")
        return

    time.sleep(turn_duration)
    stop()
    
async def turn_findobj():
    await asyncio.sleep(1)
    return [5, 1, 10] ######################테스트값
    
def turn_clockwise(angle):
    td = 1.0
    turn_duration = angle / 90.0  # Assuming 1 second per 90 degrees as a starting point
    turn_power = 100  # Full power for turning

    set_motor_power("MOTOR_LEFT", turn_power)
    set_motor_power("MOTOR_RIGHT", -turn_power)

    time.sleep(td)
    stop()

def adjust_wheel_speed(center_x, frame_width):
    base_speed = 50
    max_speed_diff = 20
    if center_x is None:
        return base_speed, base_speed

    center_offset = center_x - (frame_width // 2)
    steering_angle = center_offset * (30 / (frame_width // 2))
    speed_diff = (steering_angle / 30) * max_speed_diff

    left_wheel_speed = base_speed - speed_diff
    right_wheel_speed = base_speed + speed_diff

    left_wheel_speed = max(0, min(100, left_wheel_speed))
    right_wheel_speed = max(0, min(100, right_wheel_speed))

    return left_wheel_speed, right_wheel_speed
import cv2
import numpy as np
import time

async def main():
    camera = cv2.VideoCapture(0)
    camera.set(3, 640)
    camera.set(4, 480)
    create_trackbars()

    rho = 1
    theta = np.pi / 180
    threshold = 80
    min_line_length = 50
    max_line_gap = 10
    angle_threshold = 30

    mission_count = 9
    cooldown_duration = 10
    green_detection_time = 1
    current_mission = 0
    is_mission_active = False
    cooldown_start_time = 0
    green_start_time = 0
    green_detection_count = 0
    green_detection_threshold = 50
    mission_completed = False

    while camera.isOpened() and current_mission < mission_count:
        ret, frame = camera.read()
        if not ret:
            break

        lower_green, upper_green = get_hsv_values()
        result, crop_result, intersections, center_x, intersection_distance = detect_lines_and_intersections(
            frame, rho, theta, threshold, min_line_length, max_line_gap, angle_threshold
        )
        left_speed, right_speed = adjust_wheel_speed(center_x, crop_result.shape[1])
        current_time = time.time()

        # 비동기로 녹색 감지 실행
        green_detection_task = asyncio.create_task(detect_green_color(frame, lower_green, upper_green))
        green_detected, green_point, green_mask = await green_detection_task
        if green_detected:
            green_detection_count += 1
        else:
            green_detection_count = 0

        if not is_mission_active and current_time - cooldown_start_time > cooldown_duration:
            if current_mission in [0, 2, 4, 7]:  # 미션 1, 3, 5, 8에 해당
                if len(intersections) >= 2 and green_detection_count >= green_detection_threshold:
                    current_mission += 1
                    print(f'미션 {current_mission} 시작')
                    is_mission_active = True
                    mission_completed = False
                    green_detection_count = 0
            elif green_detection_count >= green_detection_threshold:
                current_mission += 1
                print(f'미션 {current_mission} 시작')
                is_mission_active = True
                mission_completed = False
                green_detection_count = 0
            elif green_detected:
                if green_start_time == 0:
                    green_start_time = current_time
                elif current_time - green_start_time >= green_detection_time:
                    current_mission += 1
                    print(f'미션 {current_mission} 시작')
                    is_mission_active = True
                    mission_completed = False
                    green_start_time = 0
            else:
                green_start_time = 0

        # Swap motor power settings
        set_motor_power("MOTOR_RIGHT", int(left_speed * 2.55))  # Convert to PWM range 0-255
        set_motor_power("MOTOR_LEFT", int(right_speed * 2.55))  # Convert to PWM range 0-255

        # Mission execution logic
        if is_mission_active and not mission_completed:
            if current_mission in [1, 3, 8]:
                print("좌회전")
                turn_90_degrees('left')
                mission_completed = True

            elif current_mission == 2:
                print("정지 후 왼쪽으로 90도 회전")
                stop()
                turn_clockwise(-90)
                turn_angles = await turn_findobj()
                total_angle = sum(turn_angles)
                for angle in turn_angles:
                    turn_clockwise(angle)
                turn_clockwise(-(90-total_angle))
                mission_completed = True

            elif current_mission == 5:
                print("우회전")
                turn_90_degrees('right')
                mission_completed = True

            elif current_mission in [4, 6, 7]:
                print("정지 후 왼쪽으로 90도 회전")
                stop()
                turn_clockwise(-90)
                turn_angles = await turn_findobj()
                total_angle = sum(turn_angles)
                for angle in turn_angles:
                    turn_clockwise(angle)
                turn_clockwise(90-total_angle)
                mission_completed = True

            elif current_mission == 9:
                print("정지")
                stop()
                mission_completed = True

            if mission_completed:
                is_mission_active = False
                cooldown_start_time = current_time
                print(f'미션 {current_mission} 완료. 쿨다운 시작.')

        # 화면에 정보 표시
        cv2.putText(result, f"Mission: {current_mission}/{mission_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result, f"Left Speed: {left_speed:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result, f"Right Speed: {right_speed:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result, f"Intersections: {len(intersections)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if green_point:
            cv2.circle(result, green_point, 10, (0, 255, 0), -1)
            cv2.putText(result, f"Green: ({green_point[0]}, {green_point[1]})", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if not is_mission_active:
            cooldown_time = max(0, cooldown_duration - (current_time - cooldown_start_time))
            cv2.putText(result, f"Cooldown: {cooldown_time:.1f}s", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow('Robot View', result)
        cv2.imshow('Processed View', crop_result)
        cv2.imshow('Green Mask', green_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('모든 미션 완료.')

    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    asyncio.run(main())