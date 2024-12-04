import carla
import math
import random
import time
import queue
import numpy as np
import cv2
import threading

client = carla.Client('localhost', 2000)
world  = client.get_world()
bp_lib = world.get_blueprint_library()

# Get the map spawn points
spawn_points = world.get_map().get_spawn_points()

actors = []

# spawn vehicle
#vehicle_bp =bp_lib.find('vehicle.lincoln.mkz_2020')
vehicle_bp =bp_lib.find('vehicle.tesla.model3')
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
if vehicle:
    actors.append(vehicle)

# spawn camera
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_init_trans = carla.Transform(carla.Location(x=2.5, z=1.5))
left_mirror_camera_transform = carla.Transform(carla.Location(x=-0.5, y=-1.0, z=1.0), carla.Rotation(pitch=0, yaw=180, roll=0))  # 왼쪽 사이드미러
right_mirror_camera_transform = carla.Transform(carla.Location(x=-0.5, y=1.0, z=1.0), carla.Rotation(pitch=0, yaw=180, roll=0))  # 오른쪽 사이드미러
rear_camera_transform = carla.Transform(carla.Location(x=-2.5, z=1.5), carla.Rotation(pitch=0, yaw=180))  # 후면
#카메라 추가
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
left_mirror_camera = world.spawn_actor(camera_bp, left_mirror_camera_transform, attach_to=vehicle)
right_mirror_camera = world.spawn_actor(camera_bp, right_mirror_camera_transform, attach_to=vehicle)
rear_camera = world.spawn_actor(camera_bp, rear_camera_transform, attach_to=vehicle)

#imu sensor 추가
imu_bp = bp_lib.find('sensor.other.imu')
#imu센서 위치 설정 (차량 중심부에 장착)
imu_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=1.0))
#imu 센서 차량에 부착
imu_sensor = world.spawn_actor(imu_bp, imu_transform, attach_to=vehicle)

#라이다 센서 추가
lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('range', '150') #라이다 범위 (미터)
lidar_bp.set_attribute('rotation_frequency', '20') #회전 속도 (Hz)
lidar_bp.set_attribute('channels', '32') #채널수
lidar_bp.set_attribute('points_per_second', '640000') #초당 포인트 수
lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5)) #차량 위에 설치
lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

#레이더 센서 추가
rander_bp = bp_lib.find('sensor.other.radar')
#레이더 속성 설정
rander_bp.set_attribute('horizontal_fov', '35') #수평 시야각 
rander_bp.set_attribute('vertical_fov', '20') #수직 시야각
rander_bp.set_attribute('range', '50') #레이더 범위
#레이더 센서의 위치와 방향 설정
radar_transform = carla.Transform(carla.Location(x=2.5, z=1.0))  # 차량 전방에 부착
radar_sensor = world.spawn_actor(rander_bp, radar_transform, attach_to=vehicle)

# LiDAR 데이터 큐
lidar_queue = queue.Queue()

#오토파일럿 세팅
vehicle.set_autopilot(True)

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True # Enables synchronous mode
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Create a queue to store and retrieve the sensor data
image_queue = queue.Queue()
camera.listen(image_queue.put)

left_image_queue = queue.Queue()
left_mirror_camera.listen(left_image_queue.put)
right_image_queue = queue.Queue()
right_mirror_camera.listen(right_image_queue.put)
rear_image_queue = queue.Queue()
rear_camera.listen(rear_image_queue.put)

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]

#imu 센서 수집
def process_imu_data(imu_data):
    #가속도 정보 (m/s^2)
    accel = imu_data.accelerometer
    #각속도 정보 (rad/s)
    gyro = imu_data.gyroscope
    #방향 정보 (롤, 피치, 요)
    compass = imu_data.compass

    print(f"Acceleration : {accel}")
    print(f"Gyroscope : {gyro}")
    print(f"Compass : {compass}")

#라이다 데이터 처리 함수
def process_lidar_data(lidar_data):
    points = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
    points = np.reshape(points, (int(len(points)/ 4), 4)) # (x, y, z, intensity)
    return points[:, :3] #(x, y, z) 좌표만 변환

# LiDAR 데이터 시각화 함수 (최적화 적용)
def visualize_lidar_with_opencv(points, img_size=500):
    lidar_2d = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # 포인트를 이미지 좌표로 변환 (벡터화 처리)
    mask = (-50 < points[:, 0]) & (points[:, 0] < 50) & (-50 < points[:, 1]) & (points[:, 1] < 50)
    points = points[mask]  # 범위 필터링
    u = ((points[:, 0] + 50) / 100 * img_size).astype(np.int32)
    v = ((points[:, 1] + 50) / 100 * img_size).astype(np.int32)

    # 이미지 좌표에 포인트 표시
    lidar_2d[v, u] = (255, 255, 255)

    cv2.imshow("LiDAR View", lidar_2d)
    cv2.waitKey(1)

# LiDAR 데이터 리스너
def lidar_callback(lidar_data):
    points = process_lidar_data(lidar_data)
    if not lidar_queue.full():
        lidar_queue.put(points)

#레이더 데이터 처리
def process_radar_data(radar_data):
    for detection in radar_data:
        #물체 정보 추출
        velocity = detection.velocity #상태 속도 (m/s)
        azimuth = detection.azimuth #수평 각도 (rad)
        altitude = detection.altitude #수직 각도 (rad)
        depth = detection.depth #거리 (m)
        print(f"Velocity : {velocity:.2f} m/s, Azimuth : {azimuth:.2f} rad, Altitude : {altitude:.2f} rad, Depth: {depth:.2f} m")
        
# Get the attributes from the camera
image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()
fov = camera_bp.get_attribute("fov").as_float()

# Calculate the camera projection matrix to project from 3D -> 2D
K = build_projection_matrix(image_w, image_h, fov)
K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

# Set up the set of bounding boxes from the level
# We filter for traffic lights and traffic signs
bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
bounding_box_set.extend(world.get_level_bbs(carla.CityObjectLabel.TrafficSigns))

# Remember the edge pairs
edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

for i in range(50):
    vehicle_bp = random.choice(bp_lib.filter('vehicle'))
    npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    if npc:
        actors.append(npc)
        npc.set_autopilot(True)

# IMU 데이터 수집 시작
imu_sensor.listen(process_imu_data)
#라이다 데이터 시작
lidar_sensor.listen(lidar_callback)
#레이더 데이터 수집 시작
radar_sensor.listen(process_radar_data)

# 데이터 수집 및 시각화 루프
def lidar_visualization_loop():
    while True:
        if not lidar_queue.empty():
            points = lidar_queue.get()
            visualize_lidar_with_opencv(points)

# 시각화 스레드 실행
visualization_thread = threading.Thread(target=lidar_visualization_loop, daemon=True)
visualization_thread.start()


try:
    while True:

        # Retrieve and reshape the image
        world.tick(0.1)
        image = image_queue.get()
        left_image = left_image_queue.get()
        rigtht_image = right_image_queue.get()
        rear_image = rear_image_queue.get()

        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        left_img = np.reshape(np.copy(left_image.raw_data), (left_image.height, left_image.width, 4))
        right_img = np.reshape(np.copy(rigtht_image.raw_data), (rigtht_image.height, rigtht_image.width, 4))
        rear_img = np.reshape(np.copy(rear_image.raw_data), (rear_image.height, rear_image.width, 4))

        # Get the camera matrix 
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
        world_2_left_carmea = np.array(left_mirror_camera.get_transform().get_inverse_matrix())
        world_2_right_mirror_camera = np.array(right_mirror_camera.get_transform().get_inverse_matrix())
        world_2_rear_camera = np.array(rear_camera.get_transform().get_inverse_matrix())

        #출력
        top_row = cv2.hconcat([img, rear_img])
        bottom_row = cv2.hconcat([left_img, right_img])
        combined_image = cv2.vconcat([top_row, bottom_row])

        cv2.imshow("Multi-Camera View", combined_image)

        if cv2.waitKey(1) == ord('q'):
            break

finally:
    for actor in actors:
        if actor.is_alive:
            actor.destroy()

    #cv2 닫기
    cv2.destroyAllWindows()