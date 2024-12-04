import carla
import math
import random
import time
import queue
import numpy as np
import cv2

client = carla.Client('localhost', 2000)
world  = client.get_world()
bp_lib = world.get_blueprint_library()

# Get the map spawn points
spawn_points = world.get_map().get_spawn_points()

# spawn vehicle
#vehicle_bp =bp_lib.find('vehicle.lincoln.mkz_2020')
vehicle_bp =bp_lib.find('vehicle.tesla.model3')
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

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
        npc.set_autopilot(True)

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
cv2.destroyAllWindows()