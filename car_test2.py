import carla
import random
import time
import queue

client = carla.Client('localhost', 2000)
world = client.get_world()
settings = world.get_settings()

world.apply_settings(settings)

spawn_points = world.get_map().get_spawn_points()