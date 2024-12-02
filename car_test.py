from __future__ import print_function

import glob
import os
import sys

try:
    sys.path.append(glob.glob('carla/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import random
import time 

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_f
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_o
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_t
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

def main():
    # CARLA 서버 연결
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)  # 서버 응답 시간 설정
    world = client.get_world()

    # 차량 블루프린트 로드
    blueprint_library = world.get_blueprint_library()

    # 맵의 모든 도로 위 Waypoint 가져오기
    map = world.get_map()
    waypoints = map.generate_waypoints(distance=2.0)  # 2m 간격으로 도로 위 Waypoint 생성

    # 차량 리스트 초기화
    vehicles = []

    try:
        # 차량 생성 및 초기 스폰 (도로 위 Waypoint 사용)
        print("Spawning vehicles on road...")
        for i in range(50):  # 총 20대의 차량 생성
            attempts = 0
            while attempts < 5:  # 최대 5번 스폰 시도
                try:
                    # 차량 블루프린트 무작위 선택
                    vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
                    
                    # 도로 위 Waypoint에서 무작위 위치 선택
                    spawn_waypoint = random.choice(waypoints)
                    spawn_point = spawn_waypoint.transform

                    # 차량 생성
                    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

                    if vehicle is not None:  # 스폰 성공 시
                        vehicle.set_autopilot(True)  # 자율주행 모드 활성화
                        vehicles.append(vehicle)
                        print(f"Vehicle {vehicle.id} spawned at {spawn_point.location}.")
                        break
                except Exception as e:
                    print(f"Spawn attempt {attempts+1} failed: {e}")
                attempts += 1
            else:
                print("Failed to spawn a vehicle after 5 attempts.")

            time.sleep(0.2)  # 각 차량 스폰 간격

        print("Vehicles successfully spawned on road.")

        # 지속적으로 차량을 운행
        while True:
            for vehicle in vehicles:
                # 차량의 현재 위치 가져오기
                location = vehicle.get_location()

                # 차량이 맵 경계를 벗어났는지 확인
                if not map.get_waypoint(location, project_to_road=True):
                    print(f"Vehicle {vehicle.id} is off-road. Respawning...")
                    spawn_waypoint = random.choice(waypoints)
                    spawn_point = spawn_waypoint.transform
                    vehicle.set_transform(spawn_point)  # 새로운 위치로 이동

                '''
                # 차량 속도를 주기적으로 조정
                if random.random() < 0.2:  # 20% 확률로 속도 변경
                    new_speed = random.uniform(10, 30)  # 속도 범위: 10~30 m/s
                    vehicle.set_target_velocity(carla.Vector3D(new_speed, 0, 0))
                '''

            time.sleep(0.5)  # 루프 간 대기 시간

    finally:
        # 종료 시 모든 차량 제거
        print("Cleaning up vehicles...")
        for vehicle in vehicles:
            vehicle.destroy()

if __name__ == '__main__':
    main()