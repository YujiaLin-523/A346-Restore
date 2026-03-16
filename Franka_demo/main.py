#!/usr/bin/env python3
import sys
import rospy

# 调整 Import 顺序：先导入 robot/frankapy，再导入 camera/cv2
# 这有助于避免某些系统上与 OpenCV 相关的初始化死锁问题
from src.utils import load_config
from src.robot import RobotController
from src.camera import CameraInterface
from src.calibration import HandEyeCalibration
from src.grasping import GraspDemo

def main():
    print("Initialize Configuration...")
    config = load_config()
    
    print("Initialize Robot...")
    # Initialize robot first so any ROS node init inside the driver
    # happens before camera subscribers are created.
    robot = RobotController(config)

    print("Initialize Camera...")
    camera = CameraInterface(config)
    camera.start()
    
    while not rospy.is_shutdown():
        print("\n==================================")
        print("   Franka Camera Tool Suite")
        print("==================================")
        print("1. Hand-Eye Calibration")
        print("2. Grasp Demo")
        print("0. Exit")
        
        choice = input("\nEnter choice: ").strip()
        
        if choice == '1':
            calib = HandEyeCalibration(camera, robot, config)
            calib.run()
        elif choice == '2':
            demo = GraspDemo(camera, robot, config)
            demo.run()
        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
