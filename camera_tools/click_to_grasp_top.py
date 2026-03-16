#!/usr/bin/env python3
import os, re, time
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from frankapy import FrankaArm
from autolab_core import RigidTransform

# ================= 你只需要改这些 =================
IMG_TOPIC   = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/aligned_depth_to_color/image_raw"   # 推荐：对齐后的深度
INFO_TOPIC  = "/camera/color/camera_info_calib"            # 用你标定后的；没有就改成 /camera/color/camera_info
T_TOOL_CAM_TXT = os.path.expanduser("~/camera_tools/T_tool_camera.txt")  # 你已经生成的外参

DEPTH_SCALE = 0.001  # 16UC1 通常是 mm -> m（RealSense ROS常见）
SAFE_Z_ABOVE = 0.12  # 到目标上方的安全高度（m）
GRASP_DROP_FROM_TOP = 0.03  # 从你点的“顶端”再向下多少去夹（m）
LIFT_AFTER_GRASP = 0.15     # 抓住后抬起（m）

MOVE_TIME = 3.0             # 每段动作时间（s）——故意慢
SLEEP_AFTER = 0.8           # 每段动作后停一下，确保真停稳
SPEED = 0.03                # 整体速度（越小越稳）
# ================================================

bridge = CvBridge()
g_rgb = None
g_depth = None
g_K = None
g_D = None

clicked_table_uv = None
clicked_obj_uv = None

def load_T_4x4(path: str) -> np.ndarray:
    txt = open(path, "r").read()
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+", txt)]
    if len(nums) < 16:
        raise RuntimeError(f"Cannot parse 4x4 from {path}, got {len(nums)} numbers.")
    T = np.array(nums[:16], dtype=np.float64).reshape(4, 4)
    return T

def rgb_cb(msg: Image):
    global g_rgb
    g_rgb = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

def depth_cb(msg: Image):
    global g_depth
    # 深度通常是 16UC1
    g_depth = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

def info_cb(msg: CameraInfo):
    global g_K, g_D
    g_K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
    g_D = np.array(msg.D, dtype=np.float64).reshape(-1)

def undistort_pixel_to_norm(u, v, K, D):
    """
    把像素(u,v) -> 归一化相机坐标 (x_n, y_n)（考虑畸变）
    """
    pts = np.array([[[u, v]]], dtype=np.float64)  # 1x1x2
    if D is None or np.allclose(D, 0):
        x = (u - K[0,2]) / K[0,0]
        y = (v - K[1,2]) / K[1,1]
        return x, y
    # undistortPoints 输出默认是归一化坐标
    und = cv2.undistortPoints(pts, K, D)  # 1x1x2 (norm)
    x, y = und[0,0,0], und[0,0,1]
    return x, y

def pixel_to_Pc(u, v, depth_img, K, D, depth_scale):
    """
    从像素(u,v) + 深度图 -> Pc (camera optical frame) [m]
    """
    if depth_img is None:
        return None, "no_depth"
    if v < 0 or v >= depth_img.shape[0] or u < 0 or u >= depth_img.shape[1]:
        return None, "uv_oob"

    z_raw = depth_img[v, u]
    if z_raw == 0:
        return None, "depth_zero"
    Z = float(z_raw) * depth_scale

    x_n, y_n = undistort_pixel_to_norm(u, v, K, D)
    X = x_n * Z
    Y = y_n * Z
    Pc = np.array([X, Y, Z], dtype=np.float64)
    return Pc, "ok"

def Tmul_point(T, p3):
    p4 = np.array([p3[0], p3[1], p3[2], 1.0], dtype=np.float64)
    q4 = T @ p4
    return q4[:3]

def on_mouse(event, x, y, flags, param):
    global clicked_table_uv, clicked_obj_uv
    if event == cv2.EVENT_LBUTTONDOWN:
        if clicked_table_uv is None:
            clicked_table_uv = (x, y)
            print(f"[CLICK] TABLE uv = {clicked_table_uv}")
        elif clicked_obj_uv is None:
            clicked_obj_uv = (x, y)
            print(f"[CLICK] OBJECT uv = {clicked_obj_uv}")
        else:
            print("[INFO] already have 2 clicks. Press r to reset.")

def main():
    global clicked_table_uv, clicked_obj_uv

    # 1) 机器人与外参
    T_tool_cam = load_T_4x4(T_TOOL_CAM_TXT)
    print("\n[LOAD] T_tool_camera (camera->tool):\n", T_tool_cam)

    fa = FrankaArm()
    fa.set_speed(SPEED)
    fa.open_gripper()

    # 记录初始位姿（用来最后回位）
    T_home = fa.get_pose()
    print("[HOME] t =", T_home.translation)

    # 2) 订阅相机
    rospy.Subscriber(IMG_TOPIC, Image, rgb_cb, queue_size=1)
    rospy.Subscriber(DEPTH_TOPIC, Image, depth_cb, queue_size=1)
    rospy.Subscriber(INFO_TOPIC, CameraInfo, info_cb, queue_size=1)

    cv2.namedWindow("click_to_grasp_top")
    cv2.setMouseCallback("click_to_grasp_top", on_mouse)

    print("\n=== 操作说明 ===")
    print("1) 先左键点一下【桌面】（靠近水瓶底部区域） -> 记录桌面高度")
    print("2) 再左键点一下【水瓶顶端】 -> 记录目标点")
    print("按键：")
    print("  g : 执行抓取与放置")
    print("  r : 重新点两下")
    print("  q : 退出\n")

    while not rospy.is_shutdown():
        if g_rgb is None:
            time.sleep(0.02)
            continue

        vis = g_rgb.copy()

        if clicked_table_uv is not None:
            cv2.circle(vis, clicked_table_uv, 6, (0,255,255), -1)
            cv2.putText(vis, "TABLE", (clicked_table_uv[0]+8, clicked_table_uv[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        if clicked_obj_uv is not None:
            cv2.circle(vis, clicked_obj_uv, 6, (0,255,0), -1)
            cv2.putText(vis, "OBJ_TOP", (clicked_obj_uv[0]+8, clicked_obj_uv[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("click_to_grasp_top", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('r'):
            clicked_table_uv = None
            clicked_obj_uv = None
            print("[RESET] cleared clicks.")
            continue

        if key == ord('g'):
            if clicked_table_uv is None or clicked_obj_uv is None:
                print("[WARN] 还没点够两次：先桌面，再水瓶顶端。")
                continue
            if g_K is None:
                print("[WARN] camera_info 还没收到。")
                continue

            # 3) 计算桌面点 & 目标点的 Pc
            u1, v1 = clicked_table_uv
            u2, v2 = clicked_obj_uv

            Pc_table, st1 = pixel_to_Pc(u1, v1, g_depth, g_K, g_D, DEPTH_SCALE)
            Pc_top,   st2 = pixel_to_Pc(u2, v2, g_depth, g_K, g_D, DEPTH_SCALE)
            if st1 != "ok":
                print("[WARN] 桌面点深度无效：", st1)
                continue
            if st2 != "ok":
                print("[WARN] 目标点深度无效：", st2)
                continue

            # 4) Pc -> Ptool -> Pworld
            Ptool_table = Tmul_point(T_tool_cam, Pc_table)
            Ptool_top   = Tmul_point(T_tool_cam, Pc_top)

            T_world_tool = fa.get_pose()
            R_wt = np.array(T_world_tool.rotation, dtype=np.float64)
            t_wt = np.array(T_world_tool.translation, dtype=np.float64)

            Pw_table = R_wt @ Ptool_table + t_wt
            Pw_top   = R_wt @ Ptool_top   + t_wt

            z_table = float(Pw_table[2])
            z_top   = float(Pw_top[2])

            # 你点的是“顶端”，我们让夹爪去略低一点夹住（避免空夹）
            z_grasp = z_top - GRASP_DROP_FROM_TOP

            print("\n[MEAS] Pw_table =", Pw_table)
            print("[MEAS] Pw_top   =", Pw_top)
            print(f"[MEAS] z_table={z_table:.4f}  z_top={z_top:.4f}  z_grasp={z_grasp:.4f}")

            # 5) 构造抓取轨迹（顶抓：保持当前朝向不乱转）
            R_goal = np.array(T_world_tool.rotation, dtype=np.float64)  # 保持当前姿态
            xg, yg = float(Pw_top[0]), float(Pw_top[1])

            # 目标上方安全点
            T_above = RigidTransform(
                rotation=R_goal,
                translation=np.array([xg, yg, z_top + SAFE_Z_ABOVE]),
                from_frame='franka_tool', to_frame='world'
            )
            # 抓取点
            T_grasp = RigidTransform(
                rotation=R_goal,
                translation=np.array([xg, yg, z_grasp]),
                from_frame='franka_tool', to_frame='world'
            )
            # 抬起点
            T_lift = RigidTransform(
                rotation=R_goal,
                translation=np.array([xg, yg, z_grasp + LIFT_AFTER_GRASP]),
                from_frame='franka_tool', to_frame='world'
            )

            # 放置：放到“HOME 的 x/y”，高度用桌面z决定（避免撞桌）
            home_xy = np.array([T_home.translation[0], T_home.translation[1]])
            T_place_above = RigidTransform(
                rotation=np.array(T_home.rotation, dtype=np.float64),
                translation=np.array([home_xy[0], home_xy[1], z_table + SAFE_Z_ABOVE]),
                from_frame='franka_tool', to_frame='world'
            )
            T_place = RigidTransform(
                rotation=np.array(T_home.rotation, dtype=np.float64),
                translation=np.array([home_xy[0], home_xy[1], z_table + 0.05]),  # 离桌5cm放
                from_frame='franka_tool', to_frame='world'
            )

            # 6) 执行动作（慢 + 阻抗 + 每段停一下）
            try:
                print("\n[ACTION] go ABOVE")
                fa.goto_pose(T_above, duration=MOVE_TIME, use_impedance=True)
                time.sleep(SLEEP_AFTER)

                print("[ACTION] go GRASP")
                fa.goto_pose(T_grasp, duration=MOVE_TIME, use_impedance=True)
                time.sleep(SLEEP_AFTER)

                print("[ACTION] close_gripper")
                fa.close_gripper()
                time.sleep(1.0)

                print("[ACTION] LIFT")
                fa.goto_pose(T_lift, duration=MOVE_TIME, use_impedance=True)
                time.sleep(SLEEP_AFTER)

                print("[ACTION] go PLACE_ABOVE")
                fa.goto_pose(T_place_above, duration=MOVE_TIME, use_impedance=True)
                time.sleep(SLEEP_AFTER)

                print("[ACTION] go PLACE")
                fa.goto_pose(T_place, duration=MOVE_TIME, use_impedance=True)
                time.sleep(SLEEP_AFTER)

                print("[ACTION] open_gripper")
                fa.open_gripper()
                time.sleep(1.0)

                print("[ACTION] back HOME (exact)")
                fa.goto_pose(T_home, duration=MOVE_TIME, use_impedance=True)
                time.sleep(SLEEP_AFTER)

                print("\n[DONE] pick&place finished.\n")

            except KeyboardInterrupt:
                print("\n[INTERRUPT] user stop.")
                break
            except Exception as e:
                print("\n[ERROR] action failed:", e)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
