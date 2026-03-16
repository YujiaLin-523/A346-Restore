#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

from frankapy import FrankaArm
from autolab_core import RigidTransform

# ---------- 话题名（按你当前在用的） ----------
RGB_TOPIC   = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/aligned_depth_to_color/image_raw"
INFO_TOPIC  = "/camera/color/camera_info_calib"

# ---------- 粗外参：相机在 tool 前方 6cm ----------
CAM_OFFSET_TOOL = np.array([0.06, 0.0, 0.0])   # meters, in franka_tool

# 相机“朝下”已知，但相机绕自身光轴的旋转（yaw）不确定
# 先从 0 开始，如果方向不对，就改成 90 / -90 / 180 试
CAM_YAW_DEG = 0.0

def rotz(deg):
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float64)

bridge = CvBridge()

fx = fy = cx = cy = None
latest_rgb = None
latest_depth = None
latest_depth_encoding = None
last_stat_t = 0.0

# 初始化机械臂接口（失败也不影响 Pc 输出）
fa = None
try:
    fa = FrankaArm()
except Exception as e:
    print("[WARN] FrankaArm 初始化失败，将只输出 Pc，不输出 Pw。错误：", e)

def info_cb(msg: CameraInfo):
    global fx, fy, cx, cy
    K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

def rgb_cb(msg: Image):
    global latest_rgb
    latest_rgb = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

def depth_cb(msg: Image):
    global latest_depth, latest_depth_encoding
    latest_depth_encoding = msg.encoding
    latest_depth = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

def robust_depth(depth_img, encoding, u, v, win=3):
    h, w = depth_img.shape[:2]
    u0, u1 = max(0, u-win), min(w-1, u+win)
    v0, v1 = max(0, v-win), min(h-1, v+win)
    patch = depth_img[v0:v1+1, u0:u1+1].copy()

    if encoding == "16UC1":
        vals = patch[patch > 0].astype(np.float64)
        if vals.size == 0:
            return None
        return float(np.median(vals)) * 0.001  # mm -> m
    elif encoding == "32FC1":
        vals = patch[np.isfinite(patch) & (patch > 0)].astype(np.float64)
        if vals.size == 0:
            return None
        return float(np.median(vals))
    else:
        return None

def on_mouse(event, x, y, flags, param):
    global fx, fy, cx, cy, latest_rgb, latest_depth, latest_depth_encoding, fa

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if latest_rgb is None:
        print("[WARN] 还没收到 RGB 图像")
        return
    if latest_depth is None:
        print("[WARN] 还没收到 Depth 图像（检查 align_depth:=true & DEPTH_TOPIC）")
        return
    if fx is None:
        print("[WARN] 还没收到 CameraInfo（检查 INFO_TOPIC）")
        return

    u, v = int(x), int(y)

    hd, wd = latest_depth.shape[:2]
    if u < 0 or u >= wd or v < 0 or v >= hd:
        print(f"[WARN] 点击越界：click=({u},{v}) depth_size=({wd},{hd})")
        return

    Z = robust_depth(latest_depth, latest_depth_encoding, u, v, win=3)
    if Z is None:
        print(f"[WARN] 深度无效：({u},{v}) encoding={latest_depth_encoding}")
        return

    # ---- 1) 相机系点 Pc ----
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Pc = np.array([X, Y, Z], dtype=np.float64)

    print("\n=== Click Pixel -> Pc (camera_color_optical_frame) ===")
    print(f"pixel (u,v) = ({u}, {v})")
    print(f"Z           = {Z:.4f} m (encoding={latest_depth_encoding})")
    print(f"Pc=[X,Y,Z]  = [{Pc[0]:.4f}, {Pc[1]:.4f}, {Pc[2]:.4f}] m")

    # ---- 2) 粗外参：Pc -> Pt -> Pw ----
    if fa is None:
        print("[WARN] 没有 FrankaArm，无法计算 world 点 Pw。")
        return

    # 相机->tool：旋转（先只做绕 z 的 yaw），平移（前方 6cm）
    R_tc = rotz(CAM_YAW_DEG)                 # tool <- camera
    t_tc = CAM_OFFSET_TOOL.copy()            # camera origin expressed in tool

    Pt = R_tc @ Pc + t_tc                    # in franka_tool

    # tool->world：从机械臂读当前位姿
    T_wt = fa.get_pose()                     # from franka_tool to world
    Pw = T_wt.rotation @ Pt + T_wt.translation

    print("\n--- Rough Pc -> Pw ---")
    print(f"CAM_YAW_DEG = {CAM_YAW_DEG} deg")
    print(f"t_tc(tool)  = {t_tc} m  (camera is +6cm along tool-x)")
    print(f"Pt(tool)    = [{Pt[0]:.4f}, {Pt[1]:.4f}, {Pt[2]:.4f}] m")
    print(f"Pw(world)   = [{Pw[0]:.4f}, {Pw[1]:.4f}, {Pw[2]:.4f}] m")

    # 附：打印当前 tool 的 z 轴在 world 指向，帮助你理解“朝下”
    z_tool_world = T_wt.rotation[:, 2]
    print(f"tool z-axis in world = [{z_tool_world[0]:.3f}, {z_tool_world[1]:.3f}, {z_tool_world[2]:.3f}]")

if __name__ == "__main__":
    # 如果 frankapy 已经 init_node 过，这里不要重复 init
    if not rospy.core.is_initialized():
    	rospy.init_node("click_pixel_to_world_rough", anonymous=True)
    else:
    	rospy.loginfo("rospy already initialized by another module (likely frankapy).")

    rospy.Subscriber(INFO_TOPIC, CameraInfo, info_cb, queue_size=1)
    rospy.Subscriber(RGB_TOPIC, Image, rgb_cb, queue_size=1)
    rospy.Subscriber(DEPTH_TOPIC, Image, depth_cb, queue_size=1)

    cv2.namedWindow("rgb")
    cv2.setMouseCallback("rgb", on_mouse)

    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        t = rospy.get_time()
        if t - last_stat_t > 1.0:
            last_stat_t = t
            print(f"[STAT] rgb={'OK' if latest_rgb is not None else 'None'} | "
                  f"depth={'OK' if latest_depth is not None else 'None'}({latest_depth_encoding}) | "
                  f"caminfo={'OK' if fx is not None else 'None'} | "
                  f"fa={'OK' if fa is not None else 'None'}")

        if latest_rgb is not None:
            cv2.imshow("rgb", latest_rgb)
            cv2.waitKey(1)
        rate.sleep()
