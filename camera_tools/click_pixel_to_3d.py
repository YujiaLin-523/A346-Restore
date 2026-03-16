#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

RGB_TOPIC   = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/aligned_depth_to_color/image_raw"
INFO_TOPIC  = "/camera/color/camera_info_calib"

bridge = CvBridge()

fx = fy = cx = cy = None
latest_rgb = None
latest_depth = None
latest_depth_encoding = None
last_print_t = 0.0

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
        # 16UC1 通常是 mm
        return float(np.median(vals)) * 0.001
    elif encoding == "32FC1":
        vals = patch[np.isfinite(patch) & (patch > 0)].astype(np.float64)
        if vals.size == 0:
            return None
        return float(np.median(vals))
    else:
        return None

def on_mouse(event, x, y, flags, param):
    global latest_rgb, latest_depth, latest_depth_encoding, fx, fy, cx, cy

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if latest_rgb is None:
        print("[WARN] 还没收到 RGB 图像")
        return
    if latest_depth is None:
        print("[WARN] 还没收到 Depth 图像（检查是否启动 align_depth:=true，以及 DEPTH_TOPIC 是否正确）")
        return
    if fx is None:
        print("[WARN] 还没收到 CameraInfo（INFO_TOPIC 可能不对或 calib 节点没启动）")
        return

    u, v = int(x), int(y)

    # 防止 depth 尺寸与 rgb 不一致（理论上 aligned 后一致）
    hd, wd = latest_depth.shape[:2]
    if u < 0 or u >= wd or v < 0 or v >= hd:
        print(f"[WARN] 点击像素越界：click=({u},{v}) depth_size=({wd},{hd})")
        return

    Z = robust_depth(latest_depth, latest_depth_encoding, u, v, win=3)
    if Z is None:
        print(f"[WARN] 深度无效：({u},{v}) encoding={latest_depth_encoding}")
        return

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    print("\n=== Click Pixel -> 3D @ camera_color_optical_frame ===")
    print(f"pixel (u,v) = ({u}, {v})")
    print(f"depth Z     = {Z:.4f} m   (encoding={latest_depth_encoding})")
    print(f"fx,fy,cx,cy = {fx:.3f}, {fy:.3f}, {cx:.3f}, {cy:.3f}")
    print(f"Pc=[X,Y,Z]  = [{X:.4f}, {Y:.4f}, {Z:.4f}] (m)")
    print("axes(optical): x->right, y->down, z->forward\n")

if __name__ == "__main__":
    rospy.init_node("click_pixel_to_3d_v2", anonymous=True)

    rospy.Subscriber(INFO_TOPIC, CameraInfo, info_cb, queue_size=1)
    rospy.Subscriber(RGB_TOPIC, Image, rgb_cb, queue_size=1)
    rospy.Subscriber(DEPTH_TOPIC, Image, depth_cb, queue_size=1)

    cv2.namedWindow("rgb")
    cv2.setMouseCallback("rgb", on_mouse)

    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        # 每秒打印一次“收到没有”，避免你感觉“无反应”
        t = rospy.get_time()
        if t - last_print_t > 1.0:
            last_print_t = t
            print(f"[STAT] rgb={'OK' if latest_rgb is not None else 'None'} | "
                  f"depth={'OK' if latest_depth is not None else 'None'}({latest_depth_encoding}) | "
                  f"caminfo={'OK' if fx is not None else 'None'}")

        if latest_rgb is not None:
            cv2.imshow("rgb", latest_rgb)
            cv2.waitKey(1)

        rate.sleep()
