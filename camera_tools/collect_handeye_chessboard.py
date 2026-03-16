#!/usr/bin/env python3
import os, time
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from frankapy import FrankaArm

# ====== 只改这里 ======
PATTERN_SIZE = (11, 8)        # 内角点 (cols, rows)
SQUARE_SIZE  = 0.015          # 15mm -> 0.015 m
IMG_TOPIC    = "/camera/color/image_raw"
INFO_TOPIC   = "/camera/color/camera_info_calib"   # 关键：用_calib
OUT_FILE     = os.path.expanduser("~/camera_tools/handeye_samples.npz")
GUIDE_TIME_S = 600            # guide mode 时间（10min够采30组）
# ======================

bridge = CvBridge()
g_img = None
g_K = None
g_D = None

def img_cb(msg):
    global g_img
    g_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

def info_cb(msg: CameraInfo):
    global g_K, g_D
    g_K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
    g_D = np.array(msg.D, dtype=np.float64).reshape(-1)

def make_object_points():
    objp = []
    for r in range(PATTERN_SIZE[1]):
        for c in range(PATTERN_SIZE[0]):
            objp.append([c * SQUARE_SIZE, r * SQUARE_SIZE, 0.0])
    return np.array(objp, dtype=np.float64)

def rot_angle_deg(Ra, Rb):
    dR = Ra.T @ Rb
    c = (np.trace(dR) - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def rt_from_pnp(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    return R, t

if __name__ == "__main__":

    fa = FrankaArm()
    fa.open_gripper()

    # 进入 guide mode（核心！）
    print("\n[STEP] Entering guide mode. Keep chessboard fixed on the table.")
    print("[STEP] Move the arm around, vary orientation a lot. Press Ctrl+C to stop guide mode early if needed.")
    fa.run_guide_mode(GUIDE_TIME_S, block=False)  # 非阻塞：我们边 guide 边采样

    rospy.Subscriber(IMG_TOPIC, Image, img_cb, queue_size=1)
    rospy.Subscriber(INFO_TOPIC, CameraInfo, info_cb, queue_size=1)

    objp = make_object_points()

    R_g2b, t_g2b = [], []
    R_t2c, t_t2c = [], []

    last_R = None
    last_t = None
    last_j = None

    print("\n=== Hand-Eye Collect (Eye-in-Hand) ===")
    print("q: quit & save   |   s: save sample")
    print("保存规则：相邻两次保存 旋转变化 >= 10deg（或平移 >= 2mm）否则拒绝保存")
    print(f"OUT: {OUT_FILE}\n")

    ok_pnp = False
    found = False
    rvec = tvec = None

    while not rospy.is_shutdown():
        if g_img is None or g_K is None or g_D is None:
            time.sleep(0.02)
            continue

        vis = g_img.copy()
        gray = cv2.cvtColor(g_img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)

        ok_pnp = False
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), term)
            cv2.drawChessboardCorners(vis, PATTERN_SIZE, corners2, found)

            ok_pnp, rvec, tvec = cv2.solvePnP(objp, corners2, g_K, g_D, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok_pnp:
                cv2.putText(vis, "FOUND (press s to save)", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        else:
            cv2.putText(vis, "NOT FOUND", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.putText(vis, f"saved: {len(R_g2b)}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        cv2.imshow("collect_handeye_chessboard_v2", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('s'):
            if not (found and ok_pnp):
                print("[REJECT] chessboard not found / PnP failed.")
                continue

            # 读取机器人当前位姿
            T_world_tool = fa.get_pose()
            R_world_tool = np.array(T_world_tool.rotation, dtype=np.float64)
            t_world_tool = np.array(T_world_tool.translation, dtype=np.float64).reshape(3)
            j = np.array(fa.get_joints(), dtype=np.float64)

            # 强校验：必须有足够“信息量”的运动
            if last_R is not None:
                dang = rot_angle_deg(last_R, R_world_tool)
                dt = float(np.linalg.norm(t_world_tool - last_t))
                dj = float(np.linalg.norm(j - last_j))
                print(f"[CHECK] delta_rot={dang:.2f}deg | delta_t={dt*1000:.1f}mm | delta_j={dj:.3f}")

                if dang < 10.0 and dt < 0.002:
                    print("[REJECT] robot motion too small (<10deg and <2mm). Rotate wrist more before saving!")
                    continue

            # target->camera
            Rt, tt = rt_from_pnp(rvec, tvec)

            R_g2b.append(R_world_tool)
            t_g2b.append(t_world_tool)
            R_t2c.append(Rt)
            t_t2c.append(tt)

            last_R, last_t, last_j = R_world_tool, t_world_tool, j
            print(f"[SAVE] N={len(R_g2b)} | t_world_tool={t_world_tool} | t_target2cam={tt}")

    cv2.destroyAllWindows()

    if len(R_g2b) < 10:
        print(f"[WARN] only {len(R_g2b)} samples, too few. Still saving for debug.")

    np.savez(
        OUT_FILE,
        R_gripper2base=np.stack(R_g2b, axis=0),
        t_gripper2base=np.stack(t_g2b, axis=0),
        R_target2cam=np.stack(R_t2c, axis=0),
        t_target2cam=np.stack(t_t2c, axis=0),
        pattern_size=np.array(PATTERN_SIZE),
        square_size=np.array([SQUARE_SIZE]),
    )
    print(f"\n[DONE] saved {len(R_g2b)} samples to: {OUT_FILE}\n")
