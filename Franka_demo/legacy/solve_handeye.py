#!/usr/bin/env python3
import os
import numpy as np
import cv2

DATA = os.path.expanduser("~/camera_tools/handeye_samples.npz")
OUT  = os.path.expanduser("~/camera_tools/T_tool_camera.txt")

def make_T(R, t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t.reshape(3)
    return T

if __name__ == "__main__":

    d = np.load(DATA)
    R_g2b = d["R_gripper2base"]   # world<-tool  (tool->world)
    t_g2b = d["t_gripper2base"]
    R_t2c = d["R_target2cam"]     # camera<-target (target->camera)
    t_t2c = d["t_target2cam"]

    n = R_g2b.shape[0]
    assert n >= 10, "样本太少，至少 10，推荐 20~30"

    # OpenCV 输出：R_cam2gripper, t_cam2gripper
    # 也就是：tool <- camera  (camera->tool) 这正是你抓取要用的 T_tool_camera
    R_c2g, t_c2g = cv2.calibrateHandEye(
        R_g2b, t_g2b, R_t2c, t_t2c,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    T_tool_camera = make_T(R_c2g, t_c2g)

    print("\n=== RESULT: T_tool_camera (camera -> tool) ===")
    np.set_printoptions(precision=6, suppress=True)
    print(T_tool_camera)

    # 一个很关键的“是否靠谱”检查：把每次的 target 变到 world，应该几乎不变
    # world<-target = (world<-tool) (tool<-camera) (camera<-target)
    Ts = []
    for i in range(n):
        T_w_t  = make_T(R_g2b[i], t_g2b[i])
        T_c_tar= make_T(R_t2c[i], t_t2c[i])
        T_w_tar = T_w_t @ T_tool_camera @ T_c_tar
        Ts.append(T_w_tar)
    Ts = np.stack(Ts, axis=0)
    t_mean = Ts[:, :3, 3].mean(axis=0)
    t_std  = Ts[:, :3, 3].std(axis=0)

    print("\n[CHECK] target position in world should be nearly constant")
    print("  mean(t_world_target) =", t_mean)
    print("  std (meters)         =", t_std)
    print("  (经验：std < 0.005~0.01m 属于很好；>2cm 说明采样不够或角点质量差)")

    with open(OUT, "w") as f:
        for r in range(4):
            f.write(" ".join([f"{T_tool_camera[r,c]:.8f}" for c in range(4)]) + "\n")

    print(f"\n[SAVED] {OUT}\n")
