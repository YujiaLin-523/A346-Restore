import cv2
import numpy as np
import time
import os
import threading
from .utils import load_matrix, get_robust_depth, pixel_to_3d, transform_point, make_T

class GraspDemo:
    def __init__(self, camera, robot, config):
        self.camera = camera
        self.robot = robot
        self.config = config
        
        self.T_tool_camera = None
        
        # Config params
        self.safe_height = config['grasping']['safety_height']
        self.pre_grasp_z = config['grasping']['pre_grasp_z_offset']
        self.roi_size = config['grasping']['depth_roi_size']
        self.contact_force = config['grasping'].get('contact_force_threshold', 8.0)
        
        # State
        self.pick_uv = None           # 当前点击的抓取点
        self.place_uv = None          # 当前点击的放置点
        self.has_object = False       # 是否已经抓起物体
        self.last_click = None        # 用于可视化的最近一次点击
        self.is_running = False
        self.worker = None            # 后台执行抓/放线程
        self.pick_frame = None        # (rgb, depth) 在点击时刻的拷贝
        self.place_frame = None       # (rgb, depth) 在点击时刻的拷贝

    def load_calibration(self):
        path = self.config['paths']['handeye_result']
        try:
            self.T_tool_camera = load_matrix(path)
            print(f"[INFO] Loaded Hand-Eye Matrix from {path}")
            return True
        except Exception as e:
            print(f"[ERROR] Could not load calibration: {e}")
            print("Please run calibration mode first.")
            return False

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.has_object:
                self.pick_uv = (x, y)
                self.last_click = ('pick', x, y)
                print(f"[UI] 选中抓取点: ({x}, {y})")
            else:
                self.place_uv = (x, y)
                self.last_click = ('place', x, y)
                print(f"[UI] 选中放置点: ({x}, {y})")

    def run(self):
        if not self.load_calibration():
            return

        print("====================================")
        print("          Grasp Demo")
        print("====================================")
        print("1. 点击物体顶部进行抓取；抓起成功后再点击放置点。")
        print("2. 相机画面会显示点击点（绿色=抓取，蓝色=放置）。")
        print("3. 按 'Q' 退出。")
        
        # Open gripper initially
        self.robot.open_gripper()
        
        cv2.namedWindow("Grasp Demo", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Grasp Demo", self.mouse_cb)
        
        self.is_running = True
        
        try:
            while self.is_running:
                rgb, depth = self.camera.get_frame()
                if rgb is None:
                    time.sleep(0.1)
                    continue
                
                vis = rgb.copy()

                # 画出最近一次点击点
                if self.last_click is not None:
                    mode, x, y = self.last_click
                    color = (0, 255, 0) if mode == 'pick' else (255, 128, 0)
                    cv2.circle(vis, (x, y), 6, color, 2)
                    cv2.putText(vis, mode, (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # 在最新帧上保存点击时刻的图像/深度，避免后续遮挡导致取点失败
                if self.pick_uv is not None and not self.has_object:
                    self.pick_frame = (rgb.copy(), depth.copy() if depth is not None else None)
                if self.place_uv is not None and self.has_object:
                    self.place_frame = (rgb.copy(), depth.copy() if depth is not None else None)

                # 检查后台线程状态
                if self.worker is not None and not self.worker.is_alive():
                    self.worker = None

                # 如果有抓取点击且未持物且当前无任务，则启动抓取线程
                if (self.pick_uv is not None) and (not self.has_object) and (self.worker is None):
                    u, v = self.pick_uv
                    self.pick_uv = None
                    rgb_copy, depth_copy = (self.pick_frame if self.pick_frame else (rgb.copy(), depth.copy() if depth is not None else None))
                    self.worker = threading.Thread(target=self._grasp_task, args=(u, v, depth_copy, rgb_copy))
                    self.worker.start()
                    print("[STATE] 抓取任务已开始，等待完成...")

                # 如果有放置点击且已持物且当前无任务，则启动放置线程
                if (self.place_uv is not None) and self.has_object and (self.worker is None):
                    u, v = self.place_uv
                    self.place_uv = None
                    rgb_copy, depth_copy = (self.place_frame if self.place_frame else (rgb.copy(), depth.copy() if depth is not None else None))
                    self.worker = threading.Thread(target=self._place_task, args=(u, v, depth_copy, rgb_copy))
                    self.worker.start()
                    print("[STATE] 放置任务已开始，等待完成...")
                
                cv2.imshow("Grasp Demo", vis)
                key = cv2.waitKey(30)
                if key == ord('q'):
                    self.is_running = False
                    
        finally:
            cv2.destroyWindow("Grasp Demo")

    def _compute_world_point(self, u, v, depth_img, purpose="grasp"):
        """Compute world coordinates from pixel/depth. Returns (Pw, R_w_t_curr) or (None, None)."""
        z = get_robust_depth(depth_img, u, v, roi_size=self.roi_size)
        if z is None or z < 0.1 or z > 2.0:
            print(f"[FAIL] {purpose}: Invalid depth at ({u},{v}): {z}")
            return None, None
        K, D = self.camera.get_intrinsics()
        Pc = pixel_to_3d(u, v, z, K, D)
        if Pc is None:
            print(f"[FAIL] {purpose}: Cannot backproject pixel")
            return None, None
        Pt = transform_point(self.T_tool_camera, Pc)
        R_w_t_curr, t_w_t_curr = self.robot.get_pose()
        T_w_t_curr = make_T(R_w_t_curr, t_w_t_curr)
        Pw = transform_point(T_w_t_curr, Pt)
        return Pw, R_w_t_curr

    def _safe_height(self, z, margin=0.0):
        return max(z, self.safe_height + margin)

    def _grasp_task(self, u, v, depth_img, vis_img):
        print("\n--- Executing Grasp ---")
        Pw, grasp_R = self._compute_world_point(u, v, depth_img, purpose="grasp")
        if Pw is None:
            print("[STATE] 抓取失败：无法计算世界坐标。")
            self.worker = None # 释放线程锁
            return

        print(f"[INFO] Point in World: {Pw}")

        # Waypoints with safety clamps
        target_pre = Pw.copy()
        target_pre[2] = self._safe_height(target_pre[2] + self.pre_grasp_z, margin=0.02)
        target_grasp = Pw.copy()
        target_grasp[2] = max(self.safe_height, target_grasp[2] - 0.015)

        # Execute
        print("1. Moving to Pre-Grasp...")
        # 【修改点】：增加运行结果判定
        success = self.robot.move_to_pose(grasp_R, target_pre, duration=1.5)
        if success is False: # 假设你的 move_to_pose 失败返回 False
            print("[ERROR] 移动到 Pre-Grasp 点失败 (可能发生碰撞或超出范围)，取消抓取！")
            return

        print("2. Moving to Grasp...")
        success = self.robot.move_to_pose(grasp_R, target_grasp, duration=1.2)
        if success is False:
            print("[ERROR] 移动到 Grasp 点失败，取消抓取！")
            return

        print("3. Closing Gripper...")
        self.robot.close_gripper()
        time.sleep(1.0) # 【建议】：给爪子闭合留一点物理时间

        print("4. Lifting...")
        lift_target = target_grasp.copy()
        lift_target[2] = self._safe_height(lift_target[2] + 0.20, margin=0.02)
        self.robot.move_to_pose(grasp_R, lift_target, duration=1.2)

        # Reset arm state to avoid residual skills
        self.robot.reset_arm()
        print("--- Grasp Complete ---\n")

        self.has_object = True
        print("[STATE] 抓取成功，请点击放置点。")

    def _place_task(self, u, v, depth_img, vis_img):
        print("\n--- Executing Place ---")
        Pw, place_R = self._compute_world_point(u, v, depth_img, purpose="place")
        if Pw is None:
            print("[STATE] 放置失败：无法计算世界坐标。")
            return

        print(f"[INFO] Place Point in World: {Pw}")

        target_pre = Pw.copy()
        target_pre[2] = self._safe_height(target_pre[2] + self.pre_grasp_z, margin=0.02)

        # 放置高度：允许更低，最低夹到安全高度+1cm，兼顾安全与贴近桌面
        target_place = Pw.copy()
        target_place[2] = max(self.safe_height + 0.01, target_place[2])

        print("1. Moving to Pre-Place...")
        # 【修改点 1】：拦截 Pre-Place 移动失败
        success = self.robot.move_to_pose(place_R, target_pre, duration=3)
        if success is False: 
            print("[ERROR] 移动到 Pre-Place 点失败，取消放置动作！")
            return

        print("2. Moving to Place...")
        # 分段下放，带力反馈停止
        steps = 4
        contact = False
        for i in range(1, steps + 1):
            t_interp = target_pre * (1 - i/steps) + target_place * (i/steps)

            # 【修改点 2】：拦截下探过程中的规划失败
            step_success = self.robot.move_to_pose(place_R, t_interp, duration=0.8)
            if step_success is False:
                print(f"[ERROR] 下降过程 (第 {i} 步) 运动受阻或规划失败，提前停止下探。")
                break # 跳出循环，直接执行下一步：张开爪子

            f = self.robot.get_ee_force_norm()
            print(f"[INFO] Place step {i}/{steps}, force={f:.2f}N")
            if f > self.contact_force:
                print(f"[WARN] Force {f:.2f}N > threshold {self.contact_force}N, stopping descent.")
                contact = True
                break

        print("3. Opening Gripper...")
        self.robot.open_gripper()
        time.sleep(1.0) # 【修改点 3】：给爪子完全张开留出 1 秒物理时间，防止立刻抬起刮倒水瓶

        print("4. Lifting away...")
        lift_target = target_place.copy()
        lift_target[2] = self._safe_height(lift_target[2] + 0.15, margin=0.02)
        self.robot.move_to_pose(place_R, lift_target, duration=1.2)

        # Reset arm state after placing
        self.robot.reset_arm()
        print("--- Place Complete ---\n")

        self.has_object = False
        self.last_click = None
        print("[STATE] 放置完成。")
