import cv2
import numpy as np
import time
import os
from .utils import load_config, save_matrix, make_T

class HandEyeCalibration:
    def __init__(self, camera, robot, config):
        self.camera = camera
        self.robot = robot
        
        self.pattern_size = tuple(config['calibration']['pattern_size']) # (11, 8)
        self.square_size = config['calibration']['square_size']
        self.out_file = config['paths']['handeye_result']
        self.samples_dir = config['paths']['samples_dir']
        
        # Prepare 3D points of the chessboard in target frame
        self.objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size
        
        # Data storage
        self.samples = [] # List of dicts: {R_g2b, t_g2b, R_t2c, t_t2c}

    def run(self):
        print("====================================")
        print("   Hand-Eye Calibration Mode")
        print("====================================")
        print("1. Robot will enter Guide Mode (move freely).")
        print("2. Point camera at chessboard at different angles.")
        print("3. Press 'S' to capture a sample.")
        print("4. Press 'Q' to finish and solve.")
        
        self.robot.open_gripper()
        self.robot.start_guide_mode(duration=600)
        
        # Make sure intrinsics are loaded
        K, D = self.camera.get_intrinsics()
        if K is None:
             print("[ERROR] Camera intrinsics not found. Cannot calibrate.")
             self.robot.stop_guide_mode()
             return

        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        
        last_save_time = 0
        
        try:
            wait_count = 0
            while True:
                rgb, _ = self.camera.get_frame()
                if rgb is None:
                    wait_count += 1
                    if wait_count % 20 == 0: # Print every ~2 seconds
                        print(f"[WARN] Waiting for camera image... (Is camera running?)")
                    time.sleep(0.1)
                    continue
                
                wait_count = 0 # Reset counter on success
                
                gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                found, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
                
                R_t2c, t_t2c = None, None
                valid_pose = False
                
                vis = rgb.copy()
                
                if found:
                     term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                     corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
                     cv2.drawChessboardCorners(vis, self.pattern_size, corners2, found)
                     
                     valid_pose, rvec, tvec = cv2.solvePnP(self.objp, corners2, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
                     
                     if valid_pose:
                         R_t2c, _ = cv2.Rodrigues(rvec)
                         t_t2c = tvec
                         
                         # Draw axis
                         cv2.drawFrameAxes(vis, K, D, rvec, tvec, 0.1)
                         cv2.putText(vis, "READY TO CAPTURE", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                else:
                    cv2.putText(vis, "SHOW CHESSBOARD", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                cv2.putText(vis, f"Samples: {len(self.samples)}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.imshow("Calibration", vis)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s'):
                    if not valid_pose:
                        print("[WARN] Chessboard valid pose not found, cannot save.")
                    else:
                        R_g2b, t_g2b = self.robot.get_pose()  # tool->world
                        
                        sample = {
                            "R_gripper2base": R_g2b,
                            "t_gripper2base": t_g2b,
                            "R_target2cam": R_t2c,  # camera<-target
                            "t_target2cam": t_t2c
                        }
                        self.samples.append(sample)
                        print(f"[INFO] Sample {len(self.samples)} captured.")
                        
                elif key == ord('q'):
                    break
                    
        finally:
            self.robot.stop_guide_mode()
            cv2.destroyWindow("Calibration")
            
        if len(self.samples) < 5:
            print("[WARN] Not enough samples (<5) to solve.")
            return

        self._solve_and_save()

    def _solve_and_save(self):
        print("\n[INFO] Solving Hand-Eye Calibration...")
        
        R_g2b = np.array([s["R_gripper2base"] for s in self.samples])
        t_g2b = np.array([s["t_gripper2base"] for s in self.samples])
        R_t2c = np.array([s["R_target2cam"] for s in self.samples])
        t_t2c = np.array([s["t_target2cam"] for s in self.samples])
        
        # cv2.calibrateHandEye expects:
        # R_gripper2base: Rotation part of the transformation from gripper frame to robot base frame
        # t_gripper2base: Translation part...
        # R_target2cam: ... from target frame to camera frame 
        # t_target2cam: ...
        
        # Solving for AX=XB or similar.
        # Output is R_cam2gripper, t_cam2gripper -> Camera (eye) in Hand (gripper)
        
        R_c2g, t_c2g = cv2.calibrateHandEye(
            R_g2b, t_g2b,
            R_t2c, t_t2c,
            method=cv2.CALIB_HAND_EYE_TSAI
        )
        
        T_tool_camera = make_T(R_c2g, t_c2g)
        
        print("\n=== Use this Matrix (Camera in Tool Frame) ===")
        print(T_tool_camera)
        
        # Verify
        print("\n[INFO] Consistency Check (World->Target should be constant):")
        targets_w = []
        for i in range(len(self.samples)):
             T_w_t = make_T(R_g2b[i], t_g2b[i])
             T_c_tar = make_T(R_t2c[i], t_t2c[i])
             # world<-target = world<-tool * tool<-camera * camera<-target
             T_w_tar = T_w_t @ T_tool_camera @ T_c_tar
             targets_w.append(T_w_tar[:3, 3])
             
        targets_w = np.array(targets_w)
        std_dev = np.std(targets_w, axis=0)
        mean_pos = np.mean(targets_w, axis=0)
        
        print(f"  Target Mean Pos (World): {mean_pos}")
        print(f"  Std Dev (m): {std_dev}")
        
        if np.max(std_dev) > 0.01:
            print("[WARN] Std dev > 1cm. Calibration might be inaccurate. Try collecting more diverse angles.")
        else:
            print("[SUCCESS] Calibration looks good!")
            
        save_matrix(T_tool_camera, self.out_file)
        
        # Also save raw samples just in case
        if self.samples_dir:
             os.makedirs(self.samples_dir, exist_ok=True)
             np.savez(os.path.join(self.samples_dir, f"samples_{int(time.time())}.npz"), 
                      R_gripper2base=R_g2b, t_gripper2base=t_g2b,
                      R_target2cam=R_t2c, t_target2cam=t_t2c)
