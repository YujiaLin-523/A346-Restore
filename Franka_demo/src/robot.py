import rospy
import numpy as np
try:
    from frankapy import FrankaArm
    from autolab_core import RigidTransform
except ImportError:
    print("Warning: frankapy or autolab_core not found. Robot control will fail.")
    FrankaArm = None
    RigidTransform = None

class RobotController:
    def __init__(self, config):
        self.config = config
        self.fa = None
        
        if FrankaArm is not None:
             print("[INFO] Initializing FrankaArm...")
             self.fa = FrankaArm()
             print("[INFO] FrankaArm initialized.")
        
    def get_pose(self):
        """Returns (R, t) of end-effector in base frame."""
        if not self.fa: return np.eye(3), np.zeros(3)
        pose = self.fa.get_pose()
        return pose.rotation, pose.translation

    def start_guide_mode(self, duration=600):
        if not self.fa: return
        # run_guide_mode allows manual movement
        print("[ROBOT] Entering Guide Mode.")
        self.fa.run_guide_mode(duration=duration, block=False)

    def stop_guide_mode(self):
        if not self.fa: return
        try:
             self.fa.stop_skill()
             print("[ROBOT] Guide Mode stopped.")
        except:
             pass

    def open_gripper(self):
        if self.fa: self.fa.open_gripper()

    def close_gripper(self):
        # We can implement parameters here if needed, but defaults are usually fine
        if self.fa: 
            # grasp defaults are grasp_width=0.0, force=40.0, etc.
            # Using goto_gripper is sometimes safer for simple close
            self.fa.goto_gripper(0.0, grasp=True, force=40.0)

    def move_to_pose(self, R, t, duration=2.0):
        if not self.fa: return
        
        # Ensure previous skill is stopped before starting a new one
        try:
            self.fa.stop_skill()
        except:
            pass
            
        target_pose = RigidTransform(
            rotation=R,
            translation=t,
            from_frame='franka_tool',
            to_frame='world'
        )
        
        # dynamic=False creates a discrete trajectory (easier to chain)
        self.fa.goto_pose(target_pose, duration=duration, dynamic=False, buffer_time=1)

    def get_joint_angles(self):
        if not self.fa: return np.zeros(7)
        return self.fa.get_joints()

    def get_ee_force_norm(self):
        """Return Euclidean norm of end-effector force if available, else 0."""
        if not self.fa:
            return 0.0
        if hasattr(self.fa, "get_ee_force_torque"):
            try:
                wrench = self.fa.get_ee_force_torque()
                # wrench: [Fx, Fy, Fz, Tx, Ty, Tz]
                return float(np.linalg.norm(wrench[:3]))
            except Exception:
                return 0.0
        return 0.0

    def reset_arm(self):
        """Reset the arm after a skill. Tries common frankapy reset calls."""
        if not self.fa:
            return
        # Some frankapy versions provide reset/stop helpers; fall back gracefully
        for fn in ("reset_joints", "reset", "stop_skill"):
            if hasattr(self.fa, fn):
                try:
                    getattr(self.fa, fn)()
                    return
                except Exception:
                    continue
        print("[WARN] No reset method available on FrankaArm; skipping reset.")
