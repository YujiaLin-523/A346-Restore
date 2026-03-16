import rospy
import yaml
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
import threading
import os
import cv2

class CameraInterface:
    def __init__(self, config):
        self.config = config
        self.bridge = CvBridge()
        
        # Data containers
        self.last_rgb = None
        self.last_depth = None
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Load intrinsics immediately from YAML (no need to wait for ROS topic if we have the file)
        self._load_intrinsics()
        
        # Setup ROS subscribers
        self.rgb_topic = config['common']['rgb_topic']
        self.depth_topic = config['common']['depth_topic']
        
        self._lock = threading.Lock()
        
        # We start subscribers only when initialize is called to avoid node init issues during import
        self.subs = []

    def start(self):
        """Start ROS subscribers."""
        # Do not call rospy.init_node() here; the node should be
        # initialized centrally to avoid multiple init attempts.
        # Subscribers assume a node is already running.
        self.subs.append(rospy.Subscriber(self.rgb_topic, Image, self._rgb_cb))
        self.subs.append(rospy.Subscriber(self.depth_topic, Image, self._depth_cb))
        
        # ---------------- DIAGNOSTICS START ----------------
        try:
            print("[INFO] Diagnosing ROS topics...")
            all_topics = [t for t, _ in rospy.get_published_topics()]
            if self.rgb_topic not in all_topics:
                print(f"\033[93m[WARN] Configured RGB topic '{self.rgb_topic}' is NOT published!\033[0m")
                print(f"[INFO] Available topics: {all_topics}")
                print("[HINT] Check if camera launch file is running or topic name matches config.yaml")
            else:
                print(f"\033[92m[OK] RGB topic '{self.rgb_topic}' found.\033[0m")
        except Exception as e:
            print(f"[WARN] Failed to check topics: {e}")
        # ---------------- DIAGNOSTICS END ----------------

        print(f"[INFO] Camera Interface started. Listening on:")
        print(f"       RGB: {self.rgb_topic}")
        print(f"       Depth: {self.depth_topic}")

    def _load_intrinsics(self):
        """Parse ost.yaml for camera intrinsics."""
        path = self.config['paths']['calib_yaml']
        # Handle relative path
        if not os.path.exists(path):
             base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
             path = os.path.join(base_dir, path)

        if not os.path.exists(path):
            print(f"[WARN] Calibration file {path} not found. Intrinsics will be empty.")
            return

        with open(path, "r") as f:
            d = yaml.safe_load(f)

        try:
            self.width  = int(d.get("image_width", 640))
            self.height = int(d.get("image_height", 480))
            self.dist_coeffs = np.array(d["distortion_coefficients"]["data"])
            self.camera_matrix = np.array(d["camera_matrix"]["data"]).reshape(3, 3)
            # You can also load rectification/projection if needed
            print(f"[INFO] Loaded intrinsics from {path}")
        except Exception as e:
            print(f"[ERROR] Failed to parse {path}: {e}")

    def _rgb_cb(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self._lock:
                if self.last_rgb is None:
                    print(f"\033[92m[INFO] First RGB frame received! Shape: {cv_image.shape}\033[0m")
                self.last_rgb = cv_image
        except Exception as e:
            print(f"[ERR] RGB CB: {e}")

    def _depth_cb(self, msg):
        try:
            # passthrough gives the original 16UC1 (usually)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            with self._lock:
                self.last_depth = cv_image
        except Exception as e:
            print(f"[ERR] Depth CB: {e}")

    def get_frame(self):
        """Return latest (rgb, depth) copy."""
        with self._lock:
            if self.last_rgb is None or self.last_depth is None:
                return None, None
            return self.last_rgb.copy(), self.last_depth.copy()

    def get_intrinsics(self):
        return self.camera_matrix, self.dist_coeffs
