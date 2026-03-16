import yaml, rospy
from sensor_msgs.msg import Image, CameraInfo

YAML_PATH="/home/glasgow/camera_calib_export/ost.yaml"
OUT_INFO="/camera/color/camera_info_calib"

# 你实际的图像话题（二选一，哪个存在用哪个）
CANDIDATES=["/camera/color/image_raw", "/camera/color/image_rect_raw", "/camera/color/image_rect_color"]

def pick_image_topic():
    topics = [t for t,_ in rospy.get_published_topics()]
    for c in CANDIDATES:
        if c in topics:
            return c
    return None

d = yaml.safe_load(open(YAML_PATH, "r"))

# 从 YAML 读出内参（确保是 list，不用 numpy）
width  = int(d.get("image_width", 0))
height = int(d.get("image_height", 0))
model  = d.get("distortion_model", "plumb_bob")
D = [float(x) for x in d["distortion_coefficients"]["data"][:5]]
K = [float(x) for x in d["camera_matrix"]["data"]]
R = [float(x) for x in d.get("rectification_matrix", {}).get("data", [1,0,0,0,1,0,0,0,1])]
P0 = d.get("projection_matrix", {}).get("data", None)
P = [float(x) for x in P0] if P0 and len(P0)==12 else [K[0],0,K[2],0, 0,K[4],K[5],0, 0,0,1,0]

rospy.init_node("caminfo_from_yaml", anonymous=True)

img_topic = pick_image_topic()
if not img_topic:
    raise SystemExit("找不到 /camera/color/image_* 图像话题：先确认 realsense 节点已启动并在发布图像。")

print("[OK] YAML_PATH:", YAML_PATH)
print("[OK] D:", D)
print("[OK] K:", K)
print("[SUB ]", img_topic)
print("[PUB ]", OUT_INFO)

pub = rospy.Publisher(OUT_INFO, CameraInfo, queue_size=1)

def cb(img: Image):
    msg = CameraInfo()
    msg.header = img.header
    msg.height = height
    msg.width  = width
    msg.distortion_model = model
    msg.D = list(D)
    msg.K = list(K)
    msg.R = list(R)
    msg.P = list(P)
    pub.publish(msg)

rospy.Subscriber(img_topic, Image, cb, queue_size=1)
rospy.spin()
