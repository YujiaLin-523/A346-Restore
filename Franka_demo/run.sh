#!/bin/bash

# ================= 配置区 =================
# 如果你的 catkin 工作空间在其他位置，请修改这里
WORKSPACE_SETUP_PATH=~frankapy/catkin_ws/devel/setup.zsh
# =========================================

# 1. 检查是否已经 Source 了 ROS 环境
if [ -z "$ROS_ROOT" ]; then
    echo "[INFO] Loading ROS setup..."
    if [ -f "$WORKSPACE_SETUP_PATH" ]; then
        source "$WORKSPACE_SETUP_PATH"
    else
        echo "[WARN] Cannot find workspace setup at $WORKSPACE_SETUP_PATH"
        echo "       Trying /opt/ros/noetic/setup.zsh instead..."
        source /opt/ros/noetic/setup.zsh || source /opt/ros/noetic/setup.bash
    fi
fi

# 2. 检查 Python 依赖
echo "[INFO] Checking dependencies..."
# 简单的静默检查，如果失败会报错
python3 -c "import frankapy; import cv2; import yaml" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[ERROR] Missing python dependencies (frankapy, opencv-python, pyyaml)."
    exit 1
fi

# 3. 运行主程序
echo "[INFO] Starting Camera Tool Suite..."
echo "----------------------------------------"
python3 main.py
