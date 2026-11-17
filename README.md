# emotion_TTB

Nhận diện biểu cảm khuôn mặt qua webcam và điều khiển TurtleBot 4 trong Gazebo qua topic `/cmd_vel`.
Môi trường: Ubuntu 22.04 (WSL2), ROS 2 Humble.

## Cách hoạt động

Webcam (OpenCV) -> điểm mốc khuôn mặt MediaPipe + CNN biểu cảm của DeepFace -> hợp nhất -> bảng ánh xạ -> `/cmd_vel` (`geometry_msgs/msg/Twist`).

| Biểu cảm | Hành động | linear.x (m/s) | angular.z (rad/s) |
|---|---|---|---|
| Happy | Cung tròn sang trái | 0.15 | 0.5 |
| Sad | Cung tròn sang phải | 0.15 | -0.5 |
| Neutral | Dừng | 0 | 0 |
| Angry | Lùi nhẹ | -0.1 | 0 |
| Surprise | Xoay tại chỗ | 0 | 1.0 |
| Fear | Tiến nhẹ | 0.1 | 0 |

## Cài đặt

Python (giữ đúng phiên bản, đừng chạy `pip install deepface` trần vì nó nâng numpy lên 2.x và kéo bản OpenCV không có cửa sổ):

```bash
pip install --user numpy==1.24.4 protobuf==4.25.9 tensorflow==2.15.1 tf-keras==2.15.1 keras==2.15.0 mediapipe==0.10.14
pip install --user --no-deps deepface==0.0.93
pip install --user gdown pandas Pillow tqdm requests Flask gunicorn fire mtcnn retina-face
pip uninstall -y opencv-contrib-python-headless
pip install --user --force-reinstall --no-deps opencv-python==4.10.0.84 opencv-contrib-python==4.10.0.84
```

Kiểm tra OpenCV phải là 4.10.0 và numpy là 1.24.4:

```bash
python3 -c 'import cv2, numpy; print(cv2.__version__, numpy.__version__)'
```

ROS 2 và mô phỏng:

```bash
sudo apt install ros-humble-turtlebot4-simulator ros-humble-turtlebot4-desktop
```

Build:

```bash
mkdir -p ~/tb4_ws/src && cd ~/tb4_ws/src
git clone https://github.com/boanuen/emotion_TTB.git
cp -r emotion_TTB/tb4_emotion_control .
cd ~/tb4_ws && source /opt/ros/humble/setup.bash && colcon build --packages-select tb4_emotion_control
```

## Webcam trên WSL2

Trong PowerShell (Admin) của Windows, gắn webcam vào WSL2 mỗi lần khởi động lại WSL2:

```powershell
usbipd list
usbipd bind --busid <BUSID>
usbipd attach --wsl --busid <BUSID>
```

Kiểm tra trong WSL2: `ls /dev/video*`.

## Chạy

```bash
source /opt/ros/humble/setup.bash && source ~/tb4_ws/install/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE=~/fastdds_no_shm.xml
ros2 launch tb4_emotion_control emotion.launch.py sim:=full cam:=0
```

Tham số launch:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `sim` | `full` | `full`: Gazebo đầy đủ của TurtleBot 4 (robot tự undock). `lite`: Gazebo nhẹ từ package `tb4_voice_nav` (cần package đó). `none`: chỉ chạy node |
| `cam` | `0` | Chỉ số webcam, hoặc đường dẫn file video, hoặc URL |
| `delay` | `50` | Số giây chờ trước khi chạy node nhận diện (để Gazebo lên trước) |
| `calib` | `3.0` | Số giây giữ mặt bình thường lúc đầu để lấy mức nền. 0 là tắt |
| `width` | `320` | Độ rộng khung hình camera |
| `gpu` | `true` | Chỉ với `sim:=lite`: cửa sổ Gazebo GUI vẽ bằng GPU. `false` là vẽ bằng CPU |

Khi chạy, giữ mặt bình thường trong 3 giây đầu (cửa sổ hiện chữ vàng cam). Phím `c` để hiệu chuẩn lại, `q` để thoát.
Dòng chữ vàng dưới cửa sổ là số đo hiện tại: `may` (lông mày so với nền), `mieng` (độ há miệng), `rong` (độ rộng miệng), `khoe` (khoé miệng), `mat` (độ mở mắt).

## Cấu trúc

```
tb4_emotion_control/
  package.xml, setup.py, setup.cfg
  launch/emotion.launch.py
  tb4_emotion_control/emotion_control_node.py
```
