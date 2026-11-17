"""Webcam -> biểu cảm (CNN DeepFace + hình học khuôn mặt MediaPipe) -> Twist trên /cmd_vel."""
import os
import time

# phải đặt TRƯỚC khi import TensorFlow/MediaPipe: chỉ CPU, 1 luồng, ít log
os.environ.update(CUDA_VISIBLE_DEVICES='-1', TF_CPP_MIN_LOG_LEVEL='3', OMP_NUM_THREADS='1',
                  TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1', GLOG_minloglevel='2')
os.nice(5)                                       # nhường CPU cho Gazebo

import cv2
import numpy as np
import rclpy
from rcl_interfaces.msg import Parameter, ParameterDescriptor, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from deepface.models.demography import Emotion   # CNN biểu cảm của DeepFace (ảnh xám 48x48)
from geometry_msgs.msg import Twist
from irobot_create_msgs.action import Undock
from rclpy.action import ActionClient
from rclpy.node import Node
try:
    import mediapipe as mp                       # điểm mốc khuôn mặt -> đặc trưng hình học
except ImportError:
    mp = None

cv2.setNumThreads(1)

# thứ tự đầu ra của mô hình DeepFace ('disgust' được gộp vào 'angry' trong infer)
LAB = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
IDX = {e: i for i, e in enumerate(LAB)}
BROW, OPEN, WID, CORN, EYE = range(5)            # chỉ số đặc trưng hình học

# biểu cảm -> (linear.x [m/s], angular.z [rad/s])
CMD = {
    'happy':    (0.15,  0.5),   # vòng cung trái
    'sad':      (0.15, -0.5),   # vòng cung phải
    'neutral':  (0.0,   0.0),   # dừng
    'angry':    (-0.1,  0.0),   # lùi nhẹ
    'surprise': (0.0,   1.0),   # xoay tại chỗ
    'fear':     (0.1,   0.0),   # tiến nhẹ
}


def ramp(x, lo, hi):
    """0 khi x<=lo, 1 khi x>=hi, tuyến tính ở giữa."""
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def geometry(L):
    """Đặc trưng hình học (chuẩn hoá theo khoảng cách hai khoé mắt ngoài) từ 468 điểm mốc MediaPipe."""
    d = lambda a, b: np.linalg.norm(L[a] - L[b])
    D = d(33, 263)
    return np.array([
        ((L[159][1] - L[105][1]) + (L[386][1] - L[334][1])) / 2 / D,   # BROW: khoảng cách lông mày-mắt (nhỏ = cau mày)
        d(13, 14) / D,                                                 # OPEN: độ há miệng
        d(61, 291) / D,                                                # WID : độ rộng miệng (cười = rộng)
        ((L[61][1] + L[291][1]) / 2 - (L[13][1] + L[14][1]) / 2) / D,  # CORN: khoé miệng so với tâm (dương = trề xuống)
        ((L[145][1] - L[159][1]) + (L[374][1] - L[386][1])) / 2 / D,   # EYE : độ mở mắt
    ])


class EmotionControl(Node):
    def __init__(self):
        super().__init__('emotion_control')
        for k, v in dict(cam=0, fps=15, every=3, show=True, undock=True, alpha=0.5, hold=2, miss=2,
                         backup=True, calib=3.0, width=320).items():   # cam: số (webcam) hoặc chuỗi (file/URL)
            self.declare_parameter(k, v, ParameterDescriptor(dynamic_typing=True))
        g = lambda k: self.get_parameter(k).value
        self.every, self.show, self.alpha, self.hold, self.miss = g('every'), g('show'), g('alpha'), g('hold'), g('miss')
        self.calib = g('calib')
        cam, wd = g('cam'), int(g('width'))
        if str(cam).isdigit():                   # webcam: MJPG (YUYV bị treo qua usbip/WSL2), nhỏ + ít FPS
            self.cap = cv2.VideoCapture(int(cam), cv2.CAP_V4L2)
            for k, v in ((cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')), (cv2.CAP_PROP_FRAME_WIDTH, wd),
                         (cv2.CAP_PROP_FRAME_HEIGHT, wd * 3 // 4), (cv2.CAP_PROP_FPS, g('fps')), (cv2.CAP_PROP_BUFFERSIZE, 1)):
                self.cap.set(k, v)
        else:                                    # file video / URL
            self.cap = cv2.VideoCapture(cam)
        self.file, self.wd = not str(cam).isdigit(), wd
        self.fps = (self.cap.get(cv2.CAP_PROP_FPS) or g('fps')) if self.file else g('fps')   # video: phát đúng tốc độ gốc
        if not self.cap.isOpened():
            raise RuntimeError(f'Không mở được camera: {cam} (đã usbipd attach chưa?)')
        self.model = Emotion.EmotionClient().model
        self.fm = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) if mp else None
        if self.fm is None:
            self.get_logger().warn('Không có mediapipe -> chỉ dùng CNN DeepFace (kém chính xác)')
        self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.emo, self.box, self.n = 'neutral', None, 0
        self.p = np.zeros(7)                     # xác suất CNN đã làm mượt (EMA)
        self.gv, self.gb, self.cal, self.t_cal = None, None, [], None   # hình học (EMA) / mức nền / mẫu hiệu chuẩn
        self.cand, self.cnt, self.lost = 'neutral', 0, 0
        if g('undock'):                          # robot đang trên dock thì phải undock mới chạy
            self.ac = ActionClient(self, Undock, '/undock')
            if self.ac.wait_for_server(3.0):
                self.ac.send_goal_async(Undock.Goal())
            else:
                self.get_logger().warn('Không có action /undock (bỏ qua)')
        if g('backup'):                          # Create 3 chặn lùi quá giới hạn -> nới để Angry lùi liên tục
            self.sp = self.create_client(SetParameters, '/motion_control/set_parameters')
            self.tries = 0
            self.tmr = self.create_timer(1.0, self.tune)
        self.create_timer(1.0 / self.fps, self.step)   # đọc + hiển thị theo FPS camera
        self.create_timer(0.1, self.send)              # gửi lệnh đều 10 Hz

    def tune(self):
        """Đặt safety_override=backup_only cho /motion_control (thử mỗi giây tới khi dịch vụ sẵn sàng)."""
        self.tries += 1
        if not self.sp.service_is_ready():
            if self.tries > 180:
                self.tmr.cancel()
            return
        self.tmr.cancel()
        v = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value='backup_only')
        req = SetParameters.Request(parameters=[Parameter(name='safety_override', value=v)])
        self.sp.call_async(req).add_done_callback(lambda f: self.get_logger().info(
            f'safety_override=backup_only: {f.result().results[0].successful} {f.result().results[0].reason}'))

    def locate(self, img, gray):
        """Trả về (khung mặt x,y,w,h, đặc trưng hình học hoặc None); (None, None) nếu không thấy mặt."""
        if self.fm:
            r = self.fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not r.multi_face_landmarks:
                return None, None
            h, w = gray.shape
            L = np.array([(p.x * w, p.y * h) for p in r.multi_face_landmarks[0].landmark])
            x0, x1 = int(max(L[:, 0].min(), 0)), int(min(L[:, 0].max(), w))
            y0, y1 = int(max(L[10][1], 0)), int(min(L[152][1], h))        # trán -> cằm
            return (x0, y0, x1 - x0, y1 - y0), geometry(L)
        small = cv2.equalizeHist(cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
        fs = self.haar.detectMultiScale(small, 1.2, 5, minSize=(25, 25)) * 2   # dự phòng: Haar ở 1/2 độ phân giải
        return (tuple(max(fs, key=lambda f: f[2] * f[3])), None) if len(fs) else (None, None)

    def decide(self):
        """Hợp nhất hình học (phiếu chính, so với mức nền của chính người dùng) và CNN DeepFace (phiếu phụ)."""
        if self.gb is None:                      # chưa hiệu chuẩn/không có mediapipe: chỉ dùng CNN
            return LAB[int(np.argmax(self.p * np.array([1, 0, 1, 1, 1, 1, 1])))]
        g, b = self.gv, self.gb
        mo = ramp(g[OPEN] - b[OPEN], 0.05, 0.15)                          # há miệng
        s = {'surprise': mo,
             'angry': ramp(1 - g[BROW] / b[BROW], 0.06, 0.13) * (1 - mo),    # lông mày hạ thấp so với nền
             'happy': ramp(g[WID] / b[WID] - 1, 0.08, 0.18) * (1 - mo),      # miệng rộng ra
             'sad': ramp(g[CORN] - b[CORN], 0.015, 0.035) * (1 - mo),        # khoé miệng trề xuống
             'fear': ramp(g[BROW] / b[BROW] - 1, 0.08, 0.18) * ramp(g[EYE] / b[EYE] - 1, 0.08, 0.22)}
        f = {e: 0.75 * v + 0.25 * self.p[IDX[e]] for e, v in s.items()}   # CNN góp 25%
        e = max(f, key=f.get)
        return e if f[e] >= 0.4 else 'neutral'

    def infer(self, img):
        """Dò mặt + hình học, chạy CNN trên mặt, hợp nhất, cập nhật self.emo."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        box, gv = self.locate(img, gray)
        if box is None or box[2] < 20 or box[3] < 20:     # mất mặt vài lần liên tiếp -> dừng robot
            self.lost += 1
            if self.lost >= self.miss:
                self.emo, self.box, self.cand, self.cnt, self.p[:], self.gv = 'neutral', None, 'neutral', 0, 0, None
            return
        self.lost, self.box = 0, box
        x, y, w, h = box
        face = cv2.resize(gray[y:y + h, x:x + w], (48, 48)).astype(np.float32) / 255
        q = self.model(face[None], training=False).numpy()[0].copy()
        q[0], q[1] = q[0] + q[1], 0                       # nhíu mày bị mô hình đọc là 'disgust' -> gộp vào 'angry'
        self.p += self.alpha * (q - self.p)
        if gv is not None:
            self.gv = gv if self.gv is None else self.gv + self.alpha * (gv - self.gv)
        if self.fm and self.calib > 0 and self.gb is None:    # hiệu chuẩn: giữ mặt bình thường vài giây đầu
            if gv is not None:
                self.t_cal = self.t_cal or time.time()
                self.cal.append(gv)
                if time.time() - self.t_cal >= self.calib and len(self.cal) >= 5:
                    self.gb = np.median(self.cal, axis=0)
                    self.get_logger().info('Hiệu chuẩn xong: ' + ' '.join(
                        f'{n}={v:.3f}' for n, v in zip(('brow', 'open', 'wid', 'corner', 'eye'), self.gb)))
            return
        e = self.decide()
        self.cnt = self.cnt + 1 if e == self.cand else 1
        self.cand = e
        if self.cnt >= self.hold:                # đủ ổn định mới đổi biểu cảm
            self.emo = e

    def step(self):
        ok, img = self.cap.read()
        if not ok:
            if self.file:                        # hết video -> phát lại từ đầu (tránh robot giữ mãi lệnh cuối)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        if self.file and img.shape[1] != self.wd:  # video độ phân giải cao -> thu nhỏ cho nhẹ
            img = cv2.resize(img, (self.wd, img.shape[0] * self.wd // img.shape[1]), interpolation=cv2.INTER_AREA)
        self.n += 1
        if self.n % self.every == 0:
            self.infer(img)
        if self.show:
            if self.box is not None:
                x, y, w, h = self.box
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cal = self.fm is not None and self.calib > 0 and self.gb is None
            cv2.putText(img, 'GIU MAT BINH THUONG...' if cal else self.emo.upper(), (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 if cal else 0.9, (0, 200, 255) if cal else (0, 255, 0), 2)
            top = np.argsort(self.p)[::-1][:3]   # CNN: 3 xác suất cao nhất (để chẩn đoán)
            cv2.putText(img, 'cnn ' + ' '.join(f'{LAB[i][:3]}{self.p[i]*100:.0f}' for i in top),
                        (8, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            if self.gb is not None and self.gv is not None:   # hình học so với mức nền
                g, b = self.gv, self.gb
                cv2.putText(img, f'may{g[BROW]/b[BROW]:.2f} mieng{g[OPEN]-b[OPEN]:+.2f} rong{g[WID]/b[WID]:.2f}',
                            (8, img.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                cv2.putText(img, f'khoe{g[CORN]-b[CORN]:+.3f} mat{g[EYE]/b[EYE]:.2f}',
                            (8, img.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            cv2.imshow('emotion', img)
            k = cv2.waitKey(1)
            if k == ord('q'):
                rclpy.shutdown()
            elif k == ord('c'):                  # hiệu chuẩn lại (giữ mặt bình thường 3 giây)
                self.calib = self.calib or 3.0
                self.gb, self.cal, self.t_cal = None, [], None

    def send(self):
        m = Twist()
        m.linear.x, m.angular.z = CMD[self.emo]
        self.pub.publish(m)


def main():
    rclpy.init()
    n = EmotionControl()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.pub.publish(Twist())                       # dừng robot khi thoát
    n.cap.release()
    cv2.destroyAllWindows()
    n.destroy_node()


if __name__ == '__main__':
    main()
