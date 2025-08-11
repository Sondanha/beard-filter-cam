import os
import cv2
import numpy as np
from datetime import datetime  # ==== [REC] 파일명 타임스탬프

# -----------------------------
# 필터 (이전 단계 동일)
# -----------------------------
def apply_filter(frame, mode: str):
    if mode == "NONE":
        return frame
    if mode == "GRAY":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mode == "BLUR":
        return cv2.GaussianBlur(frame, (11, 11), 0)
    if mode == "EDGE":
        edge = cv2.Canny(frame, 80, 160)
        return cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    if mode == "SEPIA":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        sepia = cv2.transform(frame, kernel)
        return np.clip(sepia, 0, 255).astype(np.uint8)
    if mode == "HSV_DEMO":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.add(s, 40)
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return frame

# -----------------------------
# PNG 스티커 로딩/리사이즈
# -----------------------------
STICKER_FILES = {
    "style1":"./images/s.png",
    "style2":"./images/s2.png",
    "style3":"./images/s3.png",
    "style4":"./images/s4.png",
    "style5":"./images/s5.png",
    "style6":"./images/s6.png",
}

# 원본 PNG를 1회 로드하여 보관
_STICKER_ORIG = {}

def _read_png_bgra(path: str) -> np.ndarray:
    """PNG를 BGRA(알파 포함)로 로드. 알파 없으면 알파 채널 추가."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # 4채널이면 그대로, 3채널이면 BGR
    if img is None:
        raise FileNotFoundError(f"스티커 파일을 찾을 수 없습니다: {path}")
    if img.ndim == 2:
        # 그레이 PNG인 경우 3채널로 변환
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        # 알파 없는 PNG → 알파 255 추가
        b, g, r = cv2.split(img)
        a = np.full_like(b, 255)
        img = cv2.merge([b, g, r, a])
    return img

def get_sticker_bgra(kind: str, scale: float = 1.0) -> np.ndarray:
    """kind에 해당하는 PNG를 읽어와서 scale 비율로 리사이즈해 반환."""
    if kind not in STICKER_FILES:
        kind = "style1"
    path = STICKER_FILES[kind]

    # 실행 파일 기준 상대경로 안전 처리
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, path)

    if kind not in _STICKER_ORIG:
        _STICKER_ORIG[kind] = _read_png_bgra(full_path)

    orig = _STICKER_ORIG[kind]
    if scale == 1.0:
        return orig

    h, w = orig.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(orig, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    return resized

# -----------------------------
# 알파 블렌딩 (BGRA → BGR)
# -----------------------------
def overlay_bgra(base_bgr: np.ndarray, sticker_bgra: np.ndarray, center_xy):
    h, w = base_bgr.shape[:2]
    sh, sw = sticker_bgra.shape[:2]
    cx, cy = center_xy

    # 좌상단/우하단
    x1 = cx - sw // 2
    y1 = cy - sh // 2
    x2 = x1 + sw
    y2 = y1 + sh

    # 화면 밖 완전 이탈 시 패스
    if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
        return

    # 클리핑
    rx1 = max(x1, 0)
    ry1 = max(y1, 0)
    rx2 = min(x2, w)
    ry2 = min(y2, h)

    sx1 = rx1 - x1
    sy1 = ry1 - y1
    sx2 = sx1 + (rx2 - rx1)
    sy2 = sy1 + (ry2 - ry1)

    roi = base_bgr[ry1:ry2, rx1:rx2]
    sticker_roi = sticker_bgra[sy1:sy2, sx1:sx2]

    sb, sg, sr, sa = cv2.split(sticker_roi)
    alpha = (sa.astype(np.float32) / 255.0)[:, :, None]  # HxWx1
    sticker_rgb = cv2.merge([sb, sg, sr]).astype(np.float32)
    roi_f = roi.astype(np.float32)

    blended = alpha * sticker_rgb + (1 - alpha) * roi_f
    base_bgr[ry1:ry2, rx1:rx2] = np.clip(blended, 0, 255).astype(np.uint8)

# -----------------------------
# HUD
# -----------------------------
def draw_hud(img, filter_mode: str, sticker_mode: str, count: int, scale: float, rec: bool):
    lines = [
        f"[Filter] {filter_mode}   [Sticker] {sticker_mode} x{scale:.2f} (count={count})",
        "      Choose a great beard!",
        "N/G/B/E/S/H =Filters| beard style 1-6",
        "Click=place|U=undo|C=clear|[,]size|ESC=quit",
    ]
    y = 28
    for i, text in enumerate(lines):
        cv2.putText(
            img,
            text,
            (10, y + i*28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
    # ==== [REC] 좌상단 REC 표시
    if rec:
        cv2.putText(img, "REC", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(img, (60, 15), 6, (0, 0, 255), -1, cv2.LINE_AA)

# -----------------------------
# 메인
# -----------------------------
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # 0번을 DirectShow 백엔드로 오픈
    # cv2.VideoCapture(index, backend) : 카메라 장치 열기
    
    # VideoCapture
    # 읽기: get
      # w  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
      # fps = cap.get(cv2.CAP_PROP_FPS)
    # 쓰기: set
      # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
      # cap.set(cv2.CAP_PROP_FPS, 30)
    # set()은 요청사항일 뿐이므로 get()으로 재확인

    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다.")
        return

    window_name = "OpenCV FilterCam - PNG Stickers"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # 표시 창 생성

    # 적용할 기능 정의 및 초기값 설정
    filter_mode = "NONE" 
    sticker_mode = "style1"
    sticker_scale = 1.0
    stickers = []  # 누적될 스티커를 위한 리스트

    # ==== [REC] VideoWriter 준비 (지연 초기화)
    # VideoWriter는 프레임 크기를 알아야 함. 
    writer = None 
    out_path = f"record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    
    # FPS: 장치에서 읽고 실패 시 30으로 대체
    # cap.get(CAP_PROP_FPS)
    # 실제 FPS는 프레임마다 시간 측정으로 계산하는 게 명확
    # 고해상도일수록 처리량↑ 지연↑. 실시간 파이프라인은 해상도↓/FPS↑의 균형이 중요
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # 코덱 선택
      # 1. mp4v : MP4 컨테이너
      # 2. XVID : AVI 컨테이너에서 흔함(용량↑)
      # 3. MJPG : 무손실에 가까운 품질/용량↑↑, 편집/디버깅용
    # 코덱별 압축 효율 차이 큼
    # 동일한 10초 녹화에서 mp4v vs XVID vs MJPG 파일 크기 비교.

    is_recording = False  # HUD에 REC 표시용 플래그

    # 마우스의 이벤트 읽기
    def on_mouse(event, x, y, flags, param):
        
        nonlocal stickers, sticker_mode, sticker_scale
        
        if event == cv2.EVENT_LBUTTONDOWN:
            stickers.append({
                "kind": sticker_mode,
                "pos": (x, y),
                "scale": sticker_scale
            })

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        ok, frame = cap.read() # 카메라에서 BGR 프레임 읽기
        # cap.read() : 한 프레임 읽기 (BGR np.ndarray)
        if not ok:
            print("❌ 프레임을 읽을 수 없습니다.")
            break

        frame = cv2.flip(frame, 1)
        # cv2.flip(img, 1) : 좌우 반전 (0: 상하, 1: 좌우, -1: 둘 다)

        # 1) 필터
        filtered = apply_filter(frame, filter_mode) # 현재 선택된 필터 적용

        # 2) 스티커 합성
        for st in stickers: # (4) 누적된 스티커들을
            sticker_img = get_sticker_bgra(st["kind"], st["scale"]) # PNG(BGRA) 로드/리사이즈
            overlay_bgra(filtered, sticker_img, st["pos"])          # 알파 블렌딩으로 합성

        # ==== [REC] writer 최초 1회 초기화 (사이즈 확정 후)
        if writer is None:
            h, w = filtered.shape[:2]
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            # cv2.VideoWriter(path, fourcc, fps, (w,h)) : 동영상 파일 기록

            if not writer.isOpened():
                print("⚠️ VideoWriter 초기화 실패. 파일 저장이 비활성화됩니다.")
            else:
                is_recording = True

        # HUD (REC 표시 포함)
        draw_hud(filtered, filter_mode, sticker_mode, len(stickers), sticker_scale, is_recording)

        # ==== [REC] 프레임 기록 (필터+스티커+HUD 모두 포함)
        if writer is not None and writer.isOpened():
            writer.write(filtered)

        # cv2.imshow는 즉시 반영되지 않고 
        # waitKey()가 GUI 이벤트(윈도우 갱신, 키 입력)를 처리x
        cv2.imshow(window_name, filtered)
        key = cv2.waitKey(1) & 0xFF
        # 

        if key == 27:  # ESC
            break
        # 필터
        elif key in (ord('n'), ord('N')):
            filter_mode = "NONE"
        elif key in (ord('g'), ord('G')):
            filter_mode = "GRAY"
        elif key in (ord('b'), ord('B')):
            filter_mode = "BLUR"
        elif key in (ord('e'), ord('E')):
            filter_mode = "EDGE"
        elif key in (ord('s'), ord('S')):
            filter_mode = "SEPIA"
        elif key in (ord('h'), ord('H')):
            filter_mode = "HSV_DEMO"

        # 스티커 선택
        elif key == ord('1'):
            sticker_mode = "style1"
        elif key == ord('2'):
            sticker_mode = "style2"
        elif key == ord('3'):
            sticker_mode = "style3"
        elif key == ord('4'):
            sticker_mode = "style4"
        elif key == ord('5'):
            sticker_mode = "style5"
        elif key == ord('6'):
            sticker_mode = "style6"

        # 편의 기능
        elif key in (ord('c'), ord('C')):
            stickers.clear()
        elif key in (ord('u'), ord('U')):
            if stickers:
                stickers.pop()

        # 크기 조절
        elif key == ord('['):
            sticker_scale = max(0.1, round(sticker_scale - 0.1, 2))
        elif key == ord(']'):
            sticker_scale = min(5.0, round(sticker_scale + 0.1, 2))

    # ==== [REC] 자원 정리
    cap.release()
    if writer is not None:
        writer.release()
        print(f"💾 저장 완료: {out_path}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
