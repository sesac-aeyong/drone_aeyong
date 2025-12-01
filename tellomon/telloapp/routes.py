# routes.py
import time, os
import cv2
import numpy as np
from flask import Blueprint, render_template, Response

def create_routes(socketio, get_tello_server, disconnect_wifi):
    """
    Factory function to create the Blueprint with injected dependencies.
    Avoids circular imports by not importing telloapp objects directly.
    """
    bp = Blueprint('main', __name__)

    # 1x1 검정 placeholder JPEG 생성
    _placeholder_img = np.zeros((1, 1, 3), dtype=np.uint8)
    _, _placeholder_buf = cv2.imencode('.jpg', _placeholder_img)
    PLACEHOLDER_JPEG = _placeholder_buf.tobytes()

    # --- 내부 유틸: features 보장 ------------------------------------------
    def _ensure_features(ts):
        if not hasattr(ts, 'features') or not isinstance(getattr(ts, 'features'), dict):
            # 기본값: 모두 OFF, 투명도 0.5
            ts.features = {'depth': False, 'pose': False, 'flow': False, 'alpha': 0.5}

    def _broadcast_features(ts):
        _ensure_features(ts)
        socketio.emit('feature_status', {
            'depth': bool(ts.features.get('depth', False)),
            'pose':  bool(ts.features.get('pose',  False)),
            'flow':  bool(ts.features.get('flow',  False)),
            'alpha': float(ts.features.get('alpha', 0.5)),
        })

    # Flask routes
    @bp.route('/')
    def index():
        return render_template('index.html')

    @bp.route('/video_feed')
    def video_feed():
        def generate():
            try:
                # 첫 프레임 즉시 전송 (start_response 오류 방지)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + PLACEHOLDER_JPEG + b'\r\n')
                while True:
                    frame = get_tello_server().get_current_frame_jpeg()
                    if frame is not None:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    else:
                        time.sleep(0.05)
                    time.sleep(0.01)
            except GeneratorExit:
                pass
        return Response(generate(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    # SocketIO events
    @socketio.on('connect')
    def handle_connect():
        print('Client connected')
        socketio.emit('connection_response', {'status': 'connected'})

    @socketio.on('disconnect')
    def handle_disconnect():
        print('Client disconnected')

    @socketio.on('connect_tello')
    def handle_connect_tello():
        ts = get_tello_server()
        _ensure_features(ts)
        success = ts.connect_tello()
        if success:
            ts.start_streaming()
            socketio.emit('tello_status', {'connected': True, 'battery': ts.battery})
            _broadcast_features(ts)  # index 초기 동기화
        else:
            socketio.emit('tello_status', {'connected': False})

    @socketio.on('get_tello_status')
    def handle_get_tello_status():
        ts = get_tello_server()
        _ensure_features(ts)
        socketio.emit('tello_status', {
            'connected': ts.is_connected,
            'battery': ts.battery
        })
        # 선택: 상태 요청 때 features도 같이 동기화하고 싶다면 아래 주석 해제
        # _broadcast_features(ts)

    @socketio.on('reconnect_tello')
    def handle_reconnect_tello():
        ts = get_tello_server()
        _ensure_features(ts)
        print("🔄 Reconnecting to Tello...")
        try: ts.stop_tracking()
        except: pass
        ts.stop_streaming()
        time.sleep(1)

        print("🔌 Disconnecting WiFi...")
        try: disconnect_wifi()
        except: pass
        time.sleep(2)

        success = ts.connect_tello()
        if success:
            ts.start_streaming()
            socketio.emit('tello_status', {'connected': True, 'battery': ts.battery})
            _broadcast_features(ts)
        else:
            socketio.emit('tello_status', {'connected': False})

    @socketio.on('send_command')
    def handle_command(data):
        ts = get_tello_server()
        command = data.get('command')
        result = ts.execute_command(command)
        socketio.emit('command_response', result)

    # ---------------------------
    # 🧩 Features: get/set (index 연동)
    # ---------------------------
    @socketio.on('get_features')
    def handle_get_features():
        ts = get_tello_server()
        _broadcast_features(ts)

    @socketio.on('set_features')
    def handle_set_features(data):
        ts = get_tello_server()
        _ensure_features(ts)

        # 입력 정규화
        try:
            depth = bool(data.get('depth', ts.features['depth']))
            pose  = bool(data.get('pose',  ts.features['pose']))
            flow  = bool(data.get('flow',  ts.features['flow']))
            alpha = float(data.get('alpha', ts.features['alpha']))
            # alpha 범위 보정
            if not (0.0 <= alpha <= 1.0):
                alpha = max(0.0, min(1.0, alpha))
        except Exception:
            # 잘못된 페이로드 → 현재 상태 재방송
            _broadcast_features(ts)
            return

        # race 방지
        try:
            with ts.lock:
                ts.features.update({'depth': depth, 'pose': pose, 'flow': flow, 'alpha': alpha})
        except AttributeError:
            ts.features.update({'depth': depth, 'pose': pose, 'flow': flow, 'alpha': alpha})

        # 사용자에게 즉시 반영됨을 알림
        _broadcast_features(ts)

        # 서버 로그(optional)
        try:
            ts.log("INFO", f"🔧 features -> depth:{depth} pose:{pose} flow:{flow} alpha:{alpha:.2f}")
        except Exception:
            pass

    # ---------------------------
    # 🎯 Target Selection (iid 또는 bbox 모두 허용)
    # ---------------------------
    @socketio.on('set_target')
    def handle_set_target(data):
        ts = get_tello_server()

        target_identity_id = data.get('target_identity_id')  # None 가능
        target_class = data.get('class')
        target_bbox  = data.get('bbox')

        # ▶ IID 필수: 없거나 0/음수면 거부
        try:
            iid_ok = (target_identity_id is not None) and (int(target_identity_id) > 0)
        except Exception:
            iid_ok = False
        if not iid_ok:
            socketio.emit('target_response', {
                'ok': False,
                'message': 'ID(??) 항목은 선택할 수 없습니다. 갤러리 2장 저장 후 ID가 부여되어야 합니다.'
            })
            return

        # 서버 상태에 저장
        try:
            with ts.lock:
                ts.target_identity_id = target_identity_id
                ts.target_class = target_class
                ts.target_bbox  = target_bbox
        except AttributeError:
            ts.target_identity_id = target_identity_id
            ts.target_class = target_class
            ts.target_bbox  = target_bbox
        try:
            ts.log("INFO", f"🎯 Target set → iid={target_identity_id}, class={target_class}, bbox={target_bbox}")
        except Exception:
            pass

        socketio.emit('target_response', {
            'ok': True,
            'target_identity_id': target_identity_id,
            'class': target_class,
            'bbox': target_bbox
        })

    # ---------------------------
    # 🚀 Start Tracking
    # ---------------------------
    @socketio.on('start_tracking')
    def handle_start_tracking():
        ts = get_tello_server()
        # TelloWebServer.start_tracking 내부에서
        # ▶ iid 필수로만 시작
        success = ts.start_tracking()
        socketio.emit('tracking_status', {
            'is_tracking': success,
            'target_identity_id': ts.target_identity_id,
            'class': ts.target_class,
        })

    # ---------------------------
    # 🛑 Stop Tracking
    # ---------------------------
    @socketio.on('stop_tracking')
    def handle_stop_tracking():
        ts = get_tello_server()
        ts.stop_tracking()
        socketio.emit('tracking_status', {
            'is_tracking': False,
            'target_identity_id': ts.target_identity_id,
            'class': ts.target_class,
        })

    # ---------------------------
    # ✔️ Shutdown Server
    # ---------------------------
    @socketio.on('shutdown_server')
    def handle_shutdown():
        ts = get_tello_server()
        try:
            ts.stop_tracking()
            ts.stop_streaming()
        except Exception:
            pass
        socketio.emit('log_message', {
            'timestamp': time.strftime('%H:%M:%S'),
            'level': 'WARNING',
            'message': '서버를 종료합니다…'
        })
        os._exit(0)

    return bp
