# routes.py
import time, os
import cv2
import numpy as np
from flask import Blueprint, render_template, Response, request, jsonify #🚨

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
                        time.sleep(0.05)  # 프레임 없으면 잠시 대기
                    time.sleep(0.01)
            except GeneratorExit:
                # 클라이언트 연결 종료 시 정상 종료
                pass
        return Response(generate(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @bp.route('/api/undistort', methods=['POST'])
    def api_undistort(): #🚨depth map 껐다키기
        """JSON {"enable": true/false} 를 받아 왜곡보정 토글"""
        data = request.get_json(force=True, silent=True) or {}
        enable = bool(data.get("enable", False))
        s = get_tello_server()
        s.set_undistort(enable)
        socketio.emit('undistort_status', {"enable": enable})
        return jsonify({"ok": True, "enable": enable})

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
        success = ts.connect_tello()
        if success:
            ts.start_streaming()
            socketio.emit('tello_status', {'connected': True, 'battery': ts.battery})
            socketio.emit('undistort_status', {"enable": ts.use_undistort}) #🚨
        else:
            socketio.emit('tello_status', {'connected': False})

    @socketio.on('get_tello_status')
    def handle_get_tello_status():
        ts = get_tello_server()
        socketio.emit('tello_status', {'connected': ts.is_connected, 'battery': ts.battery})
        socketio.emit('undistort_status', {"enable": ts.use_undistort}) #🚨

    @socketio.on('reconnect_tello')
    def handle_reconnect_tello():
        ts = get_tello_server()
        print("🔄 Reconnecting to Tello...")
        ts.stop_tracking()
        ts.stop_streaming()
        time.sleep(1)

        print("🔌 Disconnecting WiFi...")
        disconnect_wifi()
        time.sleep(2)

        success = ts.connect_tello()
        if success:
            ts.start_streaming()
            socketio.emit('tello_status', {'connected': True, 'battery': ts.battery})
            socketio.emit('undistort_status', {"enable": ts.use_undistort}) #🚨
        else:
            socketio.emit('tello_status', {'connected': False})

    @socketio.on('send_command')
    def handle_command(data):
        ts = get_tello_server()
        command = data.get('command')
        result = ts.execute_command(command)
        socketio.emit('command_response', result)

    @socketio.on('set_undistort')
    def set_undistort_event(data): #🚨depth map 껐다키기
        # data: {"enable": true/false}
        enable = bool(data.get("enable", False))
        s = get_tello_server()
        s.set_undistort(enable)
        socketio.emit('undistort_status', {"enable": enable})

    # ---------------------------
    # 🎯 Target Selection (identity_id only)
    # ---------------------------
    @socketio.on('set_target')
    def handle_set_target(data):
        ts = get_tello_server()

        # 웹에서 무조건 'target_identity_id' 로 보냄
        target_identity_id = data.get('target_identity_id')
        target_class = data.get('class')
        target_bbox  = data.get('bbox')

        if target_identity_id is None:
            socketio.emit('target_response', {
                'ok': False,
                'error': 'target_identity_id is required'
            })
            return

        ts.target_identity_id = target_identity_id
        ts.target_class = target_class
        ts.target_bbox  = target_bbox
        ts.log("INFO", f"🎯 Target identity set: {target_identity_id} ({target_class}), bbox={target_bbox}")

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
        if ts.target_identity_id is not None:
            success = ts.start_tracking()
            socketio.emit('tracking_status', {
                'is_tracking': success,
                'target_identity_id': ts.target_identity_id,
                'class': ts.target_class,
            })
        else:
            socketio.emit('tracking_status', {
                'is_tracking': False,
                'message': 'No identity selected'
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
        # 그 외(uvicorn/gevent 등) 안전 종료가 어려우면 프로세스 종료
        os._exit(0)

    return bp
