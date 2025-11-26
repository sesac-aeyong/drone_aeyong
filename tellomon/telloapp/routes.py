# routes.py
import time, os
from flask import Blueprint, render_template, Response
from telloapp.profiler import mark, log_trace_to_csv

def create_routes(socketio, get_tello_server, disconnect_wifi):
    """
    Factory function to create the Blueprint with injected dependencies.
    Avoids circular imports by not importing telloapp objects directly.
    """
    bp = Blueprint('main', __name__)

    # Flask routes
    @bp.route('/')
    def index():
        return render_template('index.html')

    @bp.route('/video_feed')
    def video_feed():
        def generate():
            while True:
                frame = get_tello_server().get_current_frame_jpeg()
                if frame is not None:
                    
                    # ===========================================
                    # 📌 HTTP 전송 지연 기록 (profiler)
                    # ===========================================
                    ts = get_tello_server()
                    trace = ts.last_trace
                    if trace is not None:
                        # 마지막 단계: 클라이언트로 나가는 순간
                        mark(trace, "ts_http_send_ns")
                        log_trace_to_csv(trace) 
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.01)
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
        success = ts.connect_tello()
        if success:
            ts.start_streaming()
            socketio.emit('tello_status', {'connected': True, 'battery': ts.battery})
        else:
            socketio.emit('tello_status', {'connected': False})

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
        else:
            socketio.emit('tello_status', {'connected': False})

    @socketio.on('send_command')
    def handle_command(data):
        ts = get_tello_server()
        command = data.get('command')
        result = ts.execute_command(command)
        socketio.emit('command_response', result)

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
