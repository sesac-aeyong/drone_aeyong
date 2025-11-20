# routes.py
import time
from flask import Blueprint, render_template, Response

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

    @socketio.on('set_target')
    def handle_set_target(data):
        ts = get_tello_server()
        target_track_id = data.get('track_id')
        target_class = data.get('class')
        target_bbox = data.get('bbox')

        ts.target_track_id = target_track_id
        ts.target_class = target_class
        ts.target_bbox = target_bbox
        ts.log("INFO", f"🎯 Target set to: ID {target_track_id} ({target_class}), bbox: {target_bbox}")

        socketio.emit('target_response', {
            'track_id': target_track_id,
            'class': target_class,
            'bbox': target_bbox
        })

    @socketio.on('start_tracking')
    def handle_start_tracking():
        ts = get_tello_server()
        if ts.target_track_id is not None:
            success = ts.start_tracking()
            socketio.emit('tracking_status', {
                'is_tracking': success,
                'target_track_id': ts.target_track_id,
                'target_class': ts.target_class
            })
        else:
            socketio.emit('tracking_status', {
                'is_tracking': False,
                'message': 'No target selected'
            })

    @socketio.on('stop_tracking')
    def handle_stop_tracking():
        ts = get_tello_server()
        ts.stop_tracking()
        socketio.emit('tracking_status', {'is_tracking': False})

    return bp
