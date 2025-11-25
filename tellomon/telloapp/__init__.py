import signal
import sys
from flask import Flask
from flask_socketio import SocketIO
from .app_tools import *

socketio = SocketIO()
tello_server = None


def get_tello_server():
    global tello_server
    if tello_server is None:
        from .tello_web_server import TelloWebServer
        tello_server = TelloWebServer(socketio)
    return tello_server


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'tello_secret_key'
    socketio.init_app(app)

    from .routes import create_routes
    bp = create_routes(socketio, get_tello_server, disconnect_wifi)
    app.register_blueprint(bp)
    
    return app
    

def cleanup_and_exit():
    """완전한 정리 후 종료"""
    print("\n🛑 Cleaning up...")
    
    global tello_server
    
    try:
        if tello_server.is_tracking:
            tello_server.stop_tracking()
            time.sleep(0.5)
    except:
        pass
    
    try:
        if tello_server.is_streaming:
            tello_server.stop_streaming()
            time.sleep(0.5)
    except:
        pass
    
    try:
        tello_server.cleanup()
    except:
        pass
    
    try:
        if tello_server.tello:
            if hasattr(tello_server.tello, 'background_frame_read'):
                if tello_server.tello.background_frame_read:
                    try:
                        tello_server.tello.background_frame_read.stop()
                        print("✅ Background frame read stopped")
                    except:
                        pass
            
            try:
                tello_server.tello.streamoff()
                time.sleep(1)
                print("✅ Stream off")
            except:
                pass
            
            try:
                tello_server.tello.end()
                print("✅ Tello connection ended")
            except:
                pass
    except:
        pass
    
    try:
        print("🔧 Killing processes on UDP port 11111...")
        subprocess.run(['fuser', '-k', '11111/udp'], 
                      stderr=subprocess.DEVNULL, 
                      stdout=subprocess.DEVNULL,
                      timeout=2)
        time.sleep(1)
        print("✅ UDP port released")
    except:
        pass
    
    print("✅ Cleanup complete")


def signal_handler(sig, frame):
    cleanup_and_exit()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

