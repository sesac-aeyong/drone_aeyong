import subprocess
import sys
import time
from telloapp import create_app, disconnect_wifi, connect_to_tello_wifi, get_local_ip, get_tello_server, cleanup_and_exit


if __name__ == '__main__':
    app = create_app()
    tello_server = get_tello_server()
    print("🔧 Cleaning up UDP port 11111...")
    try:
        subprocess.run(['fuser', '-k', '11111/udp'], 
                      stderr=subprocess.DEVNULL, 
                      stdout=subprocess.DEVNULL,
                      timeout=2)
        time.sleep(1)
        print("✅ Port cleaned")
    except:
        print("⚠️ Could not clean port (may not be in use)")

    if '--auto-connect' in sys.argv or '-a' in sys.argv:
        print("\n🔍 Auto-connecting to Tello WiFi...")
        disconnect_wifi()
        time.sleep(1)
        if connect_to_tello_wifi():
            print("✅ Auto-connected to Tello WiFi")
        else:
            print("⚠️ Auto-connect failed")
        time.sleep(2)

    local_ip = get_local_ip()
    print("\n" + "="*50)
    print(f"🚁 Tello Web Server Started!")
    print("="*50 + "\n")

    try:
        tello_server.socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        cleanup_and_exit()
