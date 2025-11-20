import socket
import subprocess
import time


def list_wifi_networks():
    result = subprocess.run(["nmcli", "-t", "-f", "SSID", "dev", "wifi"], capture_output=True, text=True)
    ssids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return ssids


def disconnect_wifi():
    subprocess.run(['nmcli', 'dev', 'disconnect', 'wlan0'])


def connect_to_wifi(ssid, password=None):
    cmd = ["nmcli", "dev", "wifi", "connect", ssid]
    if password:
        cmd.extend(["password", password])
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode:
        return False
    return True


def get_current_ssid():
    try:
        result = subprocess.run(
            ["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"],
            capture_output=True,
            text=True,
            check=True
        )
        for line in result.stdout.strip().split("\n"):
            active, ssid = line.split(":")
            if active == "yes":
                return ssid
        return None
    except Exception as e:
        print("Error:", e)
        return None
    

def refresh_wifi_list() -> None:
    subprocess.run([
        'nmcli', 'device', 'wifi', 'list'
    ])
    

def connect_to_tello_wifi():
    """Tello WiFi에 자동으로 연결"""
    ssid = get_current_ssid()
    print('Current SSID:', ssid)
    if ssid and ssid.startswith('TELLO-'):
        return True
    
    print('Looking for Tello WiFi...')
    for attempt in range(10):
        refresh_wifi_list()
        networks = set(list_wifi_networks())
        for ssid in networks:
            if ssid.startswith('TELLO-'):
                print(f'Connecting to {ssid}...')
                if connect_to_wifi(ssid):
                    print(f'✅ Connected to {ssid}')
                    return True
                else:
                    print(f'❌ Failed to connect to {ssid}')
        print(f'Retry {attempt + 1}/10...')
        time.sleep(5)
    return False


def get_local_ip():
    """현재 사용중인 IP 주소 가져오기"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.168.10.1", 8889))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "Unknown"

