import subprocess
import time
import cv2
import numpy as np
from ping3 import ping

"""
pip install ping3 
"""

TELLO_PREFIX = 'TELLO-'

class TelloConnection:
    """
    find and connect to tello 
    """
    # def __init__(self):
    #     self.tello = Tello()

    @property
    def connection_alive(self) -> bool:
        """
        pings tello and returns true when successful.
        """
        # return True
        if ping('192.168.10.1', 1):
            return True
        return False

    def _run_command(self, command):
        r = subprocess.run(command, capture_output=True)
        return r

    def _refresh_wifi_list(self) -> subprocess.CompletedProcess:
        cmd = ['nmcli', 'device', 'wifi', 'list']
        return self._run_command(cmd)

    def _list_wifi_networks(self) -> set:
        """
        returns the set of available SSIDs 
        """
        cmd = ['nmcli', '-t', '-f', 'SSID', 'dev', 'wifi']
        result = self._run_command(cmd)
        if result.returncode != 0:
            return # perhaps handle errors
        stdout = result.stdout.decode().strip()
        return set(stdout.split())
    
    def _disconnect_wifi(self) -> subprocess.CompletedProcess:
        cmd = ['nmcli', 'dev', 'disconnect', 'wlan0']
        return self._run_command(cmd)

    def _get_current_ssid(self):
        cmd = ['nmcli', '-t', '-f', 'active,ssid', 'dev', 'wifi']
        result = self._run_command(cmd)
        stdout = result.stdout.decode().strip()
        for line in stdout.split('\n'):
            active, ssid = line.split(':', 1)
            if active == 'yes':
                return ssid
        return None
    
    def _connect_wifi(self, ssid) -> bool:
        """
        returns true on success 
        """
        cmd = ['nmcli', 'dev', 'wifi', 'connect', ssid]
        result = self._run_command(cmd)
        return True if result.returncode == 0 else False
    
    def connect_to_tello(self, tello_id:str = None) -> bool:
        """
        automatically connects to first SSID found with TELLO_PREFIX
        supply tello_id to connect to specific tello.

        returns true on success 
        """
        if tello_id:
            tello_id = TELLO_PREFIX + tello_id
        print('[Telloconn] connecting to tello')
        ssid = self._get_current_ssid()
        if ssid and ssid.startswith(TELLO_PREFIX):
            if not tello_id or (tello_id == ssid):
                if self.connection_alive:
                    print('[Telloconn] existing connection is alive!')
                    return True
            self._disconnect_wifi()
        
        print('[Telloconn] looking for tello...')
        for _ in range(30):
            self._refresh_wifi_list()
            if tello_id:
                print('Trying to connect to', tello_id)
                if self._connect_wifi(tello_id):
                    return True
            else:
                ssids = self._list_wifi_networks()
                for ssid in ssids:
                    if not ssid.startswith(TELLO_PREFIX):
                        continue
                    print(f'[Telloconn] trying to connect to {ssid}')
                    if self._connect_wifi(ssid):
                        return True

            time.sleep(3)
        return False
    


    

# # tello = TelloBrain()
# # tello._refresh_wifi_list()
# # tello._list_wifi_networks()
# # tello._disconnect_wifi()
# # print(tello.connect_to_tello())
# telloconn = TelloConnection()
# telloconn.connect_to_tello()

# tello = Tello()
# tello.connect()
# print(tello.get_battery())