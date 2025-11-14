import subprocess
import time
import cv2
from djitellopy import Tello
import numpy as np
from reid_test import run, close

from object_detection_post_process import draw_detections, extract_detections, inference_result_handler, find_best_matching_detection_index, id_to_color, draw_detection


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
    
def connect_to_tello():
    ssid = get_current_ssid()
    print('cur ssid:',ssid)
    if ssid and ssid.startswith('TELLO-'): # already on tello
        return True
    
    print('looking for tello...')
    for _ in range(10):

        networks = set(list_wifi_networks())
        # for ssid in networks:
        #     print(ssid)
        for ssid in networks:
            if ssid.startswith('TELLO-'):
#                if ssid == 'TELLO-5FD7AB': # blue tello 
#                    continue
                print('connecting to',ssid)
                return connect_to_wifi(ssid)
        print('try again')
        time.sleep(5)
    return False


def main():
    disconnect_wifi()
    if not connect_to_tello():
        print('failed to connect to a tello.')
        close()
        return
    tello = Tello()
    tello.connect()
    print('Battery:', tello.get_battery())
    # tello.initiate_throw_takeoff()
    tello.streamoff()
    time.sleep(1)
    tello.streamon()
    time.sleep(2)
    for _ in range(20):  # try up to 5 times
        try:
            feed = tello.get_frame_read()
            break
        except ValueError:
            print("Stream not ready, retrying...")
            tello.streamon()
            time.sleep(3)

    
    if feed is None:
        raise RuntimeError("Failed to open Tello video stream")
    # feed = tello.get_frame_read()
    try:
        while True:
            frame = feed.frame # comes as 960 * 720
            frame = cv2.resize(frame, (640, 480))
            # print('wtf', frame.shape)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fh, fw = frame.shape[:2]
            dets, dep = run(frame)
            dep = cv2.resize(dep, (fw, fh))

            # rets.append((track_id, labels[classes[best_idx]], track.score, xmin, ymin, xmax, ymax))

            for det in dets:
                tid, label, score, x1, y1, x2, y2 = det
                draw_detection(
                    frame,
                    [x1, y1, x2, y2],
                    [label, f"ID {tid}"],  # second element used as bottom_text when track=True
                    score=score * 100.0,
                    color=(255, 255, 255),
                    track=True
                )

            dep_norm = cv2.normalize(dep, None, 0, 255, cv2.NORM_MINMAX)
            dep_color = cv2.cvtColor(dep_norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            view = np.hstack((frame, dep_color))
            # print('ok')
            # view = np.hstack((frame, dep))
            # cv2.imshow('tello feed', frame)
            # cv2.imshow('depth', dep)
            # cv2.imshow
            cv2.imshow('result', view)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    # except Exception as e:
    #     print(e)

    tello.streamoff()
    close()
    
    # time.sleep(5)
    # tello.
    # tello.turn_motor_on()



if __name__ == "__main__":
    main()
