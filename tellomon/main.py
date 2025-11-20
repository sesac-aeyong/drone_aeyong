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




# """

# TODO:: Tello ASCII art? xD 

# """

# import sys
# import threading
# import time
# import cv2
# import numpy as np
# from common.toolbox import load_json_file
# from hailo_inference import HailoRun
# from tello import TelloConnection
# from yolo_tools import draw_detection
# from djitellopy import Tello
# from settings import settings as S


# def _show(frame, dets, yolobox) -> bool:
#     if 'show' in sys.argv[1:]:
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         for det in dets:    
#             tid, label, score, x1, y1, x2, y2 = det
#             draw_detection(
#                 frame,
#                 [x1, y1, x2, y2],
#                 [label, f"ID {tid}"],  # second element used as bottom_text when track=True
#                 score=score * 100.0,
#                 color=(255, 255, 255),
#                 track=True
#             )
    
#         if 'yolo' in sys.argv[1:]:
#             for box in yolobox:
#                 x1, y1, x2, y2 = map(int, box)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 1)

#         cv2.imshow('out', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             return True
#     return False


# class TelloMon():
#     def __init__(self):
#         self.conn = TelloConnection()
#         self.hailo = HailoRun()
#         tconn = threading.Thread(target=self.conn.connect_to_tello, args=(S.tello_id,))  
#         thailo = threading.Thread(target=self.hailo.load)

#         tconn.start()
#         thailo.start()

#         tconn.join()
#         thailo.join()
        
#         self.tello = Tello()
        
#         self.tello.connect()
#         print('Tello battery:', self.tello.get_battery())
#         print('Tello temp:', self.tello.get_highest_temperature())
#         # self.tello.initiate_throw_takeoff()
#         if not self.tello.stream_on:
#             self.tello.streamon()

#         feed = None
#         for _ in range(5):  # try up to 5 times
#             try:
#                 feed = self.tello.get_frame_read()
#                 break
#             except ValueError:
#                 print("Stream not ready, retrying...")
#                 self.tello.streamon()
#                 time.sleep(3)
#         if feed is None:
#             raise Exception('Could not open tello stream')


#         ### 
#         while True:
#             frame = feed.frame
#             dets, depth, yolobox = self.hailo.run(frame)

#             if _show(frame, dets, yolobox):
#                 break

#         if self.tello.stream_on:
#             self.tello.streamoff()



# def main():
#     # raise NotImplementedError('Usage: python main.py test [fps, show, yolo, file.mp4]')
#     tellomon = TelloMon()
    

# def hailo():
#     """
#     run hailo pipeline for testing
#     """
#     if '.mp4' in sys.argv[-1]:
#         cap = cv2.VideoCapture(sys.argv[-1])
#     else:
#         cap = cv2.VideoCapture(0)
#     hailo = HailoRun() 
#     hailo.load()
#     fps = []
    
#     print_interval = 1 # seconds
#     last_print = time.time()
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             # frame = cv2.resize(frame, (960, 720))
#             ct = time.time()
#             dets, depth, yolobox = hailo.run(frame)
#             dt = time.time() - ct
#             fps_cur = 1 / dt
#             fps.append(fps_cur)
#             # fps = fps[-30:]

#             if 'fps' in sys.argv[1:]:
#                 if last_print + print_interval < time.time():
#                     print(f'current fps: {fps_cur:.1f}, avg: {sum(fps) / len(fps):.1f}')
#                     last_print = time.time()

#             if _show(frame, dets, yolobox):
#                 break
            
#     except KeyboardInterrupt:
#         pass
#     finally:
#         cv2.destroyAllWindows()
#         print(f'Hailo pipeline average framerate: {sum(fps) / len(fps):.1f}')
   

# if __name__ == '__main__':
#     if 'test' in sys.argv[1:]:
#         hailo()
#         exit(0)

#     main()

    
