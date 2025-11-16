
"""

TODO:: Tello ASCII art? xD 

"""

import sys
import time
import cv2
import numpy as np
from common.toolbox import load_json_file
from hailo import Hailo
from yolo_tools import draw_detection


class TelloMon():
    def __init__(self, config:dict):
        self.config = config
        


def main():
    raise NotImplementedError('Usage: python main.py test [fps, show, yolo, file.mp4]')
    config = load_json_file('config.json')
    tello = TelloMon(config)
    hailo = Hailo()
    
    pass

def hailo():
    """
    run hailo pipeline for testing
    """
    if '.mp4' in sys.argv[-1]:
        cap = cv2.VideoCapture(sys.argv[-1])
    else:
        cap = cv2.VideoCapture(0)
    hailo = Hailo() 
    fps = []
    
    print_interval = 1 # seconds
    last_print = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # frame = cv2.resize(frame, (960, 720))
            ct = time.time()
            dets, depth, yolobox = hailo.run(frame)
            dt = time.time() - ct
            fps_cur = 1 / dt
            fps.append(fps_cur)
            # fps = fps[-30:]

            if 'fps' in sys.argv[1:]:
                if last_print + print_interval < time.time():
                    print(f'current fps: {fps_cur:.1f}, avg: {sum(fps) / len(fps):.1f}')
                    last_print = time.time()

            if 'show' in sys.argv[1:]:
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
            
                if 'yolo' in sys.argv[1:]:
                    for box in yolobox:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 1)

                cv2.imshow('out', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print(f'Hailo pipeline average framerate: {sum(fps) / len(fps):.1f}')
   

if __name__ == '__main__':
    if 'test' in sys.argv[1:]:
        hailo()
        exit(0)

    main()

    
