
"""

TODO:: Tello ASCII art? xD 

"""

import sys
import time
import cv2
import numpy as np
from common.toolbox import load_json_file
from hailo import Hailo


class TelloMon():
    def __init__(self, config:dict):
        self.config = config
        


def main():
    raise NotImplementedError('Usage: python main.py test [fps, show]')
    config = load_json_file('config.json')
    tello = TelloMon(config)
    hailo = Hailo()
    
    pass

def hailo():
    """
    run hailo pipeline for testing
    """
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('test.mp4')
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
            dets, depth = hailo.run(frame)
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
                cv2.imshow('out', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
    except KeyboardInterrupt:
        pass
    finally:
        print(f'Hailo pipeline average framerate: {sum(fps) / len(fps):.1f}')
   

if __name__ == '__main__':
    if 'test' in sys.argv[1:]:
        hailo()
        exit(0)

    main()

    

def draw_detection(image: np.ndarray, box: list, labels: list, score: float, color: tuple, track=False):
    """
    Draw box and label for one detection.

    Args:
        image (np.ndarray): Image to draw on.
        box (list): Bounding box coordinates.
        labels (list): List of labels (1 or 2 elements).
        score (float): Detection score.
        color (tuple): Color for the bounding box.
        track (bool): Whether to include tracking info.
    """
    ymin, xmin, ymax, xmax = map(int, box)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Compose texts
    top_text = f"{labels[0]}: {score:.1f}%" if not track or len(labels) == 2 else f"{score:.1f}%"
    bottom_text = None

    if track:
        if len(labels) == 2:
            bottom_text = labels[1]
        else:
            bottom_text = labels[0]


    # Set colors
    text_color = (255, 255, 255)  # white
    border_color = (0, 0, 0)      # black

    # Draw top text with black border first
    cv2.putText(image, top_text, (xmin + 4, ymin + 20), font, 0.5, border_color, 2, cv2.LINE_AA)
    cv2.putText(image, top_text, (xmin + 4, ymin + 20), font, 0.5, text_color, 1, cv2.LINE_AA)

    # Draw bottom text if exists
    if bottom_text:
        # pos = (xmax - 50, ymax - 6)
        pos = (xmin + 4, ymin + 40)
        cv2.putText(image, bottom_text, pos, font, 0.5, border_color, 2, cv2.LINE_AA)
        cv2.putText(image, bottom_text, pos, font, 0.5, text_color, 1, cv2.LINE_AA)

