import cv2

def draw_track(frame, box, tid, color=(0,255,0)):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    cv2.putText(frame, f"ID:{tid}", (x1, max(0,y1-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)