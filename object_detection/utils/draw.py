import cv2

def draw_track(frame, box, visible_id, color=(0,255,0)):  # 초록
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # visible_id가 None이면 "??" 로, 아니면 숫자 그대로
    if visible_id is None:
        id_str = "??"
    else:
        id_str = str(visible_id)

    cv2.putText(frame, f"ID:{id_str}", (x1, max(0, y1-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    
def draw_focus(frame, box, tid, color=(0,0,255)): #레드
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.putText(frame, f"ID:{tid}", (x1, max(0, y1-6)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)