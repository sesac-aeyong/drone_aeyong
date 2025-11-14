#!/usr/bin/env python3
"""
BBox 순서 테스트 스크립트
reid_test.py의 반환값이 [xmin, ymin, xmax, ymax] 순서인지 확인
"""

import cv2
import numpy as np
import sys
import os

# object_detection 디렉토리 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'object_detection'))

from reid_test import run

# 테스트 이미지 생성 (640x480)
frame = np.zeros((480, 640, 3), dtype=np.uint8)

# 좌상단에 빨간 사각형 그리기 (위치가 명확하도록)
# x: 50-150, y: 50-150
cv2.rectangle(frame, (50, 50), (150, 150), (0, 0, 255), -1)  # 빨간색 채우기

# 우하단에 파란 사각형 그리기
# x: 490-590, y: 330-430
cv2.rectangle(frame, (490, 330), (590, 430), (255, 0, 0), -1)  # 파란색 채우기

# 텍스트 표시
cv2.putText(frame, "Top-Left Red (50,50)-(150,150)", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(frame, "Bottom-Right Blue (490,330)-(590,430)", (350, 460), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

print("="*60)
print("BBox Order Test")
print("="*60)
print(f"Frame size: {frame.shape} (H, W, C)")
print()
print("Expected detections:")
print("  Red box:  x=[50-150],  y=[50-150]")
print("  Blue box: x=[490-590], y=[330-430]")
print()

# BGR to RGB (reid_test expects RGB)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Detection 실행
try:
    dets, dep = run(frame_rgb)
    
    print(f"Got {len(dets)} detections")
    print()
    
    for i, det in enumerate(dets):
        tid, label, score, x1, y1, x2, y2 = det
        print(f"Detection #{i+1}:")
        print(f"  Track ID: {tid}")
        print(f"  Class: {label}")
        print(f"  Score: {score:.2f}")
        print(f"  BBox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        print(f"  Width: {x2-x1}, Height: {y2-y1}")
        
        # 범위 체크
        if x1 < 0 or y1 < 0 or x2 > 640 or y2 > 480:
            print(f"  ⚠️ WARNING: Out of bounds!")
        if x2 <= x1:
            print(f"  ⚠️ ERROR: x2 <= x1 (width is negative or zero)")
        if y2 <= y1:
            print(f"  ⚠️ ERROR: y2 <= y1 (height is negative or zero)")
            
        # 예상 위치와 비교
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        if 50 <= center_x <= 150 and 50 <= center_y <= 150:
            print(f"  ✅ Likely the RED box (top-left)")
        elif 490 <= center_x <= 590 and 330 <= center_y <= 430:
            print(f"  ✅ Likely the BLUE box (bottom-right)")
        else:
            print(f"  ❓ Unknown detection at ({center_x}, {center_y})")
        
        print()
    
    # 결과 시각화
    frame_result = frame.copy()
    for det in dets:
        tid, label, score, x1, y1, x2, y2 = det
        cv2.rectangle(frame_result, (int(x1), int(y1)), (int(x2), int(y2)), 
                     (0, 255, 0), 2)  # 초록색 테두리
        cv2.putText(frame_result, f"ID{tid}", (int(x1), int(y1)-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Test Frame', frame_result)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*60)
print("Test complete")
print("="*60)