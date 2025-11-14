import cv2
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

original_cwd = os.getcwd()
os.chdir(current_dir)

from reid_test import run, close
from object_detection_post_process import draw_detection

os.chdir(original_cwd)


class TelloInference:
    def __init__(self):
        """추론 엔진 초기화"""
        print("Initializing inference engine...")

        self.inference_dir = os.path.dirname(os.path.abspath(__file__))
        self.original_cwd = os.getcwd()
        
        # 추론 시 사용할 작업 디렉토리로 일시적으로 변경
        os.chdir(self.inference_dir)
    
    def process_frame(self, frame):
        """
        프레임을 받아서 객체 감지 및 깊이 추정 수행
        
        Args:
            frame: RGB 이미지 (numpy array, shape: H x W x 3)
        
        Returns:
            detections: 감지된 객체 리스트 [(track_id, label, score, x1, y1, x2, y2), ...]
            depth_map: 깊이 맵 (numpy array)
        """
        try:
            # 프레임 크기 조정 및 변환
            fh, fw = frame.shape[:2]
            
            # reid_test의 run 함수로 추론 실행
            dets, dep = run(frame)
            print(dets)
            
            # 깊이 맵 크기 조정
            dep = cv2.resize(dep, (fw, fh))
            
            # 감지 결과를 딕셔너리 형태로 변환
            detections = []
            for det in dets:
                tid, label, score, x1, y1, x2, y2 = det
                detections.append({
                    'track_id': tid,
                    'class': label,
                    'confidence': float(score),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
            
            return detections, dep
            
        except Exception as e:
            print(f"Inference error: {e}")
            return [], None
    
    def draw_detections_on_frame(self, frame, detections):
        """
        프레임에 감지 결과 그리기
        
        Args:
            frame: RGB 이미지
            detections: 감지된 객체 리스트
        
        Returns:
            annotated_frame: 감지 결과가 그려진 프레임
        """
        annotated_frame = frame.copy()
        
        for det in detections:
            tid = det['track_id']
            label = det['class']
            score = det['confidence']
            x1, y1, x2, y2 = det['bbox']
            
            draw_detection(
                annotated_frame,
                [x1, y1, x2, y2],
                [label, f"ID {tid}"],
                score=score * 100.0,
                color=(255, 255, 255),
                track=True
            )
        
        return annotated_frame
    
    def cleanup(self):
        """리소스 정리"""
        try:
            close()
            print("Inference engine cleaned up")
        except Exception as e:
            print(f"Cleanup error: {e}")