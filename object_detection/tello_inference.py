import cv2
import numpy as np
import os
import sys

# 현재 파일의 디렉토리를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 작업 디렉토리 변경
original_cwd = os.getcwd()
os.chdir(current_dir)

from reid_test import run, close
from object_detection_post_process import draw_detection

# 작업 디렉토리 복원
os.chdir(original_cwd)

class TelloInference:
    def __init__(self):
        """추론 엔진 초기화"""
        print("Initializing inference engine...")
        
        # object_detection 디렉토리 경로 저장
        self.inference_dir = os.path.dirname(os.path.abspath(__file__))
        self.original_cwd = os.getcwd()
        
        # 추론 시 사용할 작업 디렉토리로 일시적으로 변경
        os.chdir(self.inference_dir)
        
        print(f"Inference directory: {self.inference_dir}")
        print("Inference engine ready")
    
    def process_frame(self, frame):
        """
        프레임을 받아서 객체 감지 및 깊이 추정 수행
        
        Args:
            frame: RGB 이미지 (numpy array, shape: H x W x 3)
        
        Returns:
            detections: 감지된 객체 리스트
            depth_map: 깊이 맵 (numpy array)
        """
        try:
            # 작업 디렉토리를 inference_dir로 변경
            current_cwd = os.getcwd()
            os.chdir(self.inference_dir)
            
            # 프레임 크기
            fh, fw = frame.shape[:2]
            
            # reid_test의 run 함수로 추론 실행
            # run 함수는 (track_id, label, score, xmin, ymin, xmax, ymax) 형식 반환
            dets, dep = run(frame)
            
            # 작업 디렉토리 복원
            os.chdir(current_cwd)
            
            # 깊이 맵 크기 조정
            if dep is not None:
                dep = cv2.resize(dep, (fw, fh))
            
            # 감지 결과를 딕셔너리 형태로 변환
            detections = []
            for det in dets:
                tid, label, score, x1, y1, x2, y2 = det
                
                # Clamp bbox to frame boundaries
                x1 = max(0, min(int(x1), fw - 1))
                x2 = max(0, min(int(x2), fw - 1))
                y1 = max(0, min(int(y1), fh - 1))
                y2 = max(0, min(int(y2), fh - 1))
                
                # Ensure valid bbox (x2 > x1, y2 > y1)
                if x2 <= x1 or y2 <= y1:
                    continue
               
                detections.append({
                    'track_id': tid,
                    'class': label,
                    'confidence': float(score),
                    # Store in [x1, y1, x2, y2] format for frontend
                    'bbox': [x1, y1, x2, y2]
                })
            
            return detections, dep
            
        except Exception as e:
            print(f"Inference error: {e}")
            import traceback
            traceback.print_exc()
            # 에러 발생 시에도 작업 디렉토리 복원
            try:
                os.chdir(current_cwd)
            except:
                pass
            return [], None
    
    def draw_detections_on_frame(self, frame, detections, target_track_id=None):
        """
        프레임에 감지 결과 그리기
        
        Args:
            frame: RGB 이미지
            detections: 감지된 객체 리스트 (bbox in [x1, y1, x2, y2] format)
            target_track_id: 추적 중인 타겟의 track_id (빨간색으로 표시)
        
        Returns:
            annotated_frame: 감지 결과가 그려진 프레임
        """
        annotated_frame = frame.copy()
        h, w = annotated_frame.shape[:2]
        
        for det in detections:
            tid = det['track_id']
            label = det['class']
            score = det['confidence']
            x1, y1, x2, y2 = det['bbox']  # [x1, y1, x2, y2] format
            
            # 추적 중인 타겟이면 빨간색, 아니면 흰색
            is_target = (tid == target_track_id)
            color = (0, 0, 255) if is_target else (255, 255, 255)  # RGB
            
            # 라벨 수정 (추적 중이면 표시)
            if is_target:
                label_text = [f"🎯 {label}", f"ID {tid}"]
            else:
                label_text = [label, f"ID {tid}"]
            
            # draw_detection expects [ymin, xmin, ymax, xmax]
            draw_detection(
                annotated_frame,
                [y1, x1, y2, x2],
                label_text,
                score=score * 100.0,
                color=color,
                track=True
            )
            
            # 추적 중인 타겟이면 중심점도 그리기
            if is_target:
                # bbox를 프레임 범위 내로 클리핑
                x1_clipped = max(0, min(x1, w - 1))
                y1_clipped = max(0, min(y1, h - 1))
                x2_clipped = max(0, min(x2, w - 1))
                y2_clipped = max(0, min(y2, h - 1))
                
                # 유효한 bbox인지 확인
                if x2_clipped > x1_clipped and y2_clipped > y1_clipped:
                    # 클리핑된 bbox의 중심점 계산
                    center_x = int((x1_clipped + x2_clipped) / 2)
                    center_y = int((y1_clipped + y2_clipped) / 2)
                    
                    print(f"Target bbox: ({x1}, {y1}, {x2}, {y2}) -> center: ({center_x}, {center_y})")
                    
                    # 중심점이 프레임 내부에 있을 때만 그리기
                    if 0 <= center_x < w and 0 <= center_y < h:
                        cv2.circle(annotated_frame, (center_x, center_y), 10, (255, 0, 0), -1)
                        cv2.circle(annotated_frame, (center_x, center_y), 15, (255, 0, 0), 2)
        
        return annotated_frame
    
    def cleanup(self):
        """리소스 정리"""
        try:
            # 작업 디렉토리를 inference_dir로 변경
            os.chdir(self.inference_dir)
            close()
            # 원래 디렉토리로 복원
            os.chdir(self.original_cwd)
            print("Inference engine cleaned up")
        except Exception as e:
            print(f"Cleanup error: {e}")
            # 에러 발생 시에도 원래 디렉토리로 복원 시도
            try:
                os.chdir(self.original_cwd)
            except:
                pass