

import cv2
import numpy as np
from settings import settings as S

def _denormalize_and_rm_pad(box: list, size: int, padding_length: int, input_height: int, input_width: int) -> list:
    """
    Denormalize bounding box coordinates and remove padding.

    Args:
        box (list): Normalized bounding box coordinates.
        size (int): Size to scale the coordinates.
        padding_length (int): Length of padding to remove.
        input_height (int): Height of the input image.
        input_width (int): Width of the input image.

    Returns:
        list: Denormalized bounding box coordinates with padding removed.
    """
    for i, x in enumerate(box):
        box[i] = int(x * size)
        if (input_width != size) and (i % 2 != 0):
            box[i] -= padding_length
        if (input_height != size) and (i % 2 == 0):
            box[i] -= padding_length
    y1, x1, y2, x2 = box
    return x1, y1, x2, y2
    # return y1, x1, y2, x2


def extract_detections(image: np.ndarray, detections: list) -> dict:
    """
    Extract detections from the input data.

    Args:
        image (np.ndarray): Image to draw on.
        detections (list): Raw detections from the model.

    Returns:
        dict: Filtered detection results containing 'detection_boxes', 'detection_classes', 'detection_scores', and 'num_detections'.
    """

    score_threshold = S.min_vis_score_threshold
    max_detections = S.max_vis_detections

    #values used for scaling coords and removing padding
    img_height, img_width = image.shape[:2]
    size = max(img_height, img_width)
    padding_length = int(abs(img_height - img_width) / 2)

    all_detections = []

    for class_id, detection in enumerate(detections):
        for det in detection:
            bbox, score = det[:4], det[4]
            if score >= score_threshold:
                denorm_bbox = _denormalize_and_rm_pad(bbox, size, padding_length, img_height, img_width)
                all_detections.append((score, class_id, denorm_bbox))

    #sort all detections by score descending
    all_detections.sort(reverse=True, key=lambda x: x[0])

    #take top max_boxes
    top_detections = all_detections[:max_detections]

    scores, class_ids, boxes = zip(*top_detections) if top_detections else ([], [], [])

    return boxes, scores, class_ids, len(top_detections)



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
    # ymin, xmin, ymax, xmax = map(int, box)
    xmin, ymin, xmax, ymax = map(int, box)
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
        pos = (xmin + 4, ymin + 40)
        cv2.putText(image, bottom_text, pos, font, 0.5, border_color, 2, cv2.LINE_AA)
        cv2.putText(image, bottom_text, pos, font, 0.5, text_color, 1, cv2.LINE_AA)


def draw_detections_on_frame(frame: np.ndarray, detections: list, target_track_id=None) -> np.ndarray:
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
        score = float(det['confidence'])
        x1, y1, x2, y2 = det['bbox']  # [x1, y1, x2, y2] format
        
        # 추적 중인 타겟이면 빨간색, 아니면 흰색
        is_target = (tid == target_track_id)
        color = (0, 0, 255) if is_target else (255, 255, 255)  # RGB
        
        # 라벨 수정 (추적 중이면 표시)
        if is_target:
            label_text = [f"{label}", f"ID {tid}"]
        else:
            label_text = [label, f"ID {tid}"]

        draw_detection(
            annotated_frame,
            [x1, y1, x2, y2],
            label_text,
            score * 100.0,
            color,
            True
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
                
                # 중심점이 프레임 내부에 있을 때만 그리기
                if 0 <= center_x < w and 0 <= center_y < h:
                    cv2.circle(annotated_frame, (center_x, center_y), 10, (255, 0, 0), -1)
                    cv2.circle(annotated_frame, (center_x, center_y), 15, (255, 0, 0), 2)
    
    return annotated_frame