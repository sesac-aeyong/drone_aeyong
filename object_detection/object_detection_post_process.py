import cv2
import numpy as np
from common.toolbox import id_to_color


def inference_result_handler(original_frame, infer_results, labels, config_data, tracker=None):
    """
    Processes inference results and draw detections (with optional tracking).

    Args:
        infer_results (list): Raw output from the model.
        original_frame (np.ndarray): Original image frame.
        labels (list): List of class labels.
        enable_tracking (bool): Whether tracking is enabled.
        tracker (BYTETracker, optional): ByteTrack tracker instance.

    Returns:
        np.ndarray: Frame with detections or tracks drawn.
    """
    detections = extract_detections(original_frame, infer_results, config_data)  #should return dict with boxes, classes, scores
    frame_with_detections = draw_detections(detections, original_frame, labels, tracker=tracker)
    return frame_with_detections


def draw_detection(image: np.ndarray, box: list, labels: list, score: float, color: tuple, track=False):
    """
    Draw box and label for one detection.

    Args:
        image (np.ndarray): Image to draw on.
        box (list): Bounding box coordinates in [ymin, xmin, ymax, xmax] format.
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
        pos = (xmin + 4, ymin + 40)
        cv2.putText(image, bottom_text, pos, font, 0.5, border_color, 2, cv2.LINE_AA)
        cv2.putText(image, bottom_text, pos, font, 0.5, text_color, 1, cv2.LINE_AA)


def denormalize_and_rm_pad(box: list, img_height=720, img_width=960) -> list:
    """
    Denormalize bounding box coordinates.

    Args:
        box (list): Normalized bounding box coordinates from YOLO model.
                    Could be [x1_norm, y1_norm, x2_norm, y2_norm] or [y1_norm, x1_norm, y2_norm, x2_norm].
        img_height (int): Height of the original image.
        img_width (int): Width of the original image.

    Returns:
        list: Denormalized bounding box coordinates in [ymin, xmin, ymax, xmax] format.
    """
    # YOLO outputs in [y1, x1, y2, x2] format (normalized)
    y1 = int(box[0] * img_height)
    x1 = int(box[1] * img_width)
    y2 = int(box[2] * img_height)
    x2 = int(box[3] * img_width)
    
    # draw_detection expects [ymin, xmin, ymax, xmax]
    return [y1, x1, y2, x2]


def extract_detections(image: np.ndarray, detections: list, config_data) -> dict:
    """
    Extract detections from the input data.

    Args:
        image (np.ndarray): Image to draw on.
        detections (list): Raw detections from the model.
        config_data (Dict): Loaded JSON config containing post-processing metadata.

    Returns:
        dict: Filtered detection results containing 'detection_boxes', 'detection_classes', 'detection_scores', and 'num_detections'.
    """

    visualization_params = config_data["visualization_params"]
    score_threshold = visualization_params.get("score_thres", 0.5)
    max_boxes = visualization_params.get("max_boxes_to_draw", 50)

    img_height, img_width = image.shape[:2]

    all_detections = []

    for class_id, detection in enumerate(detections):
        for det in detection:
            bbox, score = det[:4], det[4]
    
            if score >= score_threshold:
                denorm_bbox = denormalize_and_rm_pad(bbox, img_height=img_height, img_width=img_width)
                all_detections.append((score, class_id, denorm_bbox))

    #sort all detections by score descending
    all_detections.sort(reverse=True, key=lambda x: x[0])

    #take top max_boxes
    top_detections = all_detections[:max_boxes]

    scores, class_ids, boxes = zip(*top_detections) if top_detections else ([], [], [])

    return {
        'detection_boxes': list(boxes),
        'detection_classes': list(class_ids),
        'detection_scores': list(scores),
        'num_detections': len(top_detections)
    }


def draw_detections(detections: dict, img_out: np.ndarray, labels, tracker=None):
    """
    Draw detections or tracking results on the image.

    Args:
        detections (dict): Raw detection outputs.
        img_out (np.ndarray): Image to draw on.
        labels (list): List of class labels.
        enable_tracking (bool): Whether to use tracker output (ByteTrack).
        tracker (BYTETracker, optional): ByteTrack tracker instance.

    Returns:
        np.ndarray: Annotated image.
    """

    #extract detection data from the dictionary
    boxes = detections["detection_boxes"]  # List of [ymin, xmin, ymax, xmax] boxes
    scores = detections["detection_scores"]  # List of detection confidences
    num_detections = detections["num_detections"]  # Total number of valid detections
    classes = detections["detection_classes"]  # List of class indices per detection

    if tracker:
        dets_for_tracker = []

        # Convert detection format to [x1, y1, x2, y2, score] for tracker
        for idx in range(num_detections):
            box = boxes[idx]  # [ymin, xmin, ymax, xmax]
            ymin, xmin, ymax, xmax = box
            score = scores[idx]
            # Tracker expects [x1, y1, x2, y2, score]
            dets_for_tracker.append([xmin, ymin, xmax, ymax, score])

        #skip tracking if no detections passed
        if not dets_for_tracker:
            return img_out

        #run BYTETracker and get active tracks
        online_targets = tracker.update(np.array(dets_for_tracker))

        #draw tracked bounding boxes with ID labels
        for track in online_targets:
            track_id = track.track_id  #unique tracker ID
            x1, y1, x2, y2 = track.tlbr  #bounding box (top-left, bottom-right) in [x1, y1, x2, y2]
            xmin, ymin, xmax, ymax = map(int, [x1, y1, x2, y2])
            
            # Convert to [ymin, xmin, ymax, xmax] for find_best_matching_detection_index
            track_box_yx = [ymin, xmin, ymax, xmax]
            best_idx = find_best_matching_detection_index(track_box_yx, boxes)
            
            color = tuple(id_to_color(classes[best_idx]).tolist()) if best_idx is not None else (255, 255, 255)
            
            # draw_detection expects [ymin, xmin, ymax, xmax]
            if best_idx is None:
                draw_detection(img_out, [ymin, xmin, ymax, xmax], [f"ID {track_id}"],
                               track.score * 100.0, color, track=True)
            else:
                draw_detection(img_out, [ymin, xmin, ymax, xmax], [labels[classes[best_idx]], f"ID {track_id}"],
                               track.score * 100.0, color, track=True)

    else:
        #No tracking — draw raw model detections
        for idx in range(num_detections):
            if labels[classes[idx]] != 'person':
                continue
            color = tuple(id_to_color(classes[idx]).tolist())  #color based on class
            draw_detection(img_out, boxes[idx], [labels[classes[idx]]], scores[idx] * 100.0, color)

    return img_out


def find_best_matching_detection_index(track_box, detection_boxes):
    """
    Finds the index of the detection box with the highest IoU relative to the given tracking box.

    Args:
        track_box (list or tuple): The tracking box in [ymin, xmin, ymax, xmax] format.
        detection_boxes (list): List of detection boxes in [ymin, xmin, ymax, xmax] format.

    Returns:
        int or None: Index of the best matching detection, or None if no match is found.
    """
    best_iou = 0
    best_idx = -1

    for i, det_box in enumerate(detection_boxes):
        iou = compute_iou(track_box, det_box)
        if iou > best_iou:
            best_iou = iou
            best_idx = i

    return best_idx if best_idx != -1 else None


def compute_iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    IoU measures the overlap between two boxes:
        IoU = (area of intersection) / (area of union)
    Values range from 0 (no overlap) to 1 (perfect overlap).

    Args:
        boxA (list or tuple): [ymin, xmin, ymax, xmax]
        boxB (list or tuple): [ymin, xmin, ymax, xmax]

    Returns:
        float: IoU value between 0 and 1.
    """
    # boxA, boxB are in [ymin, xmin, ymax, xmax] format
    yminA, xminA, ymaxA, xmaxA = boxA
    yminB, xminB, ymaxB, xmaxB = boxB
    
    xA = max(xminA, xminB)
    yA = max(yminA, yminB)
    xB = min(xmaxA, xmaxB)
    yB = min(ymaxA, ymaxB)
    
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(1e-5, (xmaxA - xminA) * (ymaxA - yminA))
    areaB = max(1e-5, (xmaxB - xminB) * (ymaxB - yminB))
    return inter / (areaA + areaB - inter + 1e-5)