import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import pandas as pd
from collections import deque
import time
import matplotlib.pyplot as plt
import os


def download_model():
    import urllib.request
    model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    model_path = "pose_landmarker_heavy.task"
    
    if not os.path.exists(model_path):
        print("Скачивание модели MediaPipe...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Модель скачана!")
    return model_path

# Создание детектора позы
def create_pose_detector():
    model_path = download_model()
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,  # Для изображений используем IMAGE
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False
    )
    return vision.PoseLandmarker.create_from_options(options)

LANDMARK_NAMES = {
    0: 'нос', 1: 'глаз(лев)', 2: 'глаз(прав)', 3: 'ухо(лев)', 4: 'ухо(прав)',
    5: 'плечо(лев)', 6: 'плечо(прав)', 7: 'локоть(лев)', 8: 'локоть(прав)',
    9: 'запястье(лев)', 10: 'запястье(прав)', 11: 'кисть(лев)', 12: 'кисть(прав)',
    13: 'бедро(лев)', 14: 'бедро(прав)', 15: 'колено(лев)', 16: 'колено(прав)',
    17: 'голень(лев)', 18: 'голень(прав)', 19: 'ступня(лев)', 20: 'ступня(прав)',
    21: 'пятка(лев)', 22: 'пятка(прав)', 23: 'носок(лев)', 24: 'носок(прав)',
    25: 'бедро(лев внутр)', 26: 'бедро(прав внутр)', 27: 'бедро(лев верх)', 
    28: 'бедро(прав верх)', 29: 'плечо(лев верх)', 30: 'плечо(прав верх)', 
    31: 'ухо(лев верх)', 32: 'ухо(прав верх)'
}

KEY_POINTS = {
    'левое_плечо': 11, 'правое_плечо': 12,
    'левое_колено': 25, 'правое_колено': 26,
    'левая_ступня': 27, 'правая_ступня': 28,
    'левый_локоть': 13, 'правый_локоть': 14,
    'левый_таз': 23, 'правый_таз': 24,
}

def convert_landmarks_to_list(pose_landmarks):
    """Конвертирует объект NormalizedLandmark в список словарей"""
    landmarks = []
    for landmark in pose_landmarks:
        landmarks.append({
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z,
            'visibility': landmark.visibility if hasattr(landmark, 'visibility') else 1.0
        })
    return landmarks

def calc_angle(A, B, C):
    
    A = np.array([A[0], A[1]])
    B = np.array([B[0], B[1]])
    C = np.array([C[0], C[1]])
    
    BA = A - B
    BC = C - B
    
    dot_product = np.dot(BA, BC)
    magnitude_BA = np.linalg.norm(BA)
    magnitude_BC = np.linalg.norm(BC)
    
    if magnitude_BA == 0 or magnitude_BC == 0:
        return 0
    
    cos_angle = dot_product / (magnitude_BA * magnitude_BC)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

class ExerciseCounter:
    def __init__(self, down_threshold=90, up_threshold=160):
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.phase = 'up'
        self.count = 0
        
    def update(self, angle):
        if self.phase == 'up':
            if angle < self.down_threshold:
                self.phase = 'down'
        elif self.phase == 'down':
            if angle > self.up_threshold:
                self.phase = 'up'
                self.count += 1
        return self.count, self.phase

class PostureClassifier:
    @staticmethod
    def classify(shoulder_y, hip_y):
        distance = abs(hip_y - shoulder_y)
        if distance > 0.35:
            return "STANDING"
        else:
            return "SITTING"

def test_on_single_image(image_path, detector):
    """Этап 1: Детектирование на одном фото"""
    print("\n" + "="*60)
    print("ЭТАП 1: Тест на одном фото")
    print("="*60)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    

    detection_result = detector.detect(mp_image)
    
    if detection_result.pose_landmarks:

        print("\nКоординаты всех 33 точек (x, y, z, visibility):")
        print("-" * 80)
        print(f"{'№':<4} {'Точка':<15} {'x':<10} {'y':<10} {'z':<10} {'vis':<8}")
        print("-" * 80)
        
        landmarks = detection_result.pose_landmarks[0]
        for idx in range(33):
            lm = landmarks[idx]
            name = LANDMARK_NAMES.get(idx, f"точка_{idx}")
            vis = lm.visibility if hasattr(lm, 'visibility') else 1.0
            print(f"{idx:<4} {name:<15} {lm.x:<10.4f} {lm.y:<10.4f} "
                  f"{lm.z:<10.4f} {vis:<8.3f}")
        
        annotated_image = image.copy()
        height, width, _ = image.shape
        
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # руки
            (11, 23), (12, 24), (23, 24),  # туловище
            (23, 25), (25, 27), (24, 26), (26, 28)  # ноги
        ]
        
        for idx, lm in enumerate(landmarks):
            x = int(lm.x * width)
            y = int(lm.y * height)
            cv2.circle(annotated_image, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(annotated_image, str(idx), (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        for connection in connections:
            if connection[0] < len(landmarks) and connection[1] < len(landmarks):
                p1 = landmarks[connection[0]]
                p2 = landmarks[connection[1]]
                x1, y1 = int(p1.x * width), int(p1.y * height)
                x2, y2 = int(p2.x * width), int(p2.y * height)
                cv2.line(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imwrite('test_output.jpg', annotated_image)
        print(f"\nРезультат сохранён в 'test_output.jpg'")
        cv2.imshow('Test Image', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Точки не обнаружены")

def main_webcam():
    """Основная функция для работы с веб-камерой"""
    print("\n" + "="*60)
    print("Работа с веб-камерой")
    print("="*60)
    print("Нажмите 'q' для выхода")
    print("Нажмите 'r' для сброса счётчика")
    
    detector = create_pose_detector()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть веб-камеру")
        return
    
    prev_time = time.time()
    
    squat_counter = ExerciseCounter()
    posture_classifier = PostureClassifier()
    coordinates_history = {name: [] for name in KEY_POINTS.keys()}
    frame_counter = 0
    
    print("Запуск обработки...")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        detection_result = detector.detect(mp_image)
        height, width, _ = frame.shape
        
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
            
            # Запись координат
            if frame_counter < 200:
                for name, idx in KEY_POINTS.items():
                    lm = landmarks[idx]
                    coordinates_history[name].append({
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z,
                        'visibility': lm.visibility if hasattr(lm, 'visibility') else 1.0,
                        'frame': frame_counter
                    })
                frame_counter += 1
                
                if frame_counter == 200:
                    print("\nЗаписано 200 кадров!")
                    save_coordinates(coordinates_history)
            
            # Рисуем скелет
            connections = [
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)
            ]
            
            for idx, lm in enumerate(landmarks):
                if idx < 33:
                    x = int(lm.x * width)
                    y = int(lm.y * height)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            for conn in connections:
                if conn[0] < 33 and conn[1] < 33:
                    p1 = landmarks[conn[0]]
                    p2 = landmarks[conn[1]]
                    x1, y1 = int(p1.x * width), int(p1.y * height)
                    x2, y2 = int(p2.x * width), int(p2.y * height)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            left_shoulder = (landmarks[11].x * width, landmarks[11].y * height)
            left_elbow = (landmarks[13].x * width, landmarks[13].y * height)
            left_wrist = (landmarks[15].x * width, landmarks[15].y * height)
            right_shoulder = (landmarks[12].x * width, landmarks[12].y * height)
            right_elbow = (landmarks[14].x * width, landmarks[14].y * height)
            right_wrist = (landmarks[16].x * width, landmarks[16].y * height)
            left_hip = (landmarks[23].x * width, landmarks[23].y * height)
            left_knee = (landmarks[25].x * width, landmarks[25].y * height)
            left_ankle = (landmarks[27].x * width, landmarks[27].y * height)
            right_hip = (landmarks[24].x * width, landmarks[24].y * height)
            right_knee = (landmarks[26].x * width, landmarks[26].y * height)
            right_ankle = (landmarks[28].x * width, landmarks[28].y * height)
            
            left_elbow_angle = calc_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calc_angle(right_shoulder, right_elbow, right_wrist)
            left_knee_angle = calc_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calc_angle(right_hip, right_knee, right_ankle)
            
            avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
            count, phase = squat_counter.update(avg_knee_angle)
            
            avg_shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
            avg_hip_y = (landmarks[23].y + landmarks[24].y) / 2
            posture = posture_classifier.classify(avg_shoulder_y, avg_hip_y)
            
            symmetry_warning = ""
            if abs(left_knee_angle - right_knee_angle) > 15:
                symmetry_warning = "⚠️ НЕСИММЕТРИЧНО!"
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"L-Elbow: {left_elbow_angle:.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"R-Elbow: {right_elbow_angle:.0f}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"L-Knee: {left_knee_angle:.0f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"R-Knee: {right_knee_angle:.0f}", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"SQUATS: {count}", (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Phase: {phase.upper()}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(frame, f"POSTURE: {posture}", (10, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if symmetry_warning:
                cv2.putText(frame, symmetry_warning, (10, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow('Pose Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            squat_counter = ExerciseCounter()
            print("Счётчик сброшен")
    
    cap.release()
    cv2.destroyAllWindows()

def save_coordinates(history):
    """Сохранение координат в CSV"""
    data = []
    for name, points in history.items():
        for p in points:
            data.append({
                'point': name,
                'frame': p['frame'],
                'x': p['x'],
                'y': p['y'],
                'z': p['z'],
                'visibility': p['visibility']
            })
    
    df = pd.DataFrame(data)
    df.to_csv('pose_coordinates.csv', index=False)
    print("Координаты сохранены в 'pose_coordinates.csv'")

if __name__ == "__main__":
    print("Выберите режим работы:")
    print("1 - Обработка фото")
    print("2 - Веб-камера")
    
    choice = input("Ваш выбор (1 или 2): ")
    
    if choice == '1':
        photo_path = input("Введите полный путь к фото: ")
        photo_path = photo_path.strip().strip('"').strip("'")
        
        if os.path.exists(photo_path):
            detector = create_pose_detector()
            test_on_single_image(photo_path, detector)
        else:
            print(f"Файл не найден: {photo_path}")
    else:
        main_webcam()