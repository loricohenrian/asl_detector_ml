import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mediapipe as mp
import os
import json
from datetime import datetime
from collections import deque
import argparse
import pickle
import time
import scipy.ndimage as ndimage
from scipy.interpolate import interp1d

class ASLNet(nn.Module):
    """Neural network for ASL letter classification (static letters)"""
    def __init__(self, input_size=126, hidden_size=256, num_classes=26):
        super(ASLNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class FastDynamicLetterNet(nn.Module):
    """Optimized LSTM network for fast dynamic ASL letter classification (J, Z)"""
    def __init__(self, input_size=126, hidden_size=128, num_layers=2, num_classes=2):
        super(FastDynamicLetterNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2, bidirectional=False)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class PhraseNet(nn.Module):
    """Enhanced LSTM-based network for ASL phrase classification with body and head movement"""
    def __init__(self, input_size=159, hidden_size=512, num_layers=4, num_classes=100):
        super(PhraseNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.4, bidirectional=True)
        self.bn1 = nn.BatchNorm1d(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.bn1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.bn2(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.bn3(x)
        x = self.fc3(x)
        return x

class ASLDetector:
    def __init__(self, headless=False):
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Updated input size: 2 hands * 21 landmarks * 3 coordinates + 11 pose landmarks * 3 coordinates
        self.input_size = 126 + 33
        self.labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.dynamic_labels = ['J', 'Z']
        self.dynamic_label_to_idx = {label: idx for idx, label in enumerate(self.dynamic_labels)}
        
        self.phrase_labels = [
            "Deaf", "Hard of hearing", "Hearing", "Bless", "Come here", "Ewan", "Excuse me",
            "Gestures", "Menu/Bill", "Ok", "Sana", "Uy", "Kamusta ka", "Sorry", "Stop",
            "Understand", "Wait", "Wrong", "Good afternoon", "Good evening", "Good morning",
            "Good night", "Good noon", "Hello", "Thank you", "Not much, you?", "Address",
            "Age", "Birthday", "Happy", "Sad", "Family", "You're welcome", "Nation", "Work",
            "What", "Bye", "Beautiful", "See you tomorrow", "Nice to meet you", "See you later",
            "I love you", "Not fine", "Next time", "Take care", "January", "February", "March",
            "April", "May", "June", "July", "August", "September", "October", "November",
            "December", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
            "Sunday", "Calendar", "Day", "Hour", "Later", "Minute", "Month", "Second", "Today",
            "Tomorrow", "Week", "Year", "Yesterday", "Absent", "Always", "Last", "Late", "Never",
            "Next", "Once", "Recent", "Sometimes", "Soon", "Twice", "Brown/Dark", "Fat", "Short",
            "Slim", "Tall", "I'm fine", "Again", "Don't know"
        ]
        self.phrase_to_idx = {phrase: idx for idx, phrase in enumerate(self.phrase_labels)}
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.letter_model = ASLNet(input_size=126, num_classes=len(self.labels))
        self.letter_model.to(self.device)
        
        self.dynamic_letter_model = FastDynamicLetterNet(input_size=126, num_classes=len(self.dynamic_labels))
        self.dynamic_letter_model.to(self.device)
        
        self.phrase_model = PhraseNet(input_size=self.input_size, num_classes=len(self.phrase_labels))
        self.phrase_model.to(self.device)
        
        self.training_data = []
        self.phrase_training_data = []
        self.current_label = None
        
        self.prediction_history = deque(maxlen=5)
        self.dynamic_prediction_history = deque(maxlen=3)
        self.phrase_prediction_history = deque(maxlen=10)
        self.sequence_buffer = deque(maxlen=90)
        self.fast_sequence_buffer = deque(maxlen=15)
        
        self.detection_mode = "letter"
        
        self.motion_threshold = 0.03
        self.last_features = None
        self.motion_frames = 0
        self.static_frames = 0
        
        self.headless = headless
        
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        if headless:
            os.makedirs('output_images', exist_ok=True)
    
    def calc_bounding_rect(self, image, landmarks):
        """Calculate bounding rectangle around landmarks"""
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_array = np.empty((0, 2), int)
        
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_array = np.append(landmark_array, [[landmark_x, landmark_y]], axis=0)
        
        x, y, w, h = cv2.boundingRect(landmark_array)
        return [x, y, x + w, y + h]
    
    def draw_bounding_rect(self, image, brect):
        """Draw bounding rectangle on image"""
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)
        return image
    
    def extract_single_hand(self, hand_landmarks):
        """Extract raw landmarks from a single hand"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks, dtype=np.float32)
    
    def extract_pose_landmarks(self, pose_landmarks):
        """Extract selected pose landmarks (upper body and head)"""
        landmarks = []
        # Select key pose landmarks: shoulders, elbows, head (nose, eyes, ears)
        key_indices = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        for idx in key_indices:
            landmark = pose_landmarks.landmark[idx]
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks, dtype=np.float32)
    
    def normalize_single_hand(self, landmarks):
        """Normalize single hand landmarks"""
        if len(landmarks) != 63:
            return np.zeros(63, dtype=np.float32)
        
        landmarks = landmarks.reshape(21, 3)
        wrist = landmarks[0]
        normalized = landmarks - wrist
        
        xs = normalized[:, 0]
        ys = normalized[:, 1]
        scale = max(np.max(np.abs(xs)), np.max(np.abs(ys))) if max(np.max(np.abs(xs)), np.max(np.abs(ys))) > 0 else 1.0
        
        normalized[:, 0] /= scale
        normalized[:, 1] /= scale
        normalized[:, 2] /= scale
        
        smoothed = ndimage.gaussian_filter1d(normalized.flatten(), sigma=0.5)
        return smoothed
    
    def normalize_pose_landmarks(self, landmarks):
        """Normalize pose landmarks"""
        if len(landmarks) != 33:
            return np.zeros(33, dtype=np.float32)
        
        landmarks = landmarks.reshape(11, 3)
        torso_center = landmarks[1:3].mean(axis=0)  # Average of shoulders
        normalized = landmarks - torso_center
        
        xs = normalized[:, 0]
        ys = normalized[:, 1]
        scale = max(np.max(np.abs(xs)), np.max(np.abs(ys))) if max(np.max(np.abs(xs)), np.max(np.abs(ys))) > 0 else 1.0
        
        normalized[:, 0] /= scale
        normalized[:, 1] /= scale
        normalized[:, 2] /= scale
        
        smoothed = ndimage.gaussian_filter1d(normalized.flatten(), sigma=0.5)
        return smoothed
    
    def extract_features(self, results):
        """Extract normalized features from hand and pose landmarks"""
        features = np.zeros(self.input_size, dtype=np.float32)
        
        # Extract hand features
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                raw_lms = self.extract_single_hand(hand_landmarks)
                features[idx*63:(idx+1)*63] = self.normalize_single_hand(raw_lms)
        
        # Extract pose features
        if results.pose_landmarks:
            raw_pose = self.extract_pose_landmarks(results.pose_landmarks)
            features[126:159] = self.normalize_pose_landmarks(raw_pose)
        
        return features
    
    def fast_resample_sequence(self, seq, target_len=12):
        """Fast resampling for dynamic letters"""
        if len(seq) == 0:
            return np.zeros((target_len, 126))
        
        seq = np.array(seq)
        
        if len(seq) < 2:
            return np.tile(seq[0], (target_len, 1)) if len(seq) > 0 else np.zeros((target_len, 126))
        
        t_old = np.linspace(0, 1, len(seq))
        t_new = np.linspace(0, 1, target_len)
        resampled = np.zeros((target_len, 126))
        
        for i in range(126):
            f = interp1d(t_old, seq[:, i], kind='linear')
            resampled[:, i] = f(t_new)
        return resampled
    
    def resample_sequence(self, seq, target_len=60):
        """Resample sequence to fixed length with temporal smoothing"""
        if len(seq) == 0:
            return np.zeros((target_len, self.input_size))
        
        seq = np.array(seq)
        
        for i in range(self.input_size):
            seq[:, i] = ndimage.gaussian_filter1d(seq[:, i], sigma=1.0)
        
        if len(seq) < 2:
            return np.tile(seq[0], (target_len, 1)) if len(seq) > 0 else np.zeros((target_len, self.input_size))
        
        t_old = np.linspace(0, 1, len(seq))
        t_new = np.linspace(0, 1, target_len)
        resampled = np.zeros((target_len, self.input_size))
        for i in range(self.input_size):
            f = interp1d(t_old, seq[:, i], kind='linear')
            resampled[:, i] = f(t_new)
        return resampled
    
    def detect_motion(self, features):
        """Enhanced motion detection for dynamic letters"""
        if self.last_features is None:
            self.last_features = features.copy()
            return False
        
        motion = np.mean(np.abs(features - self.last_features))
        
        if motion > self.motion_threshold:
            self.motion_frames += 1
            self.static_frames = 0
        else:
            self.static_frames += 1
            if self.static_frames > 5:
                self.motion_frames = 0
        
        self.last_features = features.copy()
        return self.motion_frames >= 2
    
    def collect_training_data(self):
        """Collect training data for ASL letters (static and dynamic)"""
        if self.headless:
            print("Error: Data collection not available in headless mode")
            return

        cap = cv2.VideoCapture(0)
        current_letter_idx = 0
        samples_per_letter = 200
        samples_collected = 0
        recording = False
        current_sequence = []
        dynamic_letters = ['J', 'Z']

        print("ASL Letter Training Data Collection")
        print("Instructions:")
        print("- For static letters (A–I, K–Y): Press SPACE to collect a single frame")
        print("- For dynamic letters (J, Z): Press 's' to start recording, 'e' to stop")
        print("- Press 'n' for next letter")
        print("- Press 'q' to quit")
        print(f"\nCurrent letter: {self.labels[current_letter_idx]}")
        if self.labels[current_letter_idx] in dynamic_letters:
            print("This is a dynamic letter. Use 's' to start, 'e' to stop recording.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = self.hands.process(rgb_frame)
            pose_results = self.pose.process(rgb_frame)
            results = self.HandPoseResults(hand_results, pose_results)

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks[:2]:
                    brect = self.calc_bounding_rect(frame, hand_landmarks)
                    frame = self.draw_bounding_rect(frame, brect)
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                features = self.extract_features(results)

                if recording and self.labels[current_letter_idx] in dynamic_letters:
                    current_sequence.append(features[:126])  # Only hand landmarks for dynamic letters
                    cv2.putText(frame, "RECORDING", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, f"Frames: {len(current_sequence)}", (10, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, f"Letter: {self.labels[current_letter_idx]}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {samples_collected}/{samples_per_letter}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if self.labels[current_letter_idx] in dynamic_letters:
                cv2.putText(frame, "Press 's'/'e' for dynamic", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Press SPACE for static", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            key = cv2.waitKey(1) & 0xFF
            if self.labels[current_letter_idx] in dynamic_letters:
                if key == ord('s') and not recording:
                    recording = True
                    current_sequence = []
                    print(f"Started recording for {self.labels[current_letter_idx]}")
                elif key == ord('e') and recording:
                    recording = False
                    if len(current_sequence) >= 10:
                        self.training_data.append({
                            'sequence': np.array(current_sequence),
                            'label': self.labels[current_letter_idx],
                            'is_dynamic': True
                        })
                        samples_collected += 1
                        print(f"Collected sample {samples_collected} for {self.labels[current_letter_idx]} with {len(current_sequence)} frames")
                        if samples_collected >= samples_per_letter:
                            current_letter_idx += 1
                            samples_collected = 0
                            recording = False
                            current_sequence = []
                            if current_letter_idx >= len(self.labels):
                                print("Data collection completed!")
                                break
                            print(f"\nNext letter: {self.labels[current_letter_idx]}")
                            if self.labels[current_letter_idx] in dynamic_letters:
                                print("This is a dynamic letter. Use 's' to start, 'e' to stop recording.")
                    else:
                        print("Sequence too short (minimum 10 frames)")
                    current_sequence = []
            else:
                if key == ord(' '):
                    self.training_data.append({
                        'landmarks': features[:126],  # Only hand landmarks for static letters
                        'label': self.labels[current_letter_idx],
                        'is_dynamic': False
                    })
                    samples_collected += 1
                    print(f"Collected sample {samples_collected} for letter {self.labels[current_letter_idx]}")
                    if samples_collected >= samples_per_letter:
                        current_letter_idx += 1
                        samples_collected = 0
                        if current_letter_idx >= len(self.labels):
                            print("Data collection completed!")
                            break
                        print(f"\nNext letter: {self.labels[current_letter_idx]}")
                        if self.labels[current_letter_idx] in dynamic_letters:
                            print("This is a dynamic letter. Use 's' to start, 'e' to stop recording.")

            if key == ord('n'):
                recording = False
                current_sequence = []
                current_letter_idx += 1
                samples_collected = 0
                if current_letter_idx >= len(self.labels):
                    print("Data collection completed!")
                    break
                print(f"\nNext letter: {self.labels[current_letter_idx]}")
                if self.labels[current_letter_idx] in dynamic_letters:
                    print("This is a dynamic letter. Use 's' to start, 'e' to stop recording.")

            elif key == ord('q'):
                break

            if not hand_results.multi_hand_landmarks:
                cv2.putText(frame, "No hand detected", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('ASL Data Collection', frame)

        cap.release()
        cv2.destroyAllWindows()

        if self.training_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f'data/asl_letters_{timestamp}.pkl', 'wb') as f:
                pickle.dump(self.training_data, f)
            print(f"Saved {len(self.training_data)} letter samples to data/asl_letters_{timestamp}.pkl")

    def collect_phrase_data(self):
        """Collect training data for ASL phrases with body and head movement"""
        if self.headless:
            print("Error: Data collection not available in headless mode")
            return
            
        cap = cv2.VideoCapture(0)
        current_phrase_idx = 0
        samples_per_phrase = 20
        samples_collected = 0
        recording = False
        current_sequence = []
        
        print("ASL Phrase Training Data Collection")
        print("Instructions:")
        print("- Press 's' to start recording a phrase gesture sequence")
        print("- Press 'e' to end recording")
        print("- Press 'n' for next phrase")
        print("- Press 'q' to quit")
        print(f"\nCurrent phrase: {self.phrase_labels[current_phrase_idx]}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = self.hands.process(rgb_frame)
            pose_results = self.pose.process(rgb_frame)
            
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks[:2]:
                    brect = self.calc_bounding_rect(frame, hand_landmarks)
                    frame = self.draw_bounding_rect(frame, brect)
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            if pose_results.pose_landmarks:
                brect = self.calc_bounding_rect(frame, pose_results.pose_landmarks)
                frame = self.draw_bounding_rect(frame, brect)
                self.mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
            if recording and (hand_results.multi_hand_landmarks or pose_results.pose_landmarks):
                features = self.extract_features(self.HandPoseResults(hand_results, pose_results))
                current_sequence.append(features)
                cv2.putText(frame, "RECORDING", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"Frames: {len(current_sequence)}", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No hand or pose detected", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(frame, f"Phrase: {self.phrase_labels[current_phrase_idx]}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {samples_collected}/{samples_per_phrase}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and not recording:
                recording = True
                current_sequence = []
                print(f"Started recording phrase: {self.phrase_labels[current_phrase_idx]}")
            
            elif key == ord('e') and recording:
                recording = False
                if len(current_sequence) > 20:
                    self.phrase_training_data.append({
                        'sequence': np.array(current_sequence),
                        'label': self.phrase_labels[current_phrase_idx]
                    })
                    samples_collected += 1
                    print(f"Collected phrase sample {samples_collected} with {len(current_sequence)} frames")
                    
                    if samples_collected >= samples_per_phrase:
                        current_phrase_idx += 1
                        samples_collected = 0
                        if current_phrase_idx >= len(self.phrase_labels):
                            print("Phrase data collection completed!")
                            break
                        print(f"\nNext phrase: {self.phrase_labels[current_phrase_idx]}")
                else:
                    print("Sequence too short, please try again (minimum 20 frames)")
                current_sequence = []
            
            elif key == ord('n'):
                recording = False
                current_sequence = []
                current_phrase_idx += 1
                samples_collected = 0
                if current_phrase_idx >= len(self.phrase_labels):
                    print("Phrase data collection completed!")
                    break
                print(f"\nNext phrase: {self.phrase_labels[current_phrase_idx]}")
            
            elif key == ord('q'):
                break
            
            cv2.imshow('ASL Phrase Collection', frame)
        
        cap.release()
        cv2.destroyAllWindows()
        
        if self.phrase_training_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f'data/asl_phrases_{timestamp}.pkl', 'wb') as f:
                pickle.dump(self.phrase_training_data, f)
            print(f"Saved {len(self.phrase_training_data)} phrase samples to data/asl_phrases_{timestamp}.pkl")
    
    class HandPoseResults:
        """Helper class to combine hand and pose results"""
        def __init__(self, hand_results, pose_results):
            self.multi_hand_landmarks = hand_results.multi_hand_landmarks
            self.pose_landmarks = pose_results.pose_landmarks
    
    def load_training_data(self, data_path):
        """Load training data from file"""
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data[0], dict) and 'sequence' in data[0] and 'is_dynamic' not in data[0]:
                self.phrase_training_data = data
                print(f"Loaded {len(self.phrase_training_data)} phrase training samples")
            else:
                self.training_data = data
                print(f"Loaded {len(self.training_data)} letter training samples")
    
    def train_letter_model(self, epochs=100, batch_size=32):
        """Train the ASL letter classification model for static letters"""
        if not self.training_data:
            print("No letter training data available. Please collect data first.")
            return

        static_data = [sample for sample in self.training_data if not sample.get('is_dynamic', False)]
        if not static_data:
            print("No static letter data available.")
            return

        X = []
        y = []
        for sample in static_data:
            lm = sample['landmarks']
            label = self.label_to_idx[sample['label']]
            X.append(lm)
            y.append(label)

            for _ in range(3):
                aug_lm = lm + np.random.normal(0, 0.01, lm.shape)
                X.append(aug_lm)
                y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.letter_model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        self.letter_model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.letter_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                accuracy = 100 * correct / total
                print(f'Letter Model - Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'models/asl_letters_{timestamp}.pth'
        torch.save(self.letter_model.state_dict(), model_path)
        print(f"Letter model saved to {model_path}")

    def train_dynamic_letter_model(self, epochs=50, batch_size=16):
        """Train the fast dynamic ASL letter classification model (J, Z)"""
        if not self.training_data:
            print("No training data available. Please collect data first.")
            return

        dynamic_data = [sample for sample in self.training_data if sample.get('is_dynamic', False)]
        if not dynamic_data:
            print("No dynamic letter data (J, Z) available.")
            return

        X = []
        y = []
        target_len = 12
        for sample in dynamic_data:
            sequence = sample['sequence']
            resampled = self.fast_resample_sequence(sequence, target_len)
            X.append(resampled)
            y.append(self.dynamic_label_to_idx[sample['label']])

            for _ in range(3):
                aug_seq = sequence + np.random.normal(0, 0.01, sequence.shape)
                resampled_aug = self.fast_resample_sequence(aug_seq, target_len)
                X.append(resampled_aug)
                y.append(self.dynamic_label_to_idx[sample['label']])
        
        X = np.array(X)
        y = np.array(y)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.dynamic_letter_model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        self.dynamic_letter_model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.dynamic_letter_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                accuracy = 100 * correct / total
                print(f'Fast Dynamic Letter Model - Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'models/asl_fast_dynamic_letters_{timestamp}.pth'
        torch.save(self.dynamic_letter_model.state_dict(), model_path)
        print(f"Fast dynamic letter model saved to {model_path}")
    
    def train_phrase_model(self, epochs=100, batch_size=16):
        """Train the ASL phrase classification model with augmentation"""
        if not self.phrase_training_data:
            print("No phrase training data available. Please collect data first.")
            return
        
        target_len = 60
        X = []
        y = []
        for sample in self.phrase_training_data:
            sequence = sample['sequence']
            resampled = self.resample_sequence(sequence, target_len)
            X.append(resampled)
            y.append(self.phrase_to_idx[sample['label']])
            
            for _ in range(3):
                aug_seq = sequence + np.random.normal(0, 0.01, sequence.shape)
                resampled_aug = self.resample_sequence(aug_seq, target_len)
                X.append(resampled_aug)
                y.append(self.phrase_to_idx[sample['label']])
        
        X = np.array(X)
        y = np.array(y)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.phrase_model.parameters(), lr=0.0005)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        self.phrase_model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.phrase_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                accuracy = 100 * correct / total
                print(f'Phrase Model - Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'models/asl_phrases_{timestamp}.pth'
        torch.save(self.phrase_model.state_dict(), model_path)
        print(f"Phrase model saved to {model_path}")
    
    def load_letter_model(self, model_path):
        """Load a trained letter model"""
        self.letter_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.letter_model.eval()
        print(f"Letter model loaded from {model_path}")

    def load_dynamic_letter_model(self, model_path):
        """Load a trained dynamic letter model"""
        self.dynamic_letter_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.dynamic_letter_model.eval()
        print(f"Dynamic letter model loaded from {model_path}")

    def load_phrase_model(self, model_path):
        """Load a trained phrase model"""
        self.phrase_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.phrase_model.eval()
        print(f"Phrase model loaded from {model_path}")
    
    def predict_letter(self, features, sequence=None):
        """Predict ASL letter from features (static) or sequence (dynamic)"""
        if sequence is None:
            self.letter_model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features[:126]).unsqueeze(0).to(self.device)
                outputs = self.letter_model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                predicted_label = self.labels[predicted.item()]
                return predicted_label, confidence.item()
        else:
            self.dynamic_letter_model.eval()
            with torch.no_grad():
                if len(sequence) < 5:
                    return None, 0.0
                
                sequence = [f[:126] for f in sequence]  # Only hand landmarks for dynamic letters
                resampled = self.fast_resample_sequence(sequence, 12)
                sequence_tensor = torch.FloatTensor(resampled).unsqueeze(0).to(self.device)
                outputs = self.dynamic_letter_model(sequence_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                predicted_label = self.dynamic_labels[predicted.item()]
                return predicted_label, confidence.item()

    def predict_phrase(self, sequence):
        """Predict ASL phrase from sequence with resampling"""
        self.phrase_model.eval()
        with torch.no_grad():
            if len(sequence) < 20:
                return None, 0.0
            
            resampled = self.resample_sequence(sequence, 60)
            
            sequence_tensor = torch.FloatTensor(resampled).unsqueeze(0).to(self.device)
            outputs = self.phrase_model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            top_probs, top_indices = torch.topk(probabilities, 3, dim=1)
            print(f"\nTop 3 phrase predictions: {[self.phrase_labels[i] for i in top_indices[0].tolist()]}")
            print(f"Confidences: {top_probs[0].tolist()}")
            
            return self.phrase_labels[predicted.item()], confidence.item()
    
    def run_realtime_detection_headless(self, duration=30, save_interval=2):
        """Run real-time ASL detection without GUI"""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print(f"Headless ASL Detection running for {duration} seconds")
        print(f"Mode: {self.detection_mode}")
        print(f"Saving detection images every {save_interval} seconds")
        print("Press Ctrl+C to stop early")

        start_time = time.time()
        last_save_time = start_time
        image_counter = 0

        try:
            while True:
                current_time = time.time()
                if current_time - start_time > duration:
                    print(f"\nDetection completed after {duration} seconds")
                    break

                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_results = self.hands.process(rgb_frame)
                pose_results = self.pose.process(rgb_frame)
                results = self.HandPoseResults(hand_results, pose_results)

                prediction_text = "No hand or pose detected"
                confidence_text = ""

                if hand_results.multi_hand_landmarks or pose_results.pose_landmarks:
                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks[:2]:
                            brect = self.calc_bounding_rect(frame, hand_landmarks)
                            frame = self.draw_bounding_rect(frame, brect)
                            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    if pose_results.pose_landmarks:
                        brect = self.calc_bounding_rect(frame, pose_results.pose_landmarks)
                        frame = self.draw_bounding_rect(frame, brect)
                        self.mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                    features = self.extract_features(results)

                    if self.detection_mode == "letter":
                        is_dynamic = self.detect_motion(features)

                        if is_dynamic:
                            self.fast_sequence_buffer.append(features[:126])
                            if len(self.fast_sequence_buffer) >= 5:
                                sequence = list(self.fast_sequence_buffer)
                                predicted_letter, confidence = self.predict_letter(features, sequence=sequence)
                                if predicted_letter and confidence > 0.5:
                                    self.dynamic_prediction_history.append(predicted_letter)
                                
                                if self.dynamic_prediction_history:
                                    most_common = max(set(self.dynamic_prediction_history),
                                                     key=self.dynamic_prediction_history.count)
                                    prediction_text = f"Dynamic Letter: {most_common}"
                                    confidence_text = f"Confidence: {confidence:.2f}"
                                    print(f"\rDetected: {most_common} (Confidence: {confidence:.2f})", end="", flush=True)
                        else:
                            self.fast_sequence_buffer.clear()
                            self.dynamic_prediction_history.clear()
                            
                            predicted_letter, confidence = self.predict_letter(features)
                            if confidence > 0.7:
                                self.prediction_history.append(predicted_letter)
                            if self.prediction_history:
                                most_common = max(set(self.prediction_history),
                                                 key=self.prediction_history.count)
                                prediction_text = f"Letter: {most_common}"
                                confidence_text = f"Confidence: {confidence:.2f}"
                                print(f"\rDetected: {most_common} (Confidence: {confidence:.2f})", end="", flush=True)

                    elif self.detection_mode == "phrase":
                        self.sequence_buffer.append(features)
                        if len(self.sequence_buffer) >= 30:
                            sequence = list(self.sequence_buffer)
                            predicted_phrase, confidence = self.predict_phrase(sequence)
                            if predicted_phrase and confidence > 0.75:
                                self.phrase_prediction_history.append(predicted_phrase)
                            if self.phrase_prediction_history:
                                most_common = max(set(self.phrase_prediction_history),
                                                 key=self.phrase_prediction_history.count)
                                prediction_text = f"Phrase: {most_common}"
                                confidence_text = f"Confidence: {confidence:.2f}"
                                print(f"\rDetected: {most_common} (Confidence: {confidence:.2f})", end="", flush=True)

                else:
                    self.prediction_history.clear()
                    self.dynamic_prediction_history.clear()
                    self.phrase_prediction_history.clear()
                    self.fast_sequence_buffer.clear()
                    if self.detection_mode in ["letter", "phrase"]:
                        self.sequence_buffer.clear()
                    self.last_features = None
                    self.motion_frames = 0
                    self.static_frames = 0
                    prediction_text = "No hand or pose detected"

                cv2.putText(frame, f"Mode: {self.detection_mode}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, prediction_text, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                if confidence_text:
                    cv2.putText(frame, confidence_text, (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                timestamp = datetime.now().strftime("%H:%M:%S")
                cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                if current_time - last_save_time > save_interval:
                    image_path = f"output_images/detection_{image_counter:04d}.jpg"
                    cv2.imwrite(image_path, frame)
                    print(f"\nSaved: {image_path}")
                    last_save_time = current_time
                    image_counter += 1

        except KeyboardInterrupt:
            print(f"\nDetection stopped by user after {current_time - start_time:.1f} seconds")
        finally:
            cap.release()
            print(f"\nTotal images saved: {image_counter}")
            print("Images saved in 'output_images/' directory")
    
    def run_realtime_detection(self):
        """Run real-time ASL detection with body and head movement for phrases"""
        if self.headless:
            self.run_realtime_detection_headless()
            return

        cap = cv2.VideoCapture(0)

        print("Real-time ASL Detection - OPTIMIZED for J/Z and Phrase Detection with Body/Head")
        print("Press 'm' to switch between letter and phrase mode")
        print("Press 'q' to quit")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_results = self.hands.process(rgb_frame)
                pose_results = self.pose.process(rgb_frame)
                results = self.HandPoseResults(hand_results, pose_results)

                cv2.putText(frame, f"Mode: {self.detection_mode}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, "FAST J/Z & BODY/HEAD DETECTION", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if hand_results.multi_hand_landmarks or pose_results.pose_landmarks:
                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks[:2]:
                            brect = self.calc_bounding_rect(frame, hand_landmarks)
                            frame = self.draw_bounding_rect(frame, brect)
                            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    if pose_results.pose_landmarks:
                        brect = self.calc_bounding_rect(frame, pose_results.pose_landmarks)
                        frame = self.draw_bounding_rect(frame, brect)
                        self.mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                    features = self.extract_features(results)

                    if self.detection_mode == "letter":
                        is_dynamic = self.detect_motion(features)

                        if is_dynamic:
                            self.fast_sequence_buffer.append(features[:126])
                            cv2.putText(frame, f"MOTION - Buffer: {len(self.fast_sequence_buffer)}/15",
                                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                            
                            if len(self.fast_sequence_buffer) >= 5:
                                sequence = list(self.fast_sequence_buffer)
                                predicted_letter, confidence = self.predict_letter(features, sequence=sequence)
                                if predicted_letter and confidence > 0.5:
                                    self.dynamic_prediction_history.append(predicted_letter)
                                
                                if self.dynamic_prediction_history:
                                    most_common = max(set(self.dynamic_prediction_history),
                                                     key=self.dynamic_prediction_history.count)
                                    cv2.putText(frame, f"Dynamic: {most_common}",
                                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                                    cv2.putText(frame, f"Confidence: {confidence:.2f}",
                                               (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            self.fast_sequence_buffer.clear()
                            self.dynamic_prediction_history.clear()
                            
                            cv2.putText(frame, "STATIC DETECTION",
                                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
                            predicted_letter, confidence = self.predict_letter(features)
                            if confidence > 0.7:
                                self.prediction_history.append(predicted_letter)
                            if self.prediction_history:
                                most_common = max(set(self.prediction_history),
                                                 key=self.prediction_history.count)
                                cv2.putText(frame, f"Letter: {most_common}",
                                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                                cv2.putText(frame, f"Confidence: {confidence:.2f}",
                                           (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    elif self.detection_mode == "phrase":
                        self.sequence_buffer.append(features)
                        cv2.putText(frame, f"Buffer: {len(self.sequence_buffer)}/90",
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        if len(self.sequence_buffer) >= 30:
                            sequence = list(self.sequence_buffer)
                            predicted_phrase, confidence = self.predict_phrase(sequence)
                            if predicted_phrase and confidence > 0.75:
                                self.phrase_prediction_history.append(predicted_phrase)
                            if self.phrase_prediction_history:
                                most_common = max(set(self.phrase_prediction_history),
                                                 key=self.phrase_prediction_history.count)
                                cv2.putText(frame, f"Phrase: {most_common}",
                                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                                cv2.putText(frame, f"Confidence: {confidence:.2f}",
                                           (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No hand or pose detected", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    self.prediction_history.clear()
                    self.dynamic_prediction_history.clear()
                    self.phrase_prediction_history.clear()
                    self.fast_sequence_buffer.clear()
                    if self.detection_mode in ["letter", "phrase"]:
                        self.sequence_buffer.clear()
                    self.last_features = None
                    self.motion_frames = 0
                    self.static_frames = 0

                cv2.imshow('Fast ASL Detection (J/Z & Body/Head Optimized)', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    self.detection_mode = "phrase" if self.detection_mode == "letter" else "letter"
                    self.prediction_history.clear()
                    self.dynamic_prediction_history.clear()
                    self.phrase_prediction_history.clear()
                    self.fast_sequence_buffer.clear()
                    self.sequence_buffer.clear()
                    self.last_features = None
                    self.motion_frames = 0
                    self.static_frames = 0
                    print(f"Switched to {self.detection_mode} mode")

        except cv2.error as e:
            if "is not implemented" in str(e) or "GTK" in str(e):
                print("\nGUI not supported. Switching to headless mode...")
                print("This will run for 30 seconds and save images to 'output_images/' directory")
                input("Press Enter to continue...")
                self.headless = True
                self.run_realtime_detection_headless()
            else:
                raise e
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Fast ASL Detection System - Optimized for J/Z and Phrase with Body/Head')
    parser.add_argument('--mode', choices=['collect-letters', 'collect-phrases', 'train-letters',
                                         'train-dynamic-letters', 'train-phrases', 'detect'], required=True,
                       help='Mode: collect letters/phrases, train models, or run detection')
    parser.add_argument('--data', type=str, help='Path to training data file')
    parser.add_argument('--letter-model', type=str, help='Path to trained letter model file')
    parser.add_argument('--dynamic-letter-model', type=str, help='Path to trained dynamic letter model file')
    parser.add_argument('--phrase-model', type=str, help='Path to trained phrase model file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--detection-mode', choices=['letter', 'phrase'], default='letter',
                       help='Detection mode: letter or phrase')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode (no GUI, saves images)')
    parser.add_argument('--duration', type=int, default=30,
                       help='Duration in seconds for headless mode (default: 30)')
    parser.add_argument('--save-interval', type=int, default=2,
                       help='Interval in seconds between saving images in headless mode (default: 2)')
    
    args = parser.parse_args()
    
    detector = ASLDetector(headless=args.headless)
    detector.detection_mode = args.detection_mode
    
    if args.mode == 'collect-letters':
        detector.collect_training_data()
    
    elif args.mode == 'collect-phrases':
        detector.collect_phrase_data()
    
    elif args.mode == 'train-letters':
        if args.data:
            detector.load_training_data(args.data)
        detector.train_letter_model(epochs=args.epochs)
    
    elif args.mode == 'train-dynamic-letters':
        if args.data:
            detector.load_training_data(args.data)
        detector.train_dynamic_letter_model(epochs=args.epochs)
    
    elif args.mode == 'train-phrases':
        if args.data:
            detector.load_training_data(args.data)
        detector.train_phrase_model(epochs=args.epochs)
    
    elif args.mode == 'detect':
        models_loaded = 0
        if args.letter_model:
            detector.load_letter_model(args.letter_model)
            models_loaded += 1
        if args.dynamic_letter_model:
            detector.load_dynamic_letter_model(args.dynamic_letter_model)
            models_loaded += 1
        if args.phrase_model:
            detector.load_phrase_model(args.phrase_model)
            models_loaded += 1
        
        if models_loaded == 0:
            print("Please provide at least one trained model path with --letter-model, --dynamic-letter-model, or --phrase-model")
            return
        
        if args.phrase_model and not args.letter_model and not args.dynamic_letter_model:
            detector.detection_mode = "phrase"
        elif (args.letter_model or args.dynamic_letter_model) and not args.phrase_model:
            detector.detection_mode = "letter"
        
        if args.headless:
            detector.run_realtime_detection_headless(duration=args.duration, save_interval=args.save_interval)
        else:
            detector.run_realtime_detection()

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("Fast ASL Detection System - OPTIMIZED for J/Z and Phrase with Body/Head Movement")
        print("Key Improvements:")
        print("- Faster dynamic letter detection (5 frames vs 20)")
        print("- Reduced sequence length (12 vs 60 frames)")
        print("- Optimized neural network architecture")
        print("- Lower confidence thresholds for faster response")
        print("- Enhanced motion detection algorithm")
        print("- Added body and head movement detection for phrases")

        print("\nUsage examples:")

        print("\n=== FAST DETECTION (J/Z & Body/Head Optimized) ===")
        print("  python main.py --mode detect --dynamic-letter-model models/fast_dynamic_model.pth")
        print("  python main.py --mode detect --phrase-model models/asl_phrases_TIMESTAMP.pth")

        print("\n=== DATA COLLECTION ===")
        print("  python main.py --mode collect-letters")
        print("  python main.py --mode collect-phrases")

        print("\n=== MODEL TRAINING ===")
        print("  python main.py --mode train-letters --data data/asl_letters_TIMESTAMP.pkl")
        print("  python main.py --mode train-dynamic-letters --data data/asl_letters_TIMESTAMP.pkl")
        print("  python main.py --mode train-phrases --data data/asl_phrases_TIMESTAMP.pkl")

        print("\n=== DETECTION ===")
        print("  python main.py --mode detect --letter-model models/asl_letters_TIMESTAMP.pth")
        print("  python main.py --mode detect --dynamic-letter-model models/asl_fast_dynamic_letters_TIMESTAMP.pth")
        print("  python main.py --mode detect --phrase-model models/asl_phrases_TIMESTAMP.pth")
        print("  python main.py --mode detect --letter-model models/asl_letters_TIMESTAMP.pth --dynamic-letter-model models/asl_fast_dynamic_letters_TIMESTAMP.pth")

        print("\n=== KEY OPTIMIZATIONS ===")
        print("✓ Reduced buffer size: 15 frames (was 90)")
        print("✓ Faster detection: 5 frames minimum (was 20)")
        print("✓ Smaller neural network: 128 hidden units (was 256)")
        print("✓ Lower confidence threshold: 0.5 (was 0.7)")
        print("✓ Enhanced motion detection with frame counting")
        print("✓ Immediate buffer clearing when motion stops")
        print("✓ Body and head movement detection for enhanced phrase recognition")
    else:
        main()
