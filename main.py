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

class ASLNet(nn.Module):
    """Neural network for ASL gesture classification"""
    def __init__(self, input_size=63, hidden_size=128, num_classes=26):  # 21 landmarks * 3 coords = 63
        super(ASLNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 64)
        self.fc4 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ASLDetector:
    def __init__(self, headless=False):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # ASL alphabet labels
        self.labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ASLNet(input_size=63, num_classes=len(self.labels))
        self.model.to(self.device)
        
        # Data storage
        self.training_data = []
        self.current_label = None
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=10)
        
        # Headless mode
        self.headless = headless
        
        # Create directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        if headless:
            os.makedirs('output_images', exist_ok=True)
        
    def extract_landmarks(self, hand_landmarks):
        """Extract hand landmarks as feature vector"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks, dtype=np.float32)
    
    def normalize_landmarks(self, landmarks):
        """Normalize landmarks relative to wrist position"""
        if len(landmarks) != 63:
            return landmarks
        
        # Reshape to get individual landmarks
        landmarks = landmarks.reshape(21, 3)
        
        # Get wrist position (first landmark)
        wrist = landmarks[0]
        
        # Normalize all landmarks relative to wrist
        normalized = landmarks - wrist
        
        # Calculate scale (distance from wrist to middle finger tip)
        if len(landmarks) > 12:
            scale = np.linalg.norm(normalized[12])  # Middle finger tip
            if scale > 0:
                normalized = normalized / scale
        
        return normalized.flatten()
    
    def collect_training_data(self):
        """Collect training data for ASL gestures"""
        if self.headless:
            print("Error: Data collection not available in headless mode")
            return
            
        cap = cv2.VideoCapture(0)
        current_letter_idx = 0
        samples_per_letter = 100
        samples_collected = 0
        
        print("ASL Training Data Collection")
        print("Instructions:")
        print("- Show the gesture for each letter")
        print("- Press SPACE to start collecting for current letter")
        print("- Press 'n' for next letter")
        print("- Press 'q' to quit")
        print(f"\nCurrent letter: {self.labels[current_letter_idx]}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Extract features
                    landmarks = self.extract_landmarks(hand_landmarks)
                    normalized_landmarks = self.normalize_landmarks(landmarks)
                    
                    # Display current letter and sample count
                    cv2.putText(frame, f"Letter: {self.labels[current_letter_idx]}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Samples: {samples_collected}/{samples_per_letter}", 
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Collect sample if space is pressed
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):
                        self.training_data.append({
                            'landmarks': normalized_landmarks,
                            'label': self.labels[current_letter_idx]
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
                    
                    elif key == ord('n'):
                        current_letter_idx += 1
                        samples_collected = 0
                        if current_letter_idx >= len(self.labels):
                            print("Data collection completed!")
                            break
                        print(f"\nNext letter: {self.labels[current_letter_idx]}")
                    
                    elif key == ord('q'):
                        break
            else:
                cv2.putText(frame, "No hand detected", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('ASL Data Collection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save collected data
        if self.training_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f'data/asl_data_{timestamp}.pkl', 'wb') as f:
                pickle.dump(self.training_data, f)
            print(f"Saved {len(self.training_data)} samples to data/asl_data_{timestamp}.pkl")
    
    def load_training_data(self, data_path):
        """Load training data from file"""
        with open(data_path, 'rb') as f:
            self.training_data = pickle.load(f)
        print(f"Loaded {len(self.training_data)} training samples")
    
    def train_model(self, epochs=100, batch_size=32):
        """Train the ASL classification model"""
        if not self.training_data:
            print("No training data available. Please collect data first.")
            return
        
        # Prepare training data
        X = []
        y = []
        for sample in self.training_data:
            X.append(sample['landmarks'])
            y.append(self.label_to_idx[sample['label']])
        
        X = np.array(X)
        y = np.array(y)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            if (epoch + 1) % 10 == 0:
                accuracy = 100 * correct / total
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%')
        
        # Save the trained model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'models/asl_model_{timestamp}.pth'
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load a trained model"""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {model_path}")
    
    def predict_gesture(self, landmarks):
        """Predict ASL gesture from landmarks"""
        self.model.eval()
        with torch.no_grad():
            landmarks_tensor = torch.FloatTensor(landmarks).unsqueeze(0).to(self.device)
            outputs = self.model(landmarks_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            return self.labels[predicted.item()], confidence.item()
    
    def run_realtime_detection_headless(self, duration=30, save_interval=2):
        """Run real-time ASL detection without GUI (headless mode)"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print(f"Headless ASL Detection running for {duration} seconds")
        print(f"Saving detection images every {save_interval} seconds")
        print("Press Ctrl+C to stop early")
        
        start_time = time.time()
        last_save_time = start_time
        image_counter = 0
        
        try:
            while True:
                current_time = time.time()
                
                # Check if duration exceeded
                if current_time - start_time > duration:
                    print(f"\nDetection completed after {duration} seconds")
                    break
                
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    continue
                    
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                prediction_text = "No hand detected"
                confidence_text = ""
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks
                        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        
                        # Extract and normalize landmarks
                        landmarks = self.extract_landmarks(hand_landmarks)
                        normalized_landmarks = self.normalize_landmarks(landmarks)
                        
                        # Make prediction
                        predicted_letter, confidence = self.predict_gesture(normalized_landmarks)
                        
                        # Smooth predictions
                        if confidence > 0.7:  # Only consider high-confidence predictions
                            self.prediction_history.append(predicted_letter)
                        
                        # Get most common prediction from history
                        if self.prediction_history:
                            most_common = max(set(self.prediction_history), key=self.prediction_history.count)
                            prediction_text = f"Letter: {most_common}"
                            confidence_text = f"Confidence: {confidence:.2f}"
                            
                            # Print to console
                            print(f"\rDetected: {most_common} (Confidence: {confidence:.2f})", end="", flush=True)
                else:
                    self.prediction_history.clear()
                
                # Add text to frame
                cv2.putText(frame, prediction_text, (10, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                if confidence_text:
                    cv2.putText(frame, confidence_text, (10, 90), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Add timestamp
                timestamp = datetime.now().strftime("%H:%M:%S")
                cv2.putText(frame, timestamp, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Save image at intervals
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
        """Run real-time ASL detection"""
        if self.headless:
            self.run_realtime_detection_headless()
            return
            
        cap = cv2.VideoCapture(0)
        
        print("Real-time ASL Detection")
        print("Press 'q' to quit")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks
                        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        
                        # Extract and normalize landmarks
                        landmarks = self.extract_landmarks(hand_landmarks)
                        normalized_landmarks = self.normalize_landmarks(landmarks)
                        
                        # Make prediction
                        predicted_letter, confidence = self.predict_gesture(normalized_landmarks)
                        
                        # Smooth predictions
                        if confidence > 0.7:  # Only consider high-confidence predictions
                            self.prediction_history.append(predicted_letter)
                        
                        # Get most common prediction from history
                        if self.prediction_history:
                            most_common = max(set(self.prediction_history), key=self.prediction_history.count)
                            
                            # Display prediction
                            cv2.putText(frame, f"Letter: {most_common}", 
                                      (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                            cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                                      (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No hand detected", (10, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    self.prediction_history.clear()
                
                cv2.imshow('ASL Real-time Detection', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
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
    parser = argparse.ArgumentParser(description='ASL Detection System')
    parser.add_argument('--mode', choices=['collect', 'train', 'detect'], required=True,
                       help='Mode: collect data, train model, or run detection')
    parser.add_argument('--data', type=str, help='Path to training data file')
    parser.add_argument('--model', type=str, help='Path to trained model file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--headless', action='store_true', 
                       help='Run in headless mode (no GUI, saves images)')
    parser.add_argument('--duration', type=int, default=30,
                       help='Duration in seconds for headless mode (default: 30)')
    parser.add_argument('--save-interval', type=int, default=2,
                       help='Interval in seconds between saving images in headless mode (default: 2)')
    
    args = parser.parse_args()
    
    detector = ASLDetector(headless=args.headless)
    
    if args.mode == 'collect':
        detector.collect_training_data()
    
    elif args.mode == 'train':
        if args.data:
            detector.load_training_data(args.data)
        detector.train_model(epochs=args.epochs)
    
    elif args.mode == 'detect':
        if args.model:
            detector.load_model(args.model)
        else:
            print("Please provide a trained model path with --model")
            return
        
        if args.headless:
            detector.run_realtime_detection_headless(duration=args.duration, save_interval=args.save_interval)
        else:
            detector.run_realtime_detection()

if __name__ == "__main__":

    import sys
    if len(sys.argv) == 1:
        print("ASL Detection System")
        print("Usage examples:")
        print("  python main.py --mode collect                    # Collect training data")
        print("  python main.py --mode train --data data.pkl      # Train model")
        print("  python main.py --mode detect --model model.pth   # Run detection")
        print("  python main.py --mode detect --model model.pth --headless --duration 60  # Headless mode")
        print("\nFor first time setup:")
        print("1. Collect data: python main.py --mode collect")
        print("2. Train model: python main.py --mode train --data data/asl_data_TIMESTAMP.pkl")
        print("3. Run detection: python main.py --mode detect --model models/asl_model_TIMESTAMP.pth")
        print("\nHeadless mode (for systems without GUI):")
        print("  python main.py --mode detect --model model.pth --headless --duration 60 --save-interval 3")
    else:
        main()