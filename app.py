import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
import os


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class PlantDiseaseClassifier:
    def __init__(self, model_path):
        self.window = tk.Tk()
        self.window.title("Plant Disease Classifier")
        self.window.geometry("800x600")

        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNN(num_classes=2)
        # self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        # Upload button
        self.upload_btn = tk.Button(self.window, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(pady=10)

        # Image display
        self.image_label = tk.Label(self.window)
        self.image_label.pack(pady=10)

        # Classify button
        self.classify_btn = tk.Button(self.window, text="Classify", command=self.classify_image)
        self.classify_btn.pack(pady=10)
        self.classify_btn['state'] = 'disabled'

        # Result display
        self.result_label = tk.Label(self.window, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)

        # Confidence scores display
        self.confidence_label = tk.Label(self.window, text="", font=("Arial", 12))
        self.confidence_label.pack(pady=5)

    def preprocess_image(self, image):
        # Convert PIL image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Resize
        img_resized = cv2.resize(opencv_image, (224, 224), interpolation=cv2.INTER_AREA)

        # Apply median filtering
        img_denoised = cv2.medianBlur(img_resized, 3)

        # Apply histogram equalization
        b, g, r = cv2.split(img_denoised)
        b_eq = cv2.equalizeHist(b)
        g_eq = cv2.equalizeHist(g)
        r_eq = cv2.equalizeHist(r)
        img_equalized = cv2.merge([b_eq, g_eq, r_eq])

        # Convert to tensor
        img_tensor = torch.FloatTensor(img_equalized).permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")
        ])

        if file_path:
            # Load and display image
            image = Image.open(file_path)
            image = image.resize((300, 300))  # Resize for display
            photo = ImageTk.PhotoImage(image)

            self.image_label.configure(image=photo)
            self.image_label.image = photo  # Keep a reference

            # Store the original image for classification
            self.current_image = image

            # Enable classify button
            self.classify_btn['state'] = 'normal'

            # Clear previous results
            self.result_label.config(text="")
            self.confidence_label.config(text="")

    def classify_image(self):
        if hasattr(self, 'current_image'):
            # Preprocess image
            img_tensor = self.preprocess_image(self.current_image)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(img_tensor.to(self.device))
                probabilities = torch.softmax(outputs, dim=1)[0]
                _, predicted = outputs.max(1)

                # Get prediction and confidence
                label = 'Healthy' if predicted.item() == 0 else 'Diseased'
                healthy_conf = probabilities[0].item() * 100
                diseased_conf = probabilities[1].item() * 100

                # Update GUI with results
                result_color = "green" if label == "Healthy" else "red"
                self.result_label.config(text=f"Prediction: {label}", fg=result_color)
                self.confidence_label.config(
                    text=f"Confidence Scores:\nHealthy: {healthy_conf:.2f}%\nDiseased: {diseased_conf:.2f}%"
                )

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    app = PlantDiseaseClassifier('model/best_accuracy_model_new.pth')
    app.run()
