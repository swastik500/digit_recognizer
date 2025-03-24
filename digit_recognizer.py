import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageDraw
import pickle
import os
from sklearn.neural_network import MLPClassifier
import pyttsx3

class DigitRecognizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Digit Recognizer")
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        
        # Initialize model and load if exists
        self.model_file = 'digit_model.pkl'
        if os.path.exists(self.model_file):
            with open(self.model_file, 'rb') as f:
                saved_data = pickle.load(f)
                self.model = saved_data['model']
                self.X_train = saved_data['X_train']
                self.y_train = saved_data['y_train']
        else:
            self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
            self.X_train = []
            self.y_train = []
        
        # Canvas setup
        self.canvas = tk.Canvas(self.root, width=280, height=280, bg='black')
        self.canvas.grid(row=0, column=0, columnspan=3, padx=5, pady=5)
        self.canvas.bind('<B1-Motion>', self.paint)
        self.image = Image.new('L', (280, 280), 'black')
        self.draw = ImageDraw.Draw(self.image)
        
        # Controls
        self.digit_var = tk.StringVar()
        digit_label = tk.Label(self.root, text="Select digit (0-9):")
        digit_label.grid(row=1, column=0)
        digit_combo = ttk.Combobox(self.root, textvariable=self.digit_var, values=[str(i) for i in range(10)], width=5, state='readonly')
        digit_combo.grid(row=1, column=1)
        digit_combo.set('')  # Set initial empty value
        
        # Buttons
        train_btn = tk.Button(self.root, text="Train", command=self.train)
        train_btn.grid(row=2, column=0)
        
        predict_btn = tk.Button(self.root, text="Predict", command=self.predict)
        predict_btn.grid(row=2, column=1)
        
        clear_btn = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        clear_btn.grid(row=2, column=2)

        reset_btn = tk.Button(self.root, text="Reset Training", command=self.reset_training)
        reset_btn.grid(row=2, column=3)
        
        # Prediction label
        self.pred_label = tk.Label(self.root, text="Prediction: None")
        self.pred_label.grid(row=3, column=0, columnspan=4)
        
    def paint(self, event):
        # Increase the brush size for better visibility
        brush_size = 15
        x1, y1 = (event.x - brush_size), (event.y - brush_size)
        x2, y2 = (event.x + brush_size), (event.y + brush_size)
        
        # Draw on canvas
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
        
        # Draw on PIL Image
        self.draw.ellipse([x1, y1, x2, y2], fill='white')
        self.draw.line([x1, y1, x2, y2], fill='white', width=5)
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new('L', (280, 280), 'black')
        self.draw = ImageDraw.Draw(self.image)
        self.pred_label.config(text="Prediction: None")
        
    def preprocess_image(self):
        resized_image = self.image.resize((28, 28))
        img_array = np.array(resized_image)
        img_array = img_array.reshape(1, -1) / 255.0
        return img_array
        
    def train(self):
        try:
            digit = int(self.digit_var.get())
            if 0 <= digit <= 9:
                img_array = self.preprocess_image()
                self.X_train.append(img_array[0])
                self.y_train.append(digit)
                
                if len(self.X_train) > 1:
                    self.model.fit(self.X_train, self.y_train)
                    # Save the model and training data
                    with open(self.model_file, 'wb') as f:
                        pickle.dump({
                            'model': self.model,
                            'X_train': self.X_train,
                            'y_train': self.y_train
                        }, f)
                    self.pred_label.config(text=f"Training successful! Total samples: {len(self.y_train)}")
                else:
                    self.pred_label.config(text="Need more samples!")
            else:
                self.pred_label.config(text="Please enter a digit between 0-9")
        except ValueError:
            self.pred_label.config(text="Invalid input!")
            
    def predict(self):
        if len(self.X_train) > 1:
            img_array = self.preprocess_image()
            prediction = self.model.predict(img_array)
            predicted_digit = prediction[0]
            self.pred_label.config(text=f"Prediction: {predicted_digit}")
            # Speak the predicted digit
            self.engine.say(f"The predicted digit is {predicted_digit}")
            self.engine.runAndWait()
        else:
            self.pred_label.config(text="Please train the model first!")
            
    def reset_training(self):
        if os.path.exists(self.model_file):
            os.remove(self.model_file)
        self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
        self.X_train = []
        self.y_train = []
        self.pred_label.config(text="Training data reset successfully!")
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = DigitRecognizer()
    app.run()