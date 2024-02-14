# importamos las librerias necesarias
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import models, transforms

# inicializamos el modelo
model = None

# definimos las transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# cargamos el modelo entrenado
def load_model():
    global model
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 3)
    model.load_state_dict(torch.load("src\model_servilletas.pth", map_location=torch.device('cpu')))
    model.eval()

# esta funcion se encarga de seleccionar una imagen y clasificarla
def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        class_name = classify_image(file_path)
        image = Image.open(file_path)
        image = image.resize((200, 200), Image.Resampling.LANCZOS)
        image_tk = ImageTk.PhotoImage(image)
        panel_image.config(image=image_tk)
        panel_image.image = image_tk
        text_predictions.config(text=f"{class_name}: 100%")

# funcion que clasifica la imagen
def classify_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        classes = ['en buen estado', 'no es una servilleta', 'rotas'] # definimos 3 opiones para clasificar
        return classes[predicted[0]]

# funcion para configurar la interfaz grafica con tkinter
def configure_gui():
    window.title("Clasificador de Servilletas con PyTorch y Tkinter")
    window.geometry('500x550')

    global btn_cargar_image, panel_image, text_predictions
    btn_cargar_image = tk.Button(window, text="Seleccionar imagen", command=select_image)
    btn_cargar_image.pack(pady=10)

    panel_image = tk.Label(window)
    panel_image.pack(pady=10)

    text_predictions = tk.Label(window, text="", justify=tk.LEFT)
    text_predictions.pack(pady=10)

if __name__ == "__main__":
    window = tk.Tk()
    load_model()
    configure_gui()
    window.mainloop()
