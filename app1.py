import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

# Initialize Tkinter
root = tk.Tk()
root.title("Helmet Detection")

# Set window size and background color
root.geometry("800x600")
root.configure(bg='#2c3e50')  # Dark blue-gray background

# Initialize YOLO model
best_weights_path = r"C:\Users\latitude\OneDrive\Desktop\helmete Detection\runs\detect\train4\weights\best.pt"
model = YOLO(best_weights_path)
class_names = ['non-helmet', 'helmet']

# Global variables for the image
img = None
img_display = None
uploaded_image_path = None

# Create a label to display the file path
pathlabel = tk.Label(root, bg='DarkOrange1', fg='white', font=("Helvetica", 12, "italic"))
pathlabel.place(x=400, y=650)

# Function to upload an image
def upload_image():
    global uploaded_image_path
    uploaded_image_path = filedialog.askopenfilename()
    if not uploaded_image_path:
        return
    # Update the label with the path of the uploaded image
    pathlabel.config(text=uploaded_image_path)
    messagebox.showinfo("Image Uploaded", "Image successfully uploaded. Click 'Load Image' to process the image.")

# Function to load and process the uploaded image
def load_image():
    global img, img_display, uploaded_image_path
    if uploaded_image_path is None:
        messagebox.showwarning("Warning", "No image uploaded!")
        return

    # Load the image
    img = cv2.imread(uploaded_image_path)

    # Perform inference
    results = model.predict(uploaded_image_path)

    # Process each result
    for result in results:
        boxes = result.boxes.xyxy.numpy()
        scores = result.boxes.conf.numpy()
        class_ids = result.boxes.cls.numpy().astype(int)

        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = class_names[class_id]
            # Set color based on class
            color = (0, 255, 0) if label == 'helmet' else (255, 0, 0)
            thickness = 2

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            label_position = (x1, y1 - 10)
            cv2.putText(img, f"{label} {score:.2f}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Convert image to RGB (Tkinter expects RGB images)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_display = ImageTk.PhotoImage(img_pil)

    # Display the image in the Tkinter window
    canvas.create_image(0, 0, anchor=tk.NW, image=img_display)
    canvas.config(scrollregion=canvas.bbox(tk.ALL))

# Function to save the image
def save_image():
    if img is None:
        messagebox.showwarning("Warning", "No image loaded!")
        return

    save_path = filedialog.asksaveasfilename(defaultextension=".jpg")
    if save_path:
        cv2.imwrite(save_path, img)
        messagebox.showinfo("Info", f"Image saved to {save_path}")

# Function to clear the canvas
def clear_image():
    global img, img_display, uploaded_image_path
    canvas.delete("all")
    img = None
    img_display = None
    uploaded_image_path = None
    pathlabel.config(text="")  # Clear the path label

# Function to show an About dialog
def show_about():
    messagebox.showinfo("About", "Helmet Detection Application\n Python Version 3.7.2\nDeveloped using YOLOv8 and Tkinter")

def create_heading():
    # Create a canvas for the heading with a gradient background
    heading_canvas = tk.Canvas(root, width=800, height=100, bg="#2c3e50", bd=0, highlightthickness=0)
    heading_canvas.pack()

    # Create a gradient background
    gradient_colors = ['#2c3e50', '#34495e', '#4a5b6e', '#607d8b', '#78909c']
    for i, color in enumerate(gradient_colors):
        heading_canvas.create_line(0, i*20, 800, i*20, fill=color, width=20)
    
    # Add a rounded rectangle with shadow
    heading_canvas.create_rectangle(20, 20, 780, 90, fill='#ecf0f1', outline='', width=0, tags="rectangle")
    heading_canvas.create_rectangle(22, 22, 782, 92, fill='#bdc3c7', outline='', width=0, tags="shadow")

    # Add text with shadow effect
    heading_canvas.create_text(400, 55, text="Helmet Detection", font=("Helvetica", 36, "bold"), fill="#2c3e50", tags="text")
    heading_canvas.create_text(398, 53, text="Helmet Detection", font=("Helvetica", 36, "bold"), fill="#ecf0f1", tags="text")

    # Add a decorative element (optional)
    heading_canvas.create_oval(20, 20, 80, 80, fill='#16a085', outline='', tags="decorative")
    heading_canvas.create_text(50, 50, text="⛑", font=("Helvetica", 20, "bold"), fill="#ecf0f1", tags="decorative")

    # Adjust the z-order to ensure the text is on top
    heading_canvas.tag_raise("text")

# Create the heading
create_heading()


# Create a canvas to display the image with a border
canvas = tk.Canvas(root, width=640, height=480, bg="#34495e", bd=2, relief="ridge")
canvas.pack(pady=10)

# Customize buttons with colors and font
btn_upload = tk.Button(root, text="Upload Image", command=upload_image, font=("Helvetica", 14, "bold"), bg="#16a085", fg="#ecf0f1")
btn_upload.pack(side=tk.LEFT, padx=50, pady=10)

btn_load = tk.Button(root, text="Load Image", command=load_image, font=("Helvetica", 14, "bold"), bg="#27ae60", fg="#ecf0f1")
btn_load.pack(side=tk.LEFT, padx=50, pady=10)

btn_save = tk.Button(root, text="Save Image", command=save_image, font=("Helvetica", 14, "bold"), bg="#2980b9", fg="#ecf0f1")
btn_save.pack(side=tk.LEFT, padx=50, pady=10)

btn_clear = tk.Button(root, text="Clear Image", command=clear_image, font=("Helvetica", 14, "bold"), bg="#e74c3c", fg="#ecf0f1")
btn_clear.pack(side=tk.LEFT, padx=50, pady=10)

btn_about = tk.Button(root, text="About", command=show_about, font=("Helvetica", 14, "bold"), bg="#8e44ad", fg="#ecf0f1")
btn_about.pack(side=tk.LEFT, padx=50, pady=10)

# Run the Tkinter main loop
root.mainloop()
