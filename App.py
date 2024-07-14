import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk, Image
from authtoken import auth_token
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Create the app
app = ctk.CTk()
app.geometry("532x632")
app.title("Text to Image Generator")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(master=app, height=40, width=512, text_color="black", fg_color="white")
prompt.place(x=10, y=10)
prompt.configure(font=("Arial", 20))  # Update font after initialization

# Initialize the label without any initial text
lmain = ctk.CTkLabel(master=app, height=512, width=512, text="")
lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
pipe.to(device)

def generate():
    with autocast(device):
        output = pipe(prompt.get(), guidance_scale=8.5)
        
        # Check if images are present in output
        if output.images:
            # Assuming only one image is generated, you can modify this based on your use case
            generated_image = output.images[0]
            generated_image.save('generatedimage.png')
            img = Image.open('generatedimage.png')
            img = ImageTk.PhotoImage(img)
            lmain.configure(image=img, text="")  # Clear any previous text
            lmain.image = img  # Keep a reference to avoid garbage collection
        else:
            print("No images found in pipeline output.")

trigger = ctk.CTkButton(master=app, height=40, width=120, text="Generate", text_color="white", fg_color="blue", command=generate)
trigger.place(x=206, y=60)
trigger.configure(font=("Arial", 20))  # Update font after initialization

app.mainloop()
