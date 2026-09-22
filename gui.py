import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image
from daltonizer import daltonize_image

class DaltonizerGUI:

    def __init__(self, root):

        self.root = root
        self.root.title("Daltonizer")
        self.root.geometry("550x430")
        self.root.resizable(False, False)

        # Variables
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.cvd_type = tk.StringVar(value="Deuteranopia")
        self.strength = tk.IntVar(value=100)
        self.status = tk.StringVar(value="Ready")
        self.progress_value = tk.DoubleVar(value=0)

        # Build interface
        self.create_widgets()

    # Interface construction
    def create_widgets(self):

        # Main Frame
        main = ttk.Frame(self.root, padding=20)
        main.pack(fill="both", expand=True)

        # Title and Description
        title = ttk.Label(main, text="Daltonizer", font=("TkDefaultFont", 18, "bold"))
        title.pack(pady=(0, 20))

        description = ttk.Label(main,
            text=(
                "Convert images to make colours easier to "
                "distinguish for people with colour-vision deficiency."
            ),
            wraplength=480, justify="center"
        )
        description.pack(pady=(0, 20))

        # Input
        input_frame = ttk.LabelFrame(main, text="Input", padding=10)
        input_frame.pack(fill="x", pady=5)
        input_entry = ttk.Entry(input_frame, textvariable=self.input_path)
        input_entry.pack(side="left", fill="x", expand=True)

        ttk.Button(input_frame, text="Select File", command=self.select_file
                   ).pack(side="left", padx=(5, 0))
        ttk.Button(input_frame, text="Select Folder", command=self.select_folder
                   ).pack(side="left", padx=(5, 0))

        # Output
        output_frame = ttk.LabelFrame(main, text="Output Folder", padding=10)
        output_frame.pack(fill="x", pady=5)
        output_entry = ttk.Entry(output_frame, textvariable=self.output_path)
        output_entry.pack(side="left", fill="x", expand=True)

        ttk.Button(output_frame, text="Browse", command=self.select_output_folder
                   ).pack(side="left", padx=(5, 0))

        # Correction Settings
        settings_frame = ttk.LabelFrame(main, text="Correction", padding=10)
        settings_frame.pack(fill="x", pady=10)

        ttk.Label(settings_frame, text="Colour-vision Deficiency:"
                  ).grid(row=0, column=0, sticky="w", padx=5, pady=5)

        cvd_menu = ttk.Combobox(settings_frame, textvariable=self.cvd_type,
            values=[
                "Protanopia",
                "Deuteranopia",
                "Tritanopia"
            ],
            state="readonly", width=18
        )
        cvd_menu.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        # Correction Strength
        ttk.Label(settings_frame, text="Correction Strength:"
                  ).grid(row=1, column=0, sticky="w", padx=5, pady=5)

        strength_frame = ttk.Frame(settings_frame)
        strength_frame.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        self.strength_scale = ttk.Scale(strength_frame, from_=0, to=100, orient="horizontal", command=self.update_strength)
        self.strength_scale.set(self.strength.get())
        self.strength_scale.pack(side="left", fill="x", expand=True)

        self.strength_label = ttk.Label(strength_frame, text="100%")
        self.strength_label.pack(side="left", padx=(10, 0))

        # Progress Bar
        self.progress = ttk.Progressbar(main, variable=self.progress_value, maximum=100)
        self.progress.pack(fill="x", pady=(15, 5))

        status_label = ttk.Label(main, textvariable=self.status)
        status_label.pack(pady=(0, 10))

        # Process button
        self.process_button = ttk.Button(main, text="Process", command=self.start_processing)
        self.process_button.pack(ipadx=30, ipady=5)


    # File selection
    def select_file(self):

        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                (
                    "Image files",
                    "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"
                ),
                (
                    "PNG files",
                    "*.png"
                ),
                (
                    "JPEG files",
                    "*.jpg *.jpeg"
                ),
                (
                    "All files",
                    "*.*"
                )
            ]
        )

        if path:
            self.input_path.set(path)
            directory = os.path.dirname(path)
            self.output_path.set(os.path.join(directory, "Daltonized"))


    def select_folder(self):

        path = filedialog.askdirectory(title="Select image folder")

        if path:
            self.input_path.set(path)
            self.output_path.set(os.path.join(path, "Daltonized"))


    def select_output_folder(self):

        path = filedialog.askdirectory(title="Select output folder")

        if path:
            self.output_path.set(path)

    # Strength
    def update_strength(self, value):

        value = int(float(value))
        self.strength.set(value)
        self.strength_label.config(text=f"{value}%")


    # Find images
    def get_images(self, path):

        extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".bmp",
            ".tif",
            ".tiff"
        }

        images = []

        if os.path.isfile(path):

            if os.path.splitext(path)[1].lower() in extensions:

                images.append(path)

        elif os.path.isdir(path):

            for root, _, files in os.walk(path):

                for filename in files:

                    extension = os.path.splitext(filename)[1].lower()

                    if extension in extensions:

                        images.append(os.path.join(root, filename))

        return images


    # Start processing
    def start_processing(self):

        input_path = self.input_path.get().strip()
        output_path = self.output_path.get().strip()

        if not input_path:

            messagebox.showerror("No input selected", "Please select an image or folder.")
            return

        if not os.path.exists(input_path):

            messagebox.showerror("Invalid input", "The selected file or folder does not exist.")
            return

        if not output_path:

            messagebox.showerror("No output folder", "Please select an output folder.")
            return

        images = self.get_images(input_path)

        if not images:

            messagebox.showerror("No images", "No supported image files were found.")
            return

        os.makedirs(output_path, exist_ok=True)
        self.process_button.config(state="disabled")
        self.progress_value.set(0)
        self.status.set(f"Processing 0 / {len(images)}...")

        thread = threading.Thread(
            target=self.process_images,
            args=(
                images,
                input_path,
                output_path
            ),
            daemon=True
        )

        thread.start()


    # Process images
    def process_images(self, images, input_path, output_folder):

        total = len(images)
        input_is_folder = os.path.isdir(input_path)

        try:

            for index, image_path in enumerate(images):

                with Image.open(image_path) as source:

                    image = source.convert("RGBA")

                corrected = daltonize_image(image, self.cvd_type.get(), self.strength.get())

                if input_is_folder:

                    # Get the image's path relative to the selected input folder.
                    relative_path = os.path.relpath(image_path, input_path)
                    output_path = os.path.join(output_folder, relative_path)

                else:

                    # Single files simply go directly into the output directory.
                    output_path = os.path.join(output_folder, os.path.basename(image_path))

                output_directory = os.path.dirname(output_path)
                os.makedirs(output_directory, exist_ok=True)

                corrected.save(output_path)

                completed = index + 1
                self.root.after(0, self.update_progress, completed, total)

            self.root.after(0, self.processing_finished, total)

        except Exception as exc:

            self.root.after(0, self.processing_failed, str(exc))

    # Progress
    def update_progress(self, completed, total):

        percentage = (completed / total) * 100
        self.progress_value.set(percentage)
        self.status.set(f"Processing {completed} / {total}...")


    # Finished
    def processing_finished(self, total):

        self.progress_value.set(100)
        self.status.set(f"Finished — {total} image(s) processed.")
        self.process_button.config(state="normal")
        messagebox.showinfo("Complete", f"{total} image(s) were successfully processed.")


    # Failed
    def processing_failed(self, error):

        self.process_button.config(state="normal")
        self.status.set("Processing failed.")
        messagebox.showerror("Processing error", error)



if __name__ == "__main__":

    root = tk.Tk()
    app = DaltonizerGUI(root)
    root.mainloop()