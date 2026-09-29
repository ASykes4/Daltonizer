import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageOps, ImageTk
from daltonizer import daltonize_image, simulate_cvd

class DaltonizerGUI:

    def __init__(self, root):

        self.root = root
        self.root.title("Daltonizer")
        self.root.resizable(True, True)

        # Variables
        self.running = False
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.cvd_type = tk.StringVar(value="Deuteranopia")
        self.cvd_type.trace_add("write", self.update_cvd_preview)
        self.strength = tk.IntVar(value=100)
        self.status = tk.StringVar(value="Ready")
        self.progress_value = tk.DoubleVar(value=0)

        # Image display
        self.preview_source = None
        self.preview_original = None
        self.preview_simulated = None
        self.preview_corrected = None
        self.preview_photo_original = None
        self.preview_photo_simulated = None
        self.preview_photo_corrected = None

        # Build interface
        self.create_widgets()

        # Configure the window close callback.
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

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
        self.strength_scale.pack(side="left", fill="x", expand=True)

        self.strength_label = ttk.Label(strength_frame, text="100%")
        self.strength_label.pack(side="left", padx=(10, 0))
        self.strength_scale.set(self.strength.get())

        self.create_preview_widgets(main)

        # Process button
        status_label = ttk.Label(main, textvariable=self.status)
        status_label.pack(pady=(0, 10))

        self.process_button = ttk.Button(main, text="Process", command=self.start_processing)
        self.process_button.pack(ipadx=30, ipady=5)

        # Progress Bar
        self.progress = ttk.Progressbar(main, variable=self.progress_value, maximum=100)
        self.progress.pack(fill="x", pady=(15, 5))


    # Create the three-image preview area.
    # Input: main - ttk.Frame, parent frame for the preview
    # Output: None, creates the original, simulated, and corrected preview widgets
    def create_preview_widgets(self, main: ttk.Frame) -> None:
        """Create the three-image preview area.

        Args:
            main: Parent frame containing the preview.

        Returns:
            None.
        """
        preview_frame = ttk.LabelFrame(main, text="Preview")
        preview_frame.pack(fill="both", expand=True, pady=5)

        preview_columns = (
            ("Original", "preview_original_label"),
            ("Colour-Blind Simulation", "preview_simulated_label"),
            ("Colour Corrected", "preview_corrected_label"),
        )

        for column, (title, attribute) in enumerate(preview_columns):
            ttk.Label(preview_frame, text=title).grid(
                row=0,
                column=column,
            )

            label = ttk.Label(
                preview_frame,
                text="No image selected",
                anchor="center",
                width=40
            )
            label.grid(row=1, column=column, sticky="nsew")

            setattr(self, attribute, label)

            preview_frame.columnconfigure(column, weight=1)

        preview_frame.rowconfigure(1, weight=1)


    # Resize an image to fit within the preview dimensions without distortion.
    # Input: image - Image.Image, PIL image to resize
    # Input: maximum_size - tuple[int, int], maximum preview width and height
    # Output: Image.Image, resized copy that fits within maximum_size
    def prepare_preview_image(self, image: Image.Image, maximum_size: tuple[int, int] = (350, 260)) -> Image.Image:
        """Prepare an image for display in a preview widget.

        Args:
            image: PIL image to resize.
            maximum_size: Maximum width and height of the preview.

        Returns:
            A resized copy of the image that preserves its aspect ratio.
        """
        preview = image.copy()

        width, height = image.size
        maximum_width, maximum_height = maximum_size

        scale = min(maximum_width / width, maximum_height / height)

        if scale >= 1:
            new_size = (
                max(1, round(width * scale)),
                max(1, round(height * scale))
            )
            return ImageOps.contain(preview, new_size, Image.Resampling.NEAREST)

        new_size = (
            max(1, round(width * scale)),
            max(1, round(height * scale))
        )
        return ImageOps.contain(preview, maximum_size, Image.Resampling.LANCZOS)


    # Load an image into the preview and generate its initial transformations.
    # Input: image_path - str, path to the image to preview
    # Output: None, loads the source image and updates all three previews
    def load_preview(self, image_path: str) -> None:
        """Load an image and display its three preview versions.

        Args:
            image_path: Path to the image that should be previewed.

        Returns:
            None.
        """
        with Image.open(image_path) as source:
            self.preview_source = source.convert("RGBA")

        self.update_preview()


    # Generate and display the simulation and correction previews.
    # Input: None, uses the current source image and correction settings
    # Output: None, updates all three preview widgets
    def update_preview(self) -> None:
        """Regenerate and display all three preview images.

        Returns:
            None.
        """
        if self.preview_source is None:
            return

        preview = self.prepare_preview_image(self.preview_source)

        simulated = simulate_cvd(
            preview,
            self.cvd_type.get(),
            self.strength.get()
        )

        corrected = daltonize_image(
            preview,
            self.cvd_type.get(),
            self.strength.get()
        )

        self.preview_photo_original = ImageTk.PhotoImage(preview)
        self.preview_photo_simulated = ImageTk.PhotoImage(simulated)
        self.preview_photo_corrected = ImageTk.PhotoImage(corrected)

        self.preview_original_label.config(image=self.preview_photo_original, text="")
        self.preview_simulated_label.config(image=self.preview_photo_simulated, text="")
        self.preview_corrected_label.config(image=self.preview_photo_corrected, text="")


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
            self.load_preview(path)


    # Select an input folder and load its first image into the preview.
    # Input: None, obtains the folder through the Tkinter directory dialog
    # Output: None, updates the input/output paths and preview
    def select_folder(self) -> None:
        """Select an input folder and preview its first supported image.

        Returns:
            None.
        """
        path = filedialog.askdirectory(title="Select image folder")

        if not path:
            return

        self.input_path.set(path)
        self.output_path.set(os.path.join(path, "Daltonized"))

        images = self.get_images(path)

        if images:
            self.load_preview(images[0])
        else:
            self.clear_preview()


    def select_output_folder(self):

        path = filedialog.askdirectory(title="Select output folder")

        if path:
            self.output_path.set(path)


    # Clear all preview images and restore their placeholder text.
    # Input: None, uses the three preview labels
    # Output: None, removes the currently displayed preview images
    def clear_preview(self) -> None:
        """Clear all preview images.

        Returns:
            None.
        """
        self.preview_source = None
        self.preview_photo_original = None
        self.preview_photo_simulated = None
        self.preview_photo_corrected = None

        self.preview_original_label.config(image="", text="No image selected")
        self.preview_simulated_label.config(image="", text="No image selected")
        self.preview_corrected_label.config(image="", text="No image selected")
        

    # Update the correction strength and refresh the preview.
    # Input: value - str | float, slider value supplied by Tkinter
    # Output: None, updates the strength setting and preview
    def update_strength(self, value: str | float) -> None:
        """Update the correction strength and refresh the preview.

        Args:
            value: Current value supplied by the strength slider.

        Returns:
            None.
        """
        value = int(float(value))
        self.strength.set(value)
        self.strength_label.config(text=f"{value}%")
        self.update_preview()


    # Refresh the preview after the colour-vision deficiency changes.
    # Input: name - str, Tkinter variable name
    # Input: index - str, Tkinter trace index
    # Input: mode - str, Tkinter trace operation
    # Output: None, regenerates the simulated and corrected previews
    def update_cvd_preview(self, name: str, index: str, mode: str) -> None:
        """Refresh the preview when the selected deficiency changes.

        Args:
            name: Tkinter variable name supplied by the trace callback.
            index: Tkinter trace index supplied by the trace callback.
            mode: Tkinter trace operation supplied by the trace callback.

        Returns:
            None.
        """
        self.update_preview()


    # Find images
    def get_images(self, path: str | None = None) -> list[str]:

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

        if path is None:

            path = self.input_path.get()

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
        self.running = True

        if not input_path:

            messagebox.showerror("No input selected", "Please select an image or folder.")
            self.running = False
            return

        if not os.path.exists(input_path):

            messagebox.showerror("Invalid input", "The selected file or folder does not exist.")
            self.running = False
            return

        if not output_path:

            messagebox.showerror("No output folder", "Please select an output folder.")
            self.running = False
            return

        images = self.get_images(input_path)

        if not images:

            messagebox.showerror("No images", "No supported image files were found.")
            self.running = False
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
        self.progress.update_idletasks()
        self.status.set(f"Processing {completed} / {total}...")


    # Finished
    def processing_finished(self, total):

        self.progress_value.set(100)
        self.status.set(f"Finished — {total} image(s) processed.")
        self.process_button.config(state="normal")
        self.progress.update_idletasks()
        self.running = False
        messagebox.showinfo("Complete", f"{total} image(s) were successfully processed.")


    # Failed
    def processing_failed(self, error):

        self.process_button.config(state="normal")
        self.status.set("Processing failed.")
        self.progress.update_idletasks()
        self.running = False
        messagebox.showerror("Processing error", error)

    def on_close(self):
        
        self.running = False
        self.root.destroy()



# Create the application's main Tkinter window. 
# Input: test_mode - bool, whether deterministic test geometry should be used 
# Output: tk.Tk, configured main application window 
def create_window(test_mode: bool = False) -> tk.Tk: 
    """Create and configure the main application window. 

    Args: 
        test_mode: Whether to use deterministic geometry for GUI testing. 
        
    Returns: 
        The configured Tkinter root window. 
    """ 
    
    root = tk.Tk() 
    root.title("Daltonizer") 
    root.geometry("1200x900") 
    if test_mode: 
        root.geometry("1200x900+50+50")

    root.update_idletasks() 
    return root 
    
    
# Launch the Daltonizer application. 
# Input: test_mode - bool, whether deterministic test geometry should be used 
# Output: None, runs the Tkinter application until the window closes 
def main(test_mode: bool = False) -> None: 
    """Launch the Daltonizer application. 
    
    Returns: 
        None. 
    """ 
    root = create_window() 
    app = DaltonizerGUI(root) 
    root.mainloop() 
        
if __name__ == "__main__": 
    main("--test" in sys.argv)