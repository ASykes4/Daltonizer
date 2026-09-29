"""Unit and integration tests for the Daltonizer Tkinter GUI backend."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from gui import DaltonizerGUI, create_window


# Create a Tkinter root window for GUI unit tests.
# Output: tk.Tk, hidden root window used by GUI tests
@pytest.fixture
def root():
    """Create a Tkinter root window for GUI testing.

    Yields:
        A Tkinter root window.

    Raises:
        pytest.skip: If Tkinter cannot create a graphical root window.
    """
    try:
        window = create_window(test_mode=True)
    except Exception as error:
        pytest.skip(f"Tkinter GUI unavailable: {error}")

    yield window
    window.destroy()


# Create a GUI instance for unit tests.
# Input: root - tk.Tk, Tkinter root window used by the GUI
# Output: DaltonizerGUI, initialized application instance
@pytest.fixture
def app(root) -> DaltonizerGUI:
    """Create an initialized Daltonizer GUI.

    Args:
        root: Tkinter root window used by the GUI.

    Returns:
        An initialized DaltonizerGUI instance.
    """
    return DaltonizerGUI(root)


# Create a small RGBA test image.
# Input: path - Path, destination path for the test image
# Output: None, creates a small RGBA PNG image
def create_test_image(path: Path) -> None:
    """Create a small RGBA PNG image for integration testing.

    Args:
        path: Destination path for the test image.

    Returns:
        None.
    """
    pixels = np.array(
        [
            [[255, 0, 0, 255], [0, 255, 0, 255]],
            [[0, 0, 255, 128], [255, 255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    alphaSupport = ("png","tif","tiff","webp")
    if str(path)[:-3] in alphaSupport:
        Image.fromarray(pixels).save(path)
    else:
        Image.fromarray(pixels).save(path)


# Verify that create_window creates the expected application window.
# Output: None, assertions verify the configured Tkinter root
def test_create_window() -> None:
    """Verify that create_window creates a configured Tkinter window.

    Returns:
        None.
    """
    try:
        root = create_window()
    except Exception as error:
        pytest.skip(f"Tkinter GUI unavailable: {error}")

    try:
        assert root.title() == "Daltonizer"
        assert root.geometry().startswith("800x600")
    finally:
        root.destroy()


# Verify that test mode uses deterministic window geometry.
# Output: None, assertions verify the test-mode window position
def test_create_window_test_mode() -> None:
    """Verify that test mode configures deterministic window geometry.

    Returns:
        None.
    """
    try:
        root = create_window(test_mode=True)
    except Exception as error:
        pytest.skip(f"Tkinter GUI unavailable: {error}")

    try:
        geometry = root.geometry()

        assert geometry.startswith("800x600")
        assert "+100+100" in geometry
    finally:
        root.destroy()


# Verify that DaltonizerGUI initializes its widgets and state.
# Input: app - DaltonizerGUI, initialized GUI instance
# Output: None, assertions verify initial application state
def test_gui_initialization(app: DaltonizerGUI) -> None:
    """Verify that the GUI initializes its application state.

    Args:
        app: Initialized GUI instance.

    Returns:
        None.
    """
    assert app.input_path.get() == ""
    assert app.output_path.get() == ""
    assert app.running is False


# Verify that correction strength updates the stored value.
# Input: app - DaltonizerGUI, initialized GUI instance
# Output: None, assertions verify the updated correction strength
def test_update_strength(app: DaltonizerGUI) -> None:
    """Verify that correction strength updates correctly.

    Args:
        app: Initialized GUI instance.

    Returns:
        None.
    """
    app.strength.set(75)
    assert app.strength.get() == 75

    app.update_strength(50)
    assert app.strength_label.cget("text") == "50%"


# Verify that selecting a file stores the selected input path.
# Input: app - DaltonizerGUI, initialized GUI instance
# Output: None, assertions verify the selected file path
@patch("gui.filedialog.askopenfilename")
def test_select_file(mock_dialog: MagicMock, app: DaltonizerGUI) -> None:
    """Verify that file selection stores the selected path.

    Args:
        mock_dialog: Mocked file-selection dialog.
        app: Initialized GUI instance.

    Returns:
        None.
    """
    mock_dialog.return_value = "/test/input.png"

    app.select_file()

    assert app.input_path.get() == "/test/input.png"
    mock_dialog.assert_called_once()


# Verify that cancelling file selection leaves the input unchanged.
# Input: app - DaltonizerGUI, initialized GUI instance
# Output: None, assertions verify that no path was selected
@patch("gui.filedialog.askopenfilename")
def test_select_file_cancel(mock_dialog: MagicMock, app: DaltonizerGUI) -> None:
    """Verify that cancelling file selection changes nothing.

    Args:
        mock_dialog: Mocked file-selection dialog.
        app: Initialized GUI instance.

    Returns:
        None.
    """
    app.input_path.set("/existing/input.png")
    mock_dialog.return_value = ""

    app.select_file()

    assert app.input_path.get() == "/existing/input.png"


# Verify that selecting a folder stores the selected input path.
# Input: app - DaltonizerGUI, initialized GUI instance
# Output: None, assertions verify the selected folder path
@patch("gui.filedialog.askdirectory")
def test_select_folder(mock_dialog: MagicMock, app: DaltonizerGUI) -> None:
    """Verify that folder selection stores the selected path.

    Args:
        mock_dialog: Mocked folder-selection dialog.
        app: Initialized GUI instance.

    Returns:
        None.
    """
    mock_dialog.return_value = "/test/input"

    app.select_folder()

    assert app.input_path.get() == "/test/input"
    mock_dialog.assert_called_once()


# Verify that selecting an output folder stores its path.
# Input: app - DaltonizerGUI, initialized GUI instance
# Output: None, assertions verify the selected output folder
@patch("gui.filedialog.askdirectory")
def test_select_output_folder(mock_dialog: MagicMock, app: DaltonizerGUI) -> None:
    """Verify that output folder selection stores the selected path.

    Args:
        mock_dialog: Mocked folder-selection dialog.
        app: Initialized GUI instance.

    Returns:
        None.
    """
    mock_dialog.return_value = "/test/output"

    app.select_output_folder()

    assert app.output_path.get() == "/test/output"
    mock_dialog.assert_called_once()


# Verify that get_images returns image files from a single file.
# Input: app - DaltonizerGUI, initialized GUI instance
# Input: tmp_path - Path, temporary directory used for test files
# Output: list[Path], discovered image paths
def test_get_images_file(app: DaltonizerGUI, tmp_path: Path) -> None:
    """Verify image discovery when the input is a single file.

    Args:
        app: Initialized GUI instance.
        tmp_path: Temporary directory used for test files.

    Returns:
        A list containing the selected image path.
    """
    image_path = tmp_path / "image.png"
    create_test_image(image_path)
    app.input_path.set(image_path)

    result = app.get_images(image_path)

    assert result == [image_path]


# Verify that get_images recursively finds supported image files.
# Input: app - DaltonizerGUI, initialized GUI instance
# Input: tmp_path - Path, temporary directory used for test files
# Output: list[Path], recursively discovered image paths
def test_get_images_folder(app: DaltonizerGUI, tmp_path: Path) -> None:
    """Verify recursive image discovery from an input folder.

    Args:
        app: Initialized GUI instance.
        tmp_path: Temporary directory used for test files.

    Returns:
        A list containing all discovered image paths.
    """
    nested_folder = tmp_path / "Animals" / "Dogs"
    nested_folder.mkdir(parents=True)

    first_image = tmp_path / "first.png"
    second_image = nested_folder / "second.jpg"
    create_test_image(first_image)
    create_test_image(second_image)

    (tmp_path / "document.txt").write_text("not an image")
    app.input_path.set(tmp_path)

    result = app.get_images(tmp_path)

    assert {Path(path) for path in result} == {first_image, second_image}


# Verify that process_images creates corrected output files.
# Input: app - DaltonizerGUI, initialized GUI instance
# Input: tmp_path - Path, temporary directory used for test files
# Output: None, assertions verify processed image output
def test_process_images_file_integration(app: DaltonizerGUI, tmp_path: Path) -> None:
    """Verify processing and saving a single image file.

    Args:
        app: Initialized GUI instance.
        tmp_path: Temporary directory used for test files.

    Returns:
        None.
    """
    input_path = tmp_path / "input.png"
    output_folder = tmp_path / "output"
    create_test_image(input_path)

    app.process_images([input_path], str(input_path), str(output_folder))

    output_path = output_folder / "input.png"

    assert output_path.exists()

    with Image.open(output_path) as output:
        assert output.size == (2, 2)
        assert output.mode == "RGBA"


# Verify that process_images preserves nested folder structure.
# Input: app - DaltonizerGUI, initialized GUI instance
# Input: tmp_path - Path, temporary directory used for test files
# Output: None, assertions verify the preserved directory structure
def test_process_images_folder_integration(app: DaltonizerGUI, tmp_path: Path) -> None:
    """Verify recursive processing preserves the folder structure.

    Args:
        app: Initialized GUI instance.
        tmp_path: Temporary directory used for test files.

    Returns:
        None.
    """
    input_folder = tmp_path / "Images"
    nested_folder = input_folder / "Animals" / "Dogs"
    nested_folder.mkdir(parents=True)

    first_image = input_folder / "landscape.png"
    second_image = nested_folder / "dog.png"
    create_test_image(first_image)
    create_test_image(second_image)

    output_folder = input_folder / "Daltonized"
    images = [first_image, second_image]

    app.process_images(images, str(input_folder), str(output_folder))

    first_output = output_folder / "landscape.png"
    second_output = output_folder / "Animals" / "Dogs" / "dog.png"

    assert first_output.exists()
    assert second_output.exists()

    with Image.open(first_output) as output:
        assert output.size == (2, 2)

    with Image.open(second_output) as output:
        assert output.size == (2, 2)


# Verify that start_processing refuses to run without an input path.
# Input: app - DaltonizerGUI, initialized GUI instance
# Output: None, assertions verify that processing does not start
@patch("gui.messagebox.showerror")
def test_start_processing_without_input(mock_error: MagicMock, app: DaltonizerGUI) -> None:
    """Verify processing requires an input path.

    Args:
        mock_warning: Mocked warning message box.
        app: Initialized GUI instance.

    Returns:
        None.
    """
    app.input_path.set("")
    app.output_path.set("/test/output")

    app.start_processing()

    mock_error.assert_called_once()
    assert app.running is False


# Verify that start_processing refuses to run without an output folder.
# Input: app - DaltonizerGUI, initialized GUI instance
# Output: None, assertions verify that processing does not start
@patch("gui.messagebox.showerror")
def test_start_processing_without_output(mock_error: MagicMock, app: DaltonizerGUI) -> None:
    """Verify processing requires an output folder.

    Args:
        mock_warning: Mocked warning message box.
        app: Initialized GUI instance.

    Returns:
        None.
    """
    app.input_path.set("/test/input.png")
    app.output_path.set("")

    app.start_processing()

    mock_error.assert_called_once()
    assert app.running is False


# Verify that update_progress updates the progress bar value.
# Input: app - DaltonizerGUI, initialized GUI instance
# Output: None, updates the progress bar
def test_update_progress(app: DaltonizerGUI) -> None:
    """Verify that progress updates are reflected in the progress bar.

    Args:
        app: Initialized GUI instance.

    Returns:
        None.
    """
    app.update_progress(50, 100)

    assert app.progress["value"] == 50


# Verify that processing_finished resets the processing state.
# Input: app - DaltonizerGUI, initialized GUI instance
# Output: None, assertions verify completed processing state
@patch("gui.messagebox.showinfo")
def test_processing_finished(mock_info: MagicMock, app: DaltonizerGUI) -> None:
    """Verify that successful processing resets the GUI state.

    Args:
        mock_info: Mocked information message box.
        app: Initialized GUI instance.

    Returns:
        None.
    """
    app.running = True

    app.processing_finished(42)

    assert app.running is False
    mock_info.assert_called_once()


# Verify that processing_failed resets the processing state.
# Input: app - DaltonizerGUI, initialized GUI instance
# Input: error - Exception, processing error to report
# Output: None, assertions verify failed processing state
@patch("gui.messagebox.showerror")
def test_processing_failed(mock_error: MagicMock, app: DaltonizerGUI) -> None:
    """Verify that failed processing resets the GUI state.

    Args:
        app: Initialized GUI instance.
        error: Processing error to report.

    Returns:
        None.
    """
    error = RuntimeError("Test processing error")
    app.running = True

    app.processing_failed(error)

    assert app.running is False
    mock_error.assert_called_once()
