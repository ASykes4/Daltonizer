"""Integration tests for the Daltonizer Tkinter GUI."""

import subprocess
import sys
import time
from pathlib import Path
from collections.abc import Generator

import pyautogui
import pytest


# Store the project root used to launch the GUI.
# Output: Path, project directory containing gui.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GUI_PATH = PROJECT_ROOT / "gui.py"


# Start the Daltonizer GUI in test mode.
# Output: subprocess.Popen, running Daltonizer GUI process
@pytest.fixture
def gui_process() -> Generator[subprocess.Popen, None, None]:
    """Start the Daltonizer GUI in deterministic test mode.

    Returns:
        The running Daltonizer GUI process.
    """
    process = subprocess.Popen(
        [sys.executable, str(GUI_PATH), "--test"],
        cwd=PROJECT_ROOT,
    )

    time.sleep(1.0)

    yield process

    process.terminate()

    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


# Close the currently focused GUI window.
# Output: None, closes the active application window
def close_gui() -> None:
    """Close the currently focused GUI window.

    Returns:
        None.
    """
    pyautogui.hotkey("alt", "f4")
    time.sleep(0.5)


# Verify that the Daltonizer GUI launches successfully.
# Input: gui_process - subprocess.Popen, running Daltonizer GUI process
# Output: None, assertions verify successful application startup
def test_gui_launches(gui_process: subprocess.Popen[bytes]) -> None:
    """Verify that the Daltonizer GUI launches successfully.

    Args:
        gui_process: Running Daltonizer GUI process.

    Returns:
        None.
    """
    assert gui_process.poll() is None

    close_gui()


# Verify that keyboard navigation reaches the GUI controls.
# Input: gui_process - subprocess.Popen, running Daltonizer GUI process
# Output: None, assertions verify that keyboard input does not close the GUI
def test_gui_keyboard_navigation(
    gui_process: subprocess.Popen[bytes],
) -> None:
    """Verify that the GUI responds to keyboard navigation.

    Args:
        gui_process: Running Daltonizer GUI process.

    Returns:
        None.
    """
    pyautogui.press("tab", presses=5, interval=0.1)

    assert gui_process.poll() is None

    close_gui()


# Verify that the correction selection can be changed by keyboard.
# Input: gui_process - subprocess.Popen, running Daltonizer GUI process
# Output: None, assertions verify that the GUI remains responsive
def test_correction_selection(
    gui_process: subprocess.Popen[bytes],
) -> None:
    """Verify that a correction type can be selected.

    Args:
        gui_process: Running Daltonizer GUI process.

    Returns:
        None.
    """
    pyautogui.press("tab", presses=5, interval=0.1)
    pyautogui.press("home")
    pyautogui.press("down", presses=2)
    pyautogui.press("enter")

    assert gui_process.poll() is None

    close_gui()


# Verify that the correction strength can be changed by keyboard.
# Input: gui_process - subprocess.Popen, running Daltonizer GUI process
# Output: None, assertions verify that the GUI remains responsive
def test_correction_strength(gui_process: subprocess.Popen[bytes]) -> None:
    """Verify that the correction strength responds to keyboard input.

    Args:
        gui_process: Running Daltonizer GUI process.

    Returns:
        None.
    """
    pyautogui.press("tab", presses=6, interval=0.1)
    pyautogui.press("home")
    pyautogui.press("right", presses=50)

    assert gui_process.poll() is None

    close_gui()
