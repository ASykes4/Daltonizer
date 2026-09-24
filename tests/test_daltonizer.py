"""Tests for the Daltonizer image-processing functions."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from daltonizer import calc_correct, daltonize_image, delinearize, get_transformation_matrices, linearize


# Create a small RGB image array for testing.
# Output: np.ndarray, RGB image data using uint8 values from 0 to 255
@pytest.fixture
def test_rgb_array() -> np.ndarray:
    """Create a small RGB image array for testing.

    Returns:
        A 3x2 RGB image represented as a NumPy uint8 array.
    """
    return np.array(
        [
            [[0, 0, 0], [255, 255, 255]],
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [128, 128, 128]],
        ],
        dtype=np.uint8,
    )


# Create a small RGBA PIL image for testing.
# Output: Image.Image, 3x2 RGBA test image
@pytest.fixture
def test_image() -> Image.Image:
    """Create a small RGBA PIL image for testing.

    Returns:
        A 3x2 RGBA PIL image.
    """
    pixels = np.array(
        [
            [[0, 0, 0, 255], [255, 255, 255, 255]],
            [[255, 0, 0, 128], [0, 255, 0, 64]],
            [[0, 0, 255, 0], [128, 128, 128, 200]],
        ],
        dtype=np.uint8,
    )

    return Image.fromarray(pixels, "RGBA")


# Verify that linearize produces valid linear RGB values.
# Input: test_rgb_array - np.ndarray, RGB image data to linearize
# Output: None, assertions verify the resulting array
def test_linearize_output_range(test_rgb_array: np.ndarray) -> None:
    """Verify that linearize produces values in the expected range.

    Args:
        test_rgb_array: RGB image data to linearize.

    Returns:
        None.
    """
    result = linearize(test_rgb_array)

    assert result.dtype == np.float32
    assert result.shape == test_rgb_array.shape
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


# Verify linearize against known sRGB conversion values.
# Output: None, assertions verify the known conversion results
def test_linearize_known_values() -> None:
    """Verify linearization against known sRGB values.

    Returns:
        None.
    """
    image = np.array([[[0, 255, 128]]], dtype=np.uint8)
    result = linearize(image)

    assert result[0, 0, 0] == pytest.approx(0.0)
    assert result[0, 0, 1] == pytest.approx(1.0)
    assert result[0, 0, 2] == pytest.approx(0.21586, abs=1e-4)


# Verify that delinearize produces valid uint8 RGB values.
# Output: None, assertions verify the resulting array
def test_delinearize_output_range() -> None:
    """Verify that delinearize produces values in the expected range.

    Returns:
        None.
    """
    image = np.array([[[0.0, 1.0, 0.5]]], dtype=np.float32)
    result = delinearize(image)

    assert result.dtype == np.uint8
    assert result.shape == image.shape
    assert np.all(result >= 0)
    assert np.all(result <= 255)


# Verify delinearize against known linear RGB conversion values.
# Output: None, assertions verify the known conversion results
def test_delinearize_known_values() -> None:
    """Verify delinearization against known linear RGB values.

    Returns:
        None.
    """
    image = np.array([[[0.0, 1.0, 0.5]]], dtype=np.float32)
    result = delinearize(image)

    assert result[0, 0, 0] == 0
    assert result[0, 0, 1] == 255
    assert result[0, 0, 2] == pytest.approx(188, abs=1)


# Verify correction strength interpolation for several input values.
# Input: number - float, correction value to scale
# Input: strength - int, correction strength from 0 to 100
# Input: expected - float, expected interpolated value
# Output: None, assertion verifies the interpolated value
@pytest.mark.parametrize(
    ("number", "strength", "expected"),
    [
        (1.0, 0, 0.0),
        (1.0, 50, 0.5),
        (1.0, 100, 1.0),
        (2.0, 25, 0.5),
        (-1.0, 50, -0.5),
    ],
)
def test_calc_correct(number: float, strength: int, expected: float) -> None:
    """Verify correction-strength interpolation.

    Args:
        number: Correction value to scale.
        strength: Correction strength from 0 to 100.
        expected: Expected interpolated value.

    Returns:
        None.
    """
    result = calc_correct(number, strength)

    assert result == pytest.approx(expected)


# Verify transformation matrices for each supported CVD type.
# Input: blind_type - str, colour-vision deficiency type to test
# Output: None, assertions verify the returned transformation matrices
@pytest.mark.parametrize(
    "blind_type",
    ["Protanopia", "Deuteranopia", "Tritanopia"],
)
def test_get_transformation_matrices(blind_type: str) -> None:
    """Verify transformation matrices for each supported CVD type.

    Args:
        blind_type: Colour-vision deficiency type to test.

    Returns:
        None.
    """
    matrices = get_transformation_matrices(blind_type, 100)

    assert len(matrices) == 4

    for matrix in matrices:
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (3, 3)
        assert np.all(np.isfinite(matrix))


# Verify that unsupported CVD types raise ValueError.
# Output: None, assertion verifies that ValueError is raised
def test_get_transformation_matrices_invalid_type() -> None:
    """Verify that an invalid CVD type raises ValueError.

    Returns:
        None.
    """
    with pytest.raises(ValueError):
        get_transformation_matrices("InvalidType", 100)


# Verify the basic output properties of daltonize_image.
# Input: test_image - Image.Image, RGBA image to correct
# Input: blind_type - str, colour-vision deficiency type to simulate
# Output: None, assertions verify the corrected image
@pytest.mark.parametrize(
    "blind_type",
    ["Protanopia", "Deuteranopia", "Tritanopia"],
)
def test_daltonize_image_output(test_image: Image.Image, blind_type: str) -> None:
    """Verify the basic output properties of daltonize_image.

    Args:
        test_image: RGBA image to correct.
        blind_type: Colour-vision deficiency type to simulate.

    Returns:
        None.
    """
    result = daltonize_image(test_image, blind_type, 100)

    assert isinstance(result, Image.Image)
    assert result.mode == "RGBA"
    assert result.size == test_image.size


# Verify that daltonize_image preserves the input alpha channel.
# Input: test_image - Image.Image, RGBA image whose alpha channel is tested
# Output: None, assertion verifies that alpha values are unchanged
def test_daltonize_image_preserves_alpha(test_image: Image.Image) -> None:
    """Verify that daltonization preserves the alpha channel.

    Args:
        test_image: RGBA image whose alpha channel is tested.

    Returns:
        None.
    """
    result = daltonize_image(test_image, "Deuteranopia", 100)

    original_alpha = np.asarray(test_image)[:, :, 3]
    result_alpha = np.asarray(result)[:, :, 3]

    np.testing.assert_array_equal(result_alpha, original_alpha)


# Verify that daltonize_image preserves the input dimensions.
# Input: test_image - Image.Image, RGBA image whose dimensions are tested
# Output: None, assertion verifies that dimensions are unchanged
def test_daltonize_image_preserves_dimensions(test_image: Image.Image) -> None:
    """Verify that daltonization preserves image dimensions.

    Args:
        test_image: RGBA image whose dimensions are tested.

    Returns:
        None.
    """
    result = daltonize_image(test_image, "Protanopia", 100)

    assert result.size == test_image.size


# Verify that zero correction strength leaves the image unchanged.
# Input: test_image - Image.Image, RGBA image to process
# Output: None, assertion verifies that RGB and alpha data are unchanged
def test_daltonize_image_strength_zero(test_image: Image.Image) -> None:
    """Verify that zero correction strength preserves the image.

    Args:
        test_image: RGBA image to process.

    Returns:
        None.
    """
    result = daltonize_image(test_image, "Deuteranopia", 0)

    original = np.asarray(test_image)
    corrected = np.asarray(result)

    np.testing.assert_array_equal(corrected, original)


# Verify that unsupported CVD types raise ValueError.
# Input: test_image - Image.Image, RGBA image to process
# Output: None, assertion verifies that ValueError is raised
def test_daltonize_image_invalid_type(test_image: Image.Image) -> None:
    """Verify that an invalid CVD type raises ValueError.

    Args:
        test_image: RGBA image to process.

    Returns:
        None.
    """
    with pytest.raises(ValueError):
        daltonize_image(test_image, "InvalidType", 100)


# Verify the complete daltonization pipeline for every supported CVD type.
# Input: test_image - Image.Image, RGBA image to process
# Input: blind_type - str, colour-vision deficiency type to simulate
# Output: None, assertions verify valid corrected image data
@pytest.mark.parametrize(
    "blind_type",
    ["Protanopia", "Deuteranopia", "Tritanopia"],
)
def test_daltonization_pipeline(test_image: Image.Image, blind_type: str) -> None:
    """Verify the complete image-processing pipeline.

    Args:
        test_image: RGBA image to process.
        blind_type: Colour-vision deficiency type to simulate.

    Returns:
        None.
    """
    result = daltonize_image(test_image, blind_type, 100)
    result_array = np.asarray(result)

    assert result_array.shape == (3, 2, 4)
    assert result_array.dtype == np.uint8
    assert np.all(np.isfinite(result_array))
    assert np.all(result_array[:, :, :3] >= 0)
    assert np.all(result_array[:, :, :3] <= 255)


# Verify that full-strength correction modifies RGB colour data.
# Input: test_image - Image.Image, RGBA image to process
# Input: blind_type - str, colour-vision deficiency type to simulate
# Output: None, assertion verifies that RGB data changes
@pytest.mark.parametrize(
    "blind_type",
    ["Protanopia", "Deuteranopia", "Tritanopia"],
)
def test_daltonization_changes_colour_data(test_image: Image.Image, blind_type: str) -> None:
    """Verify that full-strength daltonization changes RGB colour data.

    Args:
        test_image: RGBA image to process.
        blind_type: Colour-vision deficiency type to simulate.

    Returns:
        None.
    """
    result = daltonize_image(test_image, blind_type, 100)

    original_rgb = np.asarray(test_image)[:, :, :3]
    result_rgb = np.asarray(result)[:, :, :3]

    assert not np.array_equal(original_rgb, result_rgb)


# Verify loading, processing, saving, and reopening an image file.
# Input: test_image - Image.Image, RGBA image to process
# Input: tmp_path - Path, temporary directory for test files
# Output: None, assertions verify successful file processing
def test_image_file_integration(test_image: Image.Image, tmp_path: Path) -> None:
    """Verify the complete image file processing workflow.

    Args:
        test_image: RGBA image to process.
        tmp_path: Temporary directory used for test files.

    Returns:
        None.
    """
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"

    # Create the input image file.
    test_image.save(input_path)

    # Load and process the input image.
    with Image.open(input_path) as source:
        corrected = daltonize_image(source, "Deuteranopia", 100)

    # Save and reopen the corrected image.
    corrected.save(output_path)

    with Image.open(output_path) as output:
        assert output.size == test_image.size
        assert output.mode == "RGBA"
