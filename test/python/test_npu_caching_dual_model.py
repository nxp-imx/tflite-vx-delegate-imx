"""
Tests NPU delegate caching with dual-output TFLite models.

This test loads two TFlite models with 2 outputs, once with normal output tensor IDs
and once with swapped output tensor IDs. It runs inference using the NPU delegate with caching enabled,
verifies that the cache file is created and reused, and checks that the outputs are correct and identical
between the first (cache creation) and second (cache loading) runs.
"""

import pytest
import numpy as np
import os

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


def load_model(swap_outputs):
    """
    Load a pre-built TFLite model from test_data directory.

    Args:
        swap_outputs: Whether to load swapped or normal version

    Returns:
        bytes: TFLite model content
    """
    swap_suffix = "_swapped" if swap_outputs else "_normal"
    filename = f"dual_output_2x8{swap_suffix}.tflite"

    filepath = os.path.join(os.path.dirname(__file__), "test_data", filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found: {filepath}\n")

    with open(filepath, "rb") as f:
        return f.read()


@pytest.mark.parametrize(
    "swap_outputs",
    [False, True],
)
def test_npu_dual_output_with_cache(delegate_lib, swap_outputs):
    """
    Test NPU execution with caching enabled using pre-built TensorFlow models.

    This test demonstrates:
    1. First run: Model compilation with cache creation
    2. Second run: Loading from cache (faster execution)
    3. Verifying cache file is created and reused
    4. Testing both normal and swapped output tensor ID configurations
    """
    # Skip if delegate not provided
    if delegate_lib == "none" or not delegate_lib:
        pytest.skip(
            "VX delegate library path not provided. Use --external_delegate option."
        )

    # Load the pre-built model
    shape = [2, 8]  # Input shape used in the example model
    model_bytes = load_model(swap_outputs)

    # make sure cache file does not exist before testing
    cache_path = "test_cache.nb"
    if os.path.exists(cache_path):
        os.unlink(cache_path)

    # Create input data (same for both runs to verify consistency)
    input_data = np.random.uniform(-10, 10, shape).astype(np.float32)

    try:
        # --- First run: Create cache ---
        print(f"\n=== First run (creating cache): {cache_path} ===")

        # Load delegate with caching enabled
        delegate_options = {"allowed_cache_mode": "1", "cache_file_path": cache_path}
        delegate_first = tflite.load_delegate(delegate_lib, options=delegate_options)

        # Create interpreter with delegate
        interpreter_first = tflite.Interpreter(
            model_content=model_bytes, experimental_delegates=[delegate_first]
        )
        interpreter_first.allocate_tensors()

        # Get input and output details
        input_details = interpreter_first.get_input_details()
        output_details = interpreter_first.get_output_details()

        # Set input and run inference
        interpreter_first.set_tensor(input_details[0]["index"], input_data)
        interpreter_first.invoke()

        # When swap_outputs=True, tensor IDs are swapped:
        # output1 (tensor 3) = ABS, output2 (tensor 1) = MEAN
        # When swap_outputs=False (normal):
        # output1 (tensor 1) = ABS, output2 (tensor 3) = MEAN
        if swap_outputs:
            assert output_details[0]["index"] == 3
            assert output_details[1]["index"] == 1
        else:
            assert output_details[0]["index"] == 1
            assert output_details[1]["index"] == 3

        # Get outputs from first run
        output1_first = interpreter_first.get_tensor(output_details[0]["index"])
        output2_first = interpreter_first.get_tensor(output_details[1]["index"])

        # Verify cache was created
        assert os.path.exists(cache_path), "Cache file was not created"
        cache_size = os.path.getsize(cache_path)
        assert cache_size > 0, "Cache file is empty"
        print(f"  Cache created: {cache_size} bytes")

        # Clean up first interpreter
        del interpreter_first
        del delegate_first

        # --- Second run: Load from cache ---
        print(f"=== Second run (loading from cache) ===")

        # Load delegate again with same cache path
        delegate_second = tflite.load_delegate(delegate_lib, options=delegate_options)

        # Create new interpreter
        interpreter_second = tflite.Interpreter(
            model_content=model_bytes, experimental_delegates=[delegate_second]
        )
        interpreter_second.allocate_tensors()

        # Run inference with same input
        interpreter_second.set_tensor(input_details[0]["index"], input_data)
        interpreter_second.invoke()

        # Get outputs from second run
        output1_second = interpreter_second.get_tensor(output_details[0]["index"])
        output2_second = interpreter_second.get_tensor(output_details[1]["index"])

        print(f"  Cache loaded successfully")

        # --- Verify both runs produce identical results ---
        np.testing.assert_array_equal(
            output1_first,
            output1_second,
            err_msg="Output1 differs between cached and non-cached runs",
        )
        np.testing.assert_array_equal(
            output2_first,
            output2_second,
            err_msg="Output2 differs between cached and non-cached runs",
        )

        # Verify correctness
        expected_abs = np.abs(input_data)
        expected_mean = np.mean(expected_abs)

        # the order of outputs in the output_details list is always [ABS, MEAN]
        np.testing.assert_allclose(
            output1_second,
            expected_abs,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Output1 doesn't match expected ABS(input)",
        )
        np.testing.assert_allclose(
            output2_second,
            expected_mean,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Output2 doesn't match expected MEAN(ABS(input))",
        )

        print(f"✓ Cache test passed for shape {shape}, swap_outputs={swap_outputs}")
        print(f"  Both runs produced identical outputs")
        print(
            f"  Output1 shape: {output1_second.shape}, Output2 shape: {output2_second.shape}"
        )

        # Clean up second interpreter
        del interpreter_second
        del delegate_second

    finally:
        # Clean up cache file
        if os.path.exists(cache_path):
            os.unlink(cache_path)
            print(f"  Cache file cleaned up")


if __name__ == "__main__":
    # For manual testing without pytest
    import sys

    if len(sys.argv) > 1:
        delegate_lib = sys.argv[1]
        print(f"Testing with delegate: {delegate_lib}")
        test_npu_dual_output_with_cache(delegate_lib, False)
        test_npu_dual_output_with_cache(delegate_lib, True)
        print("\n✓ All tests passed!")
    else:
        print(
            "Usage: python test_npu_caching_dual_model.py <path_to_libvx_delegate.so>"
        )
