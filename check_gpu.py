import sys
import jax
import jaxlib
import os

def check_jax_gpu():
    """
    Check if JAX is using GPU and print relevant configuration details.
    
    Returns:
        bool: True if GPU is available and being used, False otherwise
    """
    print("JAX Configuration:")
    print(f"- JAX version: {jax.__version__}")
    print(f"- JAXlib version: {jaxlib.__version__}")
    print(f"- Python version: {sys.version.split()[0]}")
    
    # Check for GPU devices
    devices = jax.devices()
    print(f"- Available devices: {[str(d) for d in devices]}")
    
    # Check the backend
    print(f"- Default backend: {jax.default_backend()}")
    
    # Check for CUDA-specific environment variables
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
    print(f"- CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    
    # Try to get GPU device count
    gpu_count = 0
    try:
        gpu_count = jax.device_count("gpu")
        print(f"- GPU device count: {gpu_count}")
    except:
        print("- GPU devices not accessible")
    
    # More detailed platform check using the new API
    try:
        # Use the newer API to avoid deprecation warning
        platform = jax.extend.backend.get_backend().platform
    except AttributeError:
        # Fall back to the old API if the new one isn't available
        platform = jax.lib.xla_bridge.get_backend().platform
    
    print(f"- XLA platform: {platform}")
    
    # Print a summary
    is_gpu = platform == "gpu" or gpu_count > 0 or jax.default_backend() == "gpu"
    
    if is_gpu:
        print("\n✅ JAX is using GPU!")
        
        # Try to get CUDA version info
        try:
            from jax.lib import cuda_versions
            print(f"- CUDA version: {cuda_versions.cuda_version}")
            print(f"- CuDNN version: {cuda_versions.cudnn_version}")
        except:
            print("- Could not detect CUDA version details")
    else:
        print("\n❌ JAX is using CPU, not GPU!")
        print("  This could be because:")
        print("  1. GPU version of JAX is not installed")
        print("  2. No GPU is available on this system")
        print("  3. CUDA environment is not properly configured")
    
    return is_gpu

if __name__ == "__main__":
    is_gpu_available = check_jax_gpu()
    
    # Create a small matrix multiplication benchmark to test performance
    print("\nRunning a small benchmark...")
    import time
    import numpy as np
    
    # Create two random matrices
    x = np.random.normal(size=(1000, 1000)).astype(np.float32)
    y = np.random.normal(size=(1000, 1000)).astype(np.float32)
    
    # Warm up
    jax.numpy.dot(x, y).block_until_ready()
    
    # Benchmark
    start_time = time.time()
    for _ in range(10):
        result = jax.numpy.dot(x, y).block_until_ready()
    end_time = time.time()
    
    print(f"Matrix multiplication (1000x1000) x 10: {end_time - start_time:.4f} seconds")
    
    # Exit with status code based on GPU availability
    sys.exit(0 if is_gpu_available else 1)
