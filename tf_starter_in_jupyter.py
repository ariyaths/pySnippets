import os
import sys
from datetime import datetime
import pytz; ist_timezone = pytz.timezone('Asia/Kolkata')

def install_tf_cuda():
    print('Installing TensorFlow 2.12.0 with CUDA libraries...')
    # This installs TF 2.12 AND the matching NVIDIA libraries (cuDNN, etc.)
    # Note: The syntax is [and-cuda], not [with-CUDA]
    !{sys.executable} -m pip install "tensorflow[and-cuda]==2.12.0"
    print("Installation complete.")

def restart_runtime():
    """Forces the kernel to restart."""
    os.kill(os.getpid(), 9)

try:
    import tensorflow as tf
    
    # 1. Check Version
    if tf.__version__ != '2.12.0':
        print(f"Found TensorFlow {tf.__version__}. Installing 2.12.0 [and-cuda]...")
        install_tf_cuda()
        print("Restarting runtime to load new binaries...")
        restart_runtime()
        
    # 2. Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print(f"WARNING: TensorFlow {tf.__version__} is active, but NO GPU detected.")
        print("Possible fixes:")
        print("1. Ensure you are running on Linux or WSL2 (native Windows not supported for this).")
        print("2. If in Colab, go to Runtime > Change runtime type > T4 GPU.")
    else:
        print(f"SUCCESS: TensorFlow {tf.__version__} is active with {len(gpus)} GPU(s).")
        # Optional: Print specific GPU name to confirm
        !nvidia-smi -L

except ImportError:
    print("TensorFlow not found. Installing now...")
    install_tf_cuda()
    restart_runtime()

tf.config.optimizer.set_jit(True)
print("DONE at : ", datetime.now(ist_timezone).strftime("%A, %B %d, %Y %I:%M %p"))
