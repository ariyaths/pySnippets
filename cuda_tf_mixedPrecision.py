import os
import tensorflow as tf
from keras import mixed_precision

# 1. Clean up logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 2. Prevent future deadlocks (The "Seatbelt")

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"TensorFlow {tf.__version__} detected the following GPUs: {gpus}")
    try:
        # Allow memory growth for all GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU : {tf.test.gpu_device_name()} - memory growth set to TRUE")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print("GPU : ", gpu, f" - encountered error setting memory growth: {e}")
else:
    print("No GPU found. TensorFlow will run on CPU. Memory growth setting is not applicable.")
    print("To enable GPU, ensure compatible GPU, drivers, and correctly installed CUDA/cuDNN.")


# 1. Set the Global Policy to Mixed Precision
# (Use 'mixed_bfloat16' if on RTX 5070 Ti, otherwise 'mixed_float16')
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# 2. Verify it's working
print('Compute dtype:', policy.compute_dtype)  # Should be 'float16'
print('Variable dtype:', policy.variable_dtype) # Should be 'float32'

# TensorFlow 2.20.0 detected the following GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
# GPU : /device:GPU:0 - memory growth set to TRUE
# Compute dtype: float16
# Variable dtype: float32
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1765430172.031872   40121 gpu_device.cc:2020] Created device /device:GPU:0 with 2128 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3050 Ti Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.6
