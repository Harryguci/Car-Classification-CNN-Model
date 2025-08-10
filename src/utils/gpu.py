import tensorflow as tf


def enable_gpu_memory_growth() -> None:
    """Enable TensorFlow GPU memory growth to avoid OOM at startup."""
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if not gpus:
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled GPU memory growth")
    except RuntimeError as error:
        print(error)


