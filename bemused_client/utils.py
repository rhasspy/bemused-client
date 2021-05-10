import tensorflow as tf
from tensorflow.lite.experimental.microfrontend.python.ops import (
    audio_microfrontend_op as frontend_op,
)


def to_micro_spectrogram(
    audio,
    sample_rate: int = 16000,
    window_size_ms: int = 30,
    window_step_ms: int = 20,
    feature_bin_count: int = 40,
):
    int16_input = tf.cast(tf.multiply(audio, 32768), tf.int16)
    # https://git.io/Jkuux
    micro_frontend = frontend_op.audio_microfrontend(
        int16_input,
        sample_rate=sample_rate,
        window_size=window_size_ms,
        window_step=window_step_ms,
        num_channels=feature_bin_count,
        out_scale=1,
        out_type=tf.float32,
    )
    output = tf.multiply(micro_frontend, (10.0 / 256.0))
    return output
