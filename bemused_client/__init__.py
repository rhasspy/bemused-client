import logging
import threading
import time
import typing
from dataclasses import dataclass
from pathlib import Path
from queue import Queue

import numpy as np

_LOGGER = logging.getLogger("bemused_client")

# -----------------------------------------------------------------------------


@dataclass
class KeywordConfig:
    """Configuration for detector"""

    # Path to TFLite model
    model_path: Path

    # Integer labels that result in a detection
    keyword_labels: typing.Set[int]

    # Audio settings for model
    sample_rate: int = 16000
    num_channels: int = 1
    block_ms: int = 20  # block_size = sample_rate * (block_ms / 1000)

    # Seconds before a second detection can occur
    refractory_seconds: float = 1.0

    # If True, log details of every detector loop
    log_detector: bool = False


@dataclass
class KeywordDetection:
    """Details of single detection"""

    # Detected keyword label
    label: int

    # Time of detection according to time.perf_counter()
    detect_time: float

    # Optional user tag passed in from process_chunk.
    # Lets you determine which audio chunk produced the detection.
    # Needed because detections are asynchronous.
    tag: typing.Optional[typing.Any] = None


# -----------------------------------------------------------------------------


class KeywordDetector:
    def __init__(self, config: KeywordConfig):
        # Delay import
        import tensorflow as tf

        self.config = config

        # Queue to check for detections asynchronously.
        # No callbacks to avoid threading problems.
        self.detect_queue: "Queue[KeywordDetection]" = Queue()

        # Expected size of audio chunks
        self.block_size = int(self.config.sample_rate * (self.config.block_ms / 1000.0))

        # Detection thread
        self.detect_thread: typing.Optional[threading.Thread] = None
        self.running = False

        # TFLite interpreter
        self.interpreter: typing.Optional[tf.lite.Interpreter] = None

        # Queue of (audio, tag) pairs or None (signals stop)
        self.chunk_queue: "Queue[typing.Optional[typing.Tuple[np.ndarray, typing.Any]]]" = Queue()

    def start(self):
        """Load model and start detection thread"""
        # Delay import
        import tensorflow as tf

        if self.running:
            self.stop()

        _LOGGER.debug("Starting...")

        # Re-load model
        _LOGGER.debug("Loading model from %s", self.config.model_path)
        self.interpreter = tf.lite.Interpreter(model_path=str(self.config.model_path))
        self.interpreter.allocate_tensors()

        # Start detection thread with fresh queue
        self.chunk_queue = Queue()
        self.running = True
        self.detect_thread = threading.Thread(target=self._detect_proc, daemon=True)
        self.detect_thread.start()

        _LOGGER.debug("Started")

    def stop(self):
        """Stop detection thread and unload model"""
        _LOGGER.debug("Stopping...")
        self.running = False
        self.chunk_queue.put(None)

        if self.detect_thread:
            # Join with detection thread
            _LOGGER.debug("Waiting for detection thread to stop...")
            self.detect_thread.join()
            self.detect_thread = None

        self.interpreter = None
        _LOGGER.debug("Stopped")

    def process_chunk(
        self,
        chunk: typing.Union[bytes, np.ndarray],
        tag: typing.Optional[typing.Any] = None,
    ):
        """
        Send a single audio chunk to the detection thread (non-blocking).
        Detections will be placed in asynchronously self.detect_queue.

        An optional user-defined tag may be provided, and will be placed in the
        KeywordDetection object.
        """
        if not self.running:
            return

        # Convert to numpy array if necessary
        if isinstance(chunk, bytes):
            chunk_array = np.frombuffer(chunk, dtype=np.float32)
        else:
            chunk_array = chunk.astype(np.float32)

        chunk_array = np.reshape(chunk_array, (1, self.block_size))
        self.chunk_queue.put_nowait((chunk_array, tag))

    def _detect_proc(self):
        # Get input and output tensors.
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        inputs = [
            np.zeros(input_details[state_idx]["shape"], dtype=np.float32)
            for state_idx in range(len(input_details))
        ]

        # Previous label index
        last_argmax = -1

        out_max = 0

        # Reference time for refractory period
        refractory_start_time = 0

        # Detection loop
        while self.running:
            # Block until chunk comes in
            maybe_chunk_and_tag = self.chunk_queue.get()
            if not maybe_chunk_and_tag:
                # Empty value signals a stop
                break

            chunk, tag = maybe_chunk_and_tag
            loop_start_time = time.perf_counter()

            # Prepare model inputs
            self.interpreter.set_tensor(input_details[0]["index"], chunk)

            # Set input states (index 1...)
            for state_idx in range(1, len(input_details)):
                self.interpreter.set_tensor(
                    input_details[state_idx]["index"], inputs[state_idx]
                )

            # Run detection
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(output_details[0]["index"])

            if self.config.log_detector:
                # Log fine-grained details
                loop_time = time.perf_counter() - loop_start_time
                _LOGGER.debug(
                    "Loop: time=%s, last_argmax=%s, out_max=%s, output=%s",
                    loop_time,
                    last_argmax,
                    out_max,
                    output_data,
                )

            # Get output states and set it back to input states, which will be
            # fed into the next inference cycle.
            for state_idx in range(1, len(input_details)):
                # The function `get_tensor()` returns a copy of the tensor data.
                # Use `tensor()` in order to get a pointer to the tensor.
                inputs[state_idx] = self.interpreter.get_tensor(
                    output_details[state_idx]["index"]
                )

            # Determine the winner
            out_tflite_argmax = np.argmax(output_data)
            if last_argmax == out_tflite_argmax:
                if output_data[0][out_tflite_argmax] > out_max:
                    out_max = output_data[0][out_tflite_argmax]
            else:
                out_max = 0
                detect_label = int(out_tflite_argmax)

                if detect_label in self.config.keyword_labels:
                    # Detection occurred
                    detect_time = time.perf_counter()

                    # Check refractory period
                    if (
                        detect_time - refractory_start_time
                    ) > self.config.refractory_seconds:
                        # Post to detection queue
                        self.detect_queue.put_nowait(
                            KeywordDetection(
                                label=detect_label,
                                detect_time=detect_time,
                                tag=tag,
                            )
                        )

                        refractory_start_time = time.perf_counter()

            last_argmax = out_tflite_argmax
