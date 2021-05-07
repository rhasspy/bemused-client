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

DEFAULT_NUM_HITS_TO_DETECT = 9


@dataclass
class KeywordConfig:
    """Configuration for detector"""

    # Path to TFLite model
    model_path: Path

    # Integer labels that result in a detection
    keyword_labels: typing.Set[int]

    # Audio settings for model
    sample_rate: int = 16000
    sample_bytes: int = 2
    num_channels: int = 1

    # Size of audio chunks to read at a time
    block_ms: int = 20  # block_samples = sample_rate * (block_ms / 1000)

    # Size of audio chunk to process at a time
    process_ms: int = 20  # process_samples = sample_rate * (process_ms / 1000)

    # If True, model state is kept and passed back in
    streaming: bool = False

    # Seconds before a second detection can occur
    refractory_seconds: float = 1.0

    # Number of times in a row that a label must be the highest probability to
    # trigger a detection.
    num_hits_to_detect: int = DEFAULT_NUM_HITS_TO_DETECT

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
        self.block_samples = int(
            self.config.sample_rate * (self.config.block_ms / 1000.0)
        )
        self.process_samples = int(
            self.config.sample_rate * (self.config.process_ms / 1000.0)
        )

        # Detection thread
        self.detect_thread: typing.Optional[threading.Thread] = None
        self.running = False

        # TFLite interpreter
        self.interpreter: typing.Optional[tf.lite.Interpreter] = None

        # Queue of (audio, tag) pairs or None (signals stop)
        self.chunk_queue: "Queue[typing.Optional[typing.Tuple[np.ndarray, typing.Any]]]" = Queue()

        self.chunk_buffer: typing.Optional[np.ndarray] = None

        self.last_label: int = -1
        self.chunk_processed_event = threading.Event()
        self.detected: bool = False

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
        wait: bool = False,
    ):
        """
        Send a single audio chunk to the detection thread (non-blocking).
        Detections will be placed in asynchronously self.detect_queue.

        An optional user-defined tag may be provided, and will be placed in the
        KeywordDetection object.
        """
        if not self.running:
            return

        self.detected = False

        # Convert to numpy array if necessary
        if isinstance(chunk, bytes):
            chunk_array = np.frombuffer(chunk, dtype=np.int16)
        else:
            chunk_array = chunk

        chunk_array = chunk_array.astype(np.float32)
        chunk_array = np.reshape(chunk_array, (1, len(chunk_array)))

        if self.chunk_buffer is None:
            self.chunk_buffer = chunk_array
        else:
            self.chunk_buffer = np.concatenate((self.chunk_buffer, chunk_array), axis=1)

        if self.chunk_buffer.shape[1] >= self.process_samples:
            self.chunk_processed_event.clear()

            self.chunk_queue.put_nowait(
                (self.chunk_buffer[:, : self.process_samples], tag)
            )
            self.chunk_buffer = self.chunk_buffer[:, self.block_samples :]

            if wait:
                self.chunk_processed_event.wait()

        return self.last_label

    def _detect_proc(self):
        # Get input and output tensors.
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        inputs = [
            np.zeros(input_details[state_idx]["shape"], dtype=np.float32)
            for state_idx in range(len(input_details))
        ]

        num_labels = output_details[0]["shape"][1]
        label_counts = np.zeros(num_labels)

        # Indexes of non-keyword labels
        kw_labels = list(self.config.keyword_labels)

        # Samples for refractory period
        refractory_samples_left = 0
        refractory_num_samples = (
            self.config.refractory_seconds * self.config.sample_rate
        )

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
            if self.config.streaming:
                for state_idx in range(1, len(input_details)):
                    self.interpreter.set_tensor(
                        input_details[state_idx]["index"], inputs[state_idx]
                    )

            # Run detection
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(output_details[0]["index"])

            if self.config.streaming:
                # Get output states and set it back to input states, which will be
                # fed into the next inference cycle.
                for state_idx in range(1, len(input_details)):
                    # The function `get_tensor()` returns a copy of the tensor data.
                    # Use `tensor()` in order to get a pointer to the tensor.
                    inputs[state_idx] = self.interpreter.get_tensor(
                        output_details[state_idx]["index"]
                    )

            # Determine best label
            max_label = int(np.argmax(output_data[0]))

            detected = False
            if max_label in self.config.keyword_labels:
                label_counts[max_label] += 1
                if label_counts[max_label] >= self.config.num_hits_to_detect:
                    detected = True
            else:
                label_counts[kw_labels] -= 1
                label_counts = np.clip(label_counts, 0, None)

            if self.config.log_detector:
                # Log fine-grained details
                loop_time = time.perf_counter() - loop_start_time
                _LOGGER.debug(
                    "Loop: time=%s, max=%s, counts=%s, output=%s",
                    loop_time,
                    max_label,
                    label_counts,
                    output_data[0],
                )

            self.last_label = max_label

            if detected:
                # Detection occurred
                detect_time = time.perf_counter()

                # Check refractory period
                if refractory_samples_left <= 0:
                    # Post to detection queue
                    self.detected = True
                    self.detect_queue.put_nowait(
                        KeywordDetection(
                            label=max_label, detect_time=detect_time, tag=tag
                        )
                    )

                    refractory_samples_left = refractory_num_samples
                    _LOGGER.debug(
                        "Keyword detected at %s, counts=%s", detect_time, label_counts
                    )
                else:
                    _LOGGER.debug(
                        "Detection occurred during refractory period (%s/%s sample(s) left)",
                        refractory_samples_left,
                        refractory_num_samples,
                    )

                # Reset all label counts
                label_counts[:] = 0

                # Reset streaming state
                if self.config.streaming:
                    inputs = [
                        np.zeros(input_details[state_idx]["shape"], dtype=np.float32)
                        for state_idx in range(len(input_details))
                    ]

            if refractory_samples_left > 0:
                refractory_samples_left -= chunk.shape[1]

            self.chunk_processed_event.set()


# -----------------------------------------------------------------------------


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
