"""
EEG stream gateway adapter backed by StreamingThread.
"""

from typing import Callable, Optional

from brainbridge_v2.infrastructure.acquisition.streaming_thread import StreamingThread


class EEGStreamGatewayAdapter:
    """
    Infrastructure adapter that maps EEGStreamGateway to StreamingThread.
    """

    def __init__(self):
        self._thread: Optional[StreamingThread] = None
        self._data_callback: Optional[Callable[[object], None]] = None
        self._connection_callback: Optional[Callable[[bool], None]] = None

    def connect(self, host: str, port: int) -> bool:
        if self._thread is not None and self._thread.isRunning():
            return True

        self._thread = StreamingThread()
        if self._data_callback is not None:
            self._thread.data_received.connect(self._data_callback)
        if self._connection_callback is not None:
            self._thread.connection_status.connect(self._connection_callback)
        self._thread.start_streaming(host, port)
        return True

    def disconnect(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.stop_streaming()

    def is_running(self) -> bool:
        return bool(self._thread is not None and self._thread.isRunning())

    def is_mock_mode(self) -> bool:
        return bool(self._thread is not None and getattr(self._thread, "is_mock_mode", False))

    def set_data_callback(self, callback: Callable[[object], None]) -> None:
        self._data_callback = callback
        if self._thread is not None:
            self._thread.data_received.connect(callback)

    def set_connection_callback(self, callback: Callable[[bool], None]) -> None:
        self._connection_callback = callback
        if self._thread is not None:
            self._thread.connection_status.connect(callback)
