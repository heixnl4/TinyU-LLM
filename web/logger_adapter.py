"""
日志适配器：捕获训练过程中的 print 输出和指标，
通过回调推送到前端或保存到内存队列中。
"""
import sys
import threading
import json
from datetime import datetime
from typing import Callable, Optional
from collections import deque


class LogCapture:
    """捕获 stdout 的输出，同时保留原始输出行为。"""

    def __init__(self, callback: Optional[Callable] = None, max_lines: int = 5000):
        self._callback = callback
        self._original_stdout = sys.stdout
        self._lock = threading.Lock()
        self._buffer = deque(maxlen=max_lines)
        self._active = False

    def start(self):
        if not self._active:
            self._active = True
            sys.stdout = self

    def stop(self):
        if self._active:
            self._active = False
            sys.stdout = self._original_stdout

    def write(self, text):
        # 保持原始输出
        self._original_stdout.write(text)
        self._original_stdout.flush()

        # 缓存并回调
        if text.strip():
            entry = {
                "time": datetime.now().isoformat(),
                "level": "INFO",
                "message": text.rstrip("\n")
            }
            with self._lock:
                self._buffer.append(entry)
            if self._callback:
                try:
                    self._callback(entry)
                except Exception:
                    pass

    def flush(self):
        self._original_stdout.flush()

    def get_logs(self, last_n: Optional[int] = None):
        with self._lock:
            logs = list(self._buffer)
        if last_n:
            logs = logs[-last_n:]
        return logs

    def clear(self):
        with self._lock:
            self._buffer.clear()


class MetricsCollector:
    """收集训练指标（loss、lr、step 等），用于前端图表展示。"""

    def __init__(self, max_points: int = 2000):
        self._metrics = {
            "loss": deque(maxlen=max_points),
            "aux_loss": deque(maxlen=max_points),
            "learning_rate": deque(maxlen=max_points),
            "step": deque(maxlen=max_points),
        }
        self._lock = threading.Lock()
        self._callback: Optional[Callable] = None

    def set_callback(self, callback: Callable):
        self._callback = callback

    def record(self, step: int, loss: Optional[float] = None,
               aux_loss: Optional[float] = None, lr: Optional[float] = None):
        point = {"step": step}
        if loss is not None:
            point["loss"] = loss
            self._metrics["loss"].append({"x": step, "y": loss})
        if aux_loss is not None:
            point["aux_loss"] = aux_loss
            self._metrics["aux_loss"].append({"x": step, "y": aux_loss})
        if lr is not None:
            point["lr"] = lr
            self._metrics["learning_rate"].append({"x": step, "y": lr})

        if self._callback:
            try:
                self._callback(point)
            except Exception:
                pass

    def get_series(self):
        with self._lock:
            return {
                k: list(v) for k, v in self._metrics.items()
            }

    def clear(self):
        with self._lock:
            for v in self._metrics.values():
                v.clear()
