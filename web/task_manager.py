"""
训练任务管理器
负责管理后台训练进程的生命周期：启动、停止、状态查询、日志/指标收集。
"""
import os
import sys
import time
import uuid
import threading
import traceback
from datetime import datetime
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field

# 把项目根目录加入路径，确保可以导入 trainer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from web.logger_adapter import LogCapture, MetricsCollector
from web.schemas import TaskStatus


@dataclass
class TaskRecord:
    task_id: str
    task_type: str  # "pretrain" | "sft" | "ppo"
    status: TaskStatus = TaskStatus.idle
    config: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error_message: Optional[str] = None
    log_capture: Optional[LogCapture] = None
    metrics: Optional[MetricsCollector] = None
    thread: Optional[threading.Thread] = None
    stop_event: Optional[threading.Event] = None


class TaskManager:
    """
    单例任务管理器。
    同一时间只允许一个训练任务运行。
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self._current_task: Optional[TaskRecord] = None
        self._lock = threading.Lock()
        self._history: list = []

    # ---------- 公共查询接口 ----------

    def get_current_task(self) -> Optional[TaskRecord]:
        with self._lock:
            return self._current_task

    def get_history(self) -> list:
        with self._lock:
            return list(self._history)

    def is_busy(self) -> bool:
        with self._lock:
            return self._current_task is not None and self._current_task.status == TaskStatus.running

    def get_logs(self, last_n: Optional[int] = None) -> list:
        task = self.get_current_task()
        if task and task.log_capture:
            return task.log_capture.get_logs(last_n)
        return []

    def get_metrics(self) -> dict:
        task = self.get_current_task()
        if task and task.metrics:
            return task.metrics.get_series()
        return {}

    # ---------- 启动训练 ----------

    def start_pretrain(self, config: dict,
                       log_callback: Optional[Callable] = None,
                       metric_callback: Optional[Callable] = None) -> str:
        """启动预训练任务，返回 task_id。"""
        from trainer.train_pretrain_entry import run_pretrain
        return self._start_task("pretrain", config, run_pretrain, log_callback, metric_callback)

    def start_sft(self, config: dict,
                  log_callback: Optional[Callable] = None,
                  metric_callback: Optional[Callable] = None) -> str:
        """启动 SFT (LoRA) 微调任务，返回 task_id。"""
        from trainer.train_sft_entry import run_sft
        return self._start_task("sft", config, run_sft, log_callback, metric_callback)

    def _start_task(self, task_type: str, config: dict,
                    train_func: Callable,
                    log_callback: Optional[Callable] = None,
                    metric_callback: Optional[Callable] = None) -> str:
        if self.is_busy():
            raise RuntimeError("已有任务正在运行，请先停止当前任务")

        task_id = str(uuid.uuid4())[:8]
        stop_event = threading.Event()

        log_cap = LogCapture(callback=log_callback)
        metrics_col = MetricsCollector()
        if metric_callback:
            metrics_col.set_callback(metric_callback)

        task = TaskRecord(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.running,
            config=config,
            start_time=datetime.now().isoformat(),
            log_capture=log_cap,
            metrics=metrics_col,
            stop_event=stop_event
        )

        def _run():
            log_cap.start()
            try:
                # 把 stop_event 和 metrics 注入 config，让训练函数可以响应停止信号
                config["_stop_event"] = stop_event
                config["_metrics_collector"] = metrics_col
                train_func(config)
                task.status = TaskStatus.completed
            except Exception as e:
                task.status = TaskStatus.failed
                task.error_message = str(e)
                traceback.print_exc()
            finally:
                task.end_time = datetime.now().isoformat()
                log_cap.stop()
                with self._lock:
                    self._history.append(task)
                    # 如果当前任务还是这个，清理引用
                    if self._current_task is task:
                        self._current_task = None

        thread = threading.Thread(target=_run, daemon=True)
        task.thread = thread

        with self._lock:
            self._current_task = task

        thread.start()
        return task_id

    # ---------- 停止任务 ----------

    def stop_current_task(self) -> bool:
        """发送停止信号给当前训练任务。"""
        with self._lock:
            task = self._current_task
        if task is None:
            return False
        if task.status != TaskStatus.running:
            return False

        task.status = TaskStatus.stopped
        if task.stop_event:
            task.stop_event.set()
        return True

    # ---------- 清理 ----------

    def clear_history(self):
        with self._lock:
            self._history.clear()
