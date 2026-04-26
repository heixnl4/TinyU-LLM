"""
TinyU-LLM FastAPI 后端服务
提供训练管理、模型推理、文件操作的 HTTP 和 WebSocket 接口。
"""
import os
import sys
import json
import glob
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from web.schemas import (
    PretrainConfig, SFTConfig, ChatRequest, ChatResponse,
    LoadWeightRequest, ApiResponse, TaskStatus, TaskInfo
)
from web.task_manager import TaskManager
from web.chat_engine import ChatEngine

app = FastAPI(
    title="TinyU-LLM Web Backend",
    description="TinyU-LLM 模型训练与推理的 Web 后端 API",
    version="0.1.0",
)

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- 全局单例 ----------------
task_mgr = TaskManager()
chat_engine = ChatEngine()


def _task_to_dict(task) -> dict:
    """将 TaskRecord 转换为可序列化的字典。"""
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "status": task.status.value if isinstance(task.status, TaskStatus) else str(task.status),
        "config": task.config,
        "start_time": task.start_time,
        "end_time": task.end_time,
        "error_message": task.error_message,
    }


# ==================== 健康检查 ====================
@app.get("/api/status")
def api_status():
    return {
        "status": "ok",
        "time": datetime.now().isoformat(),
        "task_running": task_mgr.is_busy(),
        "model_loaded": chat_engine.is_model_loaded(),
    }


# ==================== 训练管理 ====================
@app.post("/api/train/pretrain")
def start_pretrain(cfg: PretrainConfig):
    if task_mgr.is_busy():
        raise HTTPException(status_code=409, detail="已有任务正在运行")
    config_dict = cfg.model_dump()
    task_id = task_mgr.start_pretrain(config_dict)
    return {"code": 0, "message": "预训练任务已启动", "data": {"task_id": task_id}}


@app.post("/api/train/sft")
def start_sft(cfg: SFTConfig):
    if task_mgr.is_busy():
        raise HTTPException(status_code=409, detail="已有任务正在运行")
    config_dict = cfg.model_dump()
    task_id = task_mgr.start_sft(config_dict)
    return {"code": 0, "message": "SFT 微调任务已启动", "data": {"task_id": task_id}}


@app.post("/api/train/stop")
def stop_train():
    ok = task_mgr.stop_current_task()
    if not ok:
        raise HTTPException(status_code=400, detail="当前没有正在运行的任务")
    return {"code": 0, "message": "停止信号已发送"}


@app.get("/api/train/status")
def train_status():
    task = task_mgr.get_current_task()
    if task is None:
        return {"code": 0, "data": None}
    return {"code": 0, "data": _task_to_dict(task)}


@app.get("/api/train/history")
def train_history():
    history = task_mgr.get_history()
    return {"code": 0, "data": [_task_to_dict(t) for t in history]}


@app.get("/api/train/logs")
def train_logs(last_n: Optional[int] = 200):
    logs = task_mgr.get_logs(last_n=last_n)
    return {"code": 0, "data": logs}


@app.get("/api/train/metrics")
def train_metrics():
    metrics = task_mgr.get_metrics()
    return {"code": 0, "data": metrics}


# ==================== WebSocket：训练日志实时推送 ====================
@app.websocket("/api/ws/train")
async def ws_train_logs(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # 每 1 秒推送一次最新日志和指标
            logs = task_mgr.get_logs(last_n=50)
            metrics = task_mgr.get_metrics()
            task = task_mgr.get_current_task()
            payload = {
                "type": "train_update",
                "logs": logs,
                "metrics": metrics,
                "task": _task_to_dict(task) if task else None,
            }
            await websocket.send_json(payload)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ==================== 模型管理 ====================
@app.post("/api/model/load")
def load_model(req: LoadWeightRequest):
    try:
        info = chat_engine.load_model(req)
        return {"code": 0, "message": "模型加载成功", "data": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/model/unload")
def unload_model():
    chat_engine.unload_model()
    return {"code": 0, "message": "模型已卸载"}


@app.get("/api/model/status")
def model_status():
    return {"code": 0, "data": chat_engine.get_status()}


# ==================== 对话接口 ====================
@app.post("/api/chat")
def chat(req: ChatRequest):
    if not chat_engine.is_model_loaded():
        raise HTTPException(status_code=400, detail="模型尚未加载")
    try:
        tokens = []
        for token in chat_engine.generate_stream(
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        ):
            tokens.append(token)
        return {"code": 0, "data": {"response": "".join(tokens)}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/api/ws/chat")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_json()
            prompt = msg.get("prompt", "")
            max_new_tokens = msg.get("max_new_tokens", 100)
            temperature = msg.get("temperature", 0.8)
            top_k = msg.get("top_k", 50)
            top_p = msg.get("top_p", 0.9)

            if not chat_engine.is_model_loaded():
                await websocket.send_json({"error": "模型尚未加载"})
                continue

            for token in chat_engine.generate_stream(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            ):
                await websocket.send_json({"token": token})
            await websocket.send_json({"done": True})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"error": str(e)})


# ==================== 文件管理 ====================
@app.get("/api/files/checkpoints")
def list_checkpoints():
    """递归扫描 checkpoints/ 和 out/ 目录，列出所有 .pth 文件。"""
    results = []
    for base_dir in ["./checkpoints", "./out"]:
        if not os.path.exists(base_dir):
            continue
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.endswith(".pth"):
                    full = os.path.join(root, f)
                    results.append({
                        "path": full,
                        "name": f,
                        "dir": root,
                        "size": os.path.getsize(full),
                        "mtime": datetime.fromtimestamp(os.path.getmtime(full)).isoformat(),
                    })
    # 按修改时间倒序
    results.sort(key=lambda x: x["mtime"], reverse=True)
    return {"code": 0, "data": results}


@app.get("/api/files/datasets")
def list_datasets():
    """列出 dataset/ 目录下的 .jsonl 文件。"""
    results = []
    base_dir = "./dataset"
    if os.path.exists(base_dir):
        for f in os.listdir(base_dir):
            if f.endswith(".jsonl"):
                full = os.path.join(base_dir, f)
                results.append({
                    "path": full,
                    "name": f,
                    "size": os.path.getsize(full),
                    "mtime": datetime.fromtimestamp(os.path.getmtime(full)).isoformat(),
                })
    results.sort(key=lambda x: x["mtime"], reverse=True)
    return {"code": 0, "data": results}


@app.post("/api/files/upload")
def upload_dataset(file: UploadFile = File(...)):
    """上传数据集文件到 dataset/ 目录。"""
    if not file.filename.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="仅支持 .jsonl 文件")
    save_dir = "./dataset"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file.filename)
    with open(save_path, "wb") as f:
        f.write(file.file.read())
    return {"code": 0, "message": f"文件已保存: {save_path}", "data": {"path": save_path}}


# ==================== 启动入口 ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
