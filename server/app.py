import os
import time
from flask import Flask, request, jsonify
import torch

from shared.serialization import bytes_to_tensor, tensor_to_bytes
from shared.split_resnet import ResNet18Splitter
from shared.split_mobilenetv2 import MobileNetV2Splitter
from pathlib import Path
from shared.split_mlp_covertype import MLPCovertypeSplitter

app = Flask(__name__)
MLP_CKPT = Path(__file__).resolve().parents[1] / "artifacts" / "mlp_covertype.pt"

DEVICE = torch.device("cpu")
splitter = ResNet18Splitter(device=DEVICE)
mobilenet_splitter = MobileNetV2Splitter(device=DEVICE)
mlp_splitter = MLPCovertypeSplitter(checkpoint_path=MLP_CKPT, device=DEVICE)


@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.post("/infer_full")
def infer_full():
    t0 = time.time()
    if "x" not in request.files:
        return jsonify({"error": "missing file field 'x'"}), 400

    x_bytes = request.files["x"].read()
    x = bytes_to_tensor(x_bytes)
    y = splitter.full(x)

    y_bytes = tensor_to_bytes(y)
    t1 = time.time()
    headers = {
        "X-Server-Compute-ms": f"{(t1 - t0)*1000.0:.3f}",
        "X-Response-Bytes": str(len(y_bytes)),
    }
    return (y_bytes, 200, headers)

@app.post("/infer_back")
def infer_back():
    t0 = time.time()
    split_id = request.form.get("split_id", None)
    if split_id is None:
        return jsonify({"error": "missing form field 'split_id'"}), 400
    if "act" not in request.files:
        return jsonify({"error": "missing file field 'act'"}), 400

    act_bytes = request.files["act"].read()
    act = bytes_to_tensor(act_bytes)
    y = splitter.back(act, split_id)

    y_bytes = tensor_to_bytes(y)
    t1 = time.time()
    headers = {
        "X-Server-Compute-ms": f"{(t1 - t0)*1000.0:.3f}",
        "X-Response-Bytes": str(len(y_bytes)),
    }
    return (y_bytes, 200, headers)

@app.post("/infer_full_mobilenetv2")
def infer_full_mobilenetv2():
    t0 = time.time()
    if "x" not in request.files:
        return jsonify({"error": "missing file field 'x'"}), 400
    x_bytes = request.files["x"].read()
    x = bytes_to_tensor(x_bytes)
    y = mobilenet_splitter.full(x)
    y_bytes = tensor_to_bytes(y)
    t1 = time.time()
    headers = {
        "X-Server-Compute-ms": f"{(t1 - t0)*1000.0:.3f}",
        "X-Response-Bytes": str(len(y_bytes)),
    }
    return (y_bytes, 200, headers)

@app.post("/infer_back_mobilenetv2")
def infer_back_mobilenetv2():
    t0 = time.time()
    split_id = request.form.get("split_id", None)
    if split_id is None:
        return jsonify({"error": "missing form field 'split_id'"}), 400
    if "act" not in request.files:
        return jsonify({"error": "missing file field 'act'"}), 400
    act_bytes = request.files["act"].read()
    act = bytes_to_tensor(act_bytes)
    y = mobilenet_splitter.back(act, split_id)
    y_bytes = tensor_to_bytes(y)
    t1 = time.time()
    headers = {
        "X-Server-Compute-ms": f"{(t1 - t0)*1000.0:.3f}",
        "X-Response-Bytes": str(len(y_bytes)),
    }
    return (y_bytes, 200, headers)

@app.post("/infer_full_mlp_covertype")
def infer_full_mlp_covertype():
    t0 = time.time()
    if "x" not in request.files:
        return jsonify({"error": "missing file field 'x'"}), 400
    x_bytes = request.files["x"].read()
    x = bytes_to_tensor(x_bytes)
    y = mlp_splitter.full(x)
    y_bytes = tensor_to_bytes(y)
    t1 = time.time()
    headers = {
        "X-Server-Compute-ms": f"{(t1 - t0)*1000.0:.3f}",
        "X-Response-Bytes": str(len(y_bytes)),
    }
    return (y_bytes, 200, headers)

@app.post("/infer_back_mlp_covertype")
def infer_back_mlp_covertype():
    t0 = time.time()
    split_id = request.form.get("split_id", None)
    if split_id is None:
        return jsonify({"error": "missing form field 'split_id'"}), 400
    if "act" not in request.files:
        return jsonify({"error": "missing file field 'act'"}), 400
    act_bytes = request.files["act"].read()
    act = bytes_to_tensor(act_bytes)
    y = mlp_splitter.back(act, split_id)
    y_bytes = tensor_to_bytes(y)
    t1 = time.time()
    headers = {
        "X-Server-Compute-ms": f"{(t1 - t0)*1000.0:.3f}",
        "X-Response-Bytes": str(len(y_bytes)),
    }
    return (y_bytes, 200, headers)



if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port, threaded=True)
