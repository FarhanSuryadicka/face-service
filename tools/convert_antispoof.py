"""
Konversi model anti-spoofing MiniFASNet (Silent-Face-Anti-Spoofing, Apache-2.0)
dari PyTorch (.pth) ke ONNX agar bisa dijalankan dengan OpenCV DNN (tanpa PyTorch).

Hanya dijalankan SEKALI di mesin developer. Image Docker tidak butuh PyTorch.

Pemakaian:
    git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git
    pip install torch onnx opencv-python-headless numpy
    python tools/convert_antispoof.py --repo path/ke/Silent-Face-Anti-Spoofing --out models
"""
import argparse
import os
import sys
from collections import OrderedDict

import numpy as np

# nama file .pth -> nama file .onnx yang dipakai service
MODELS = {
    "2.7_80x80_MiniFASNetV2.pth": "antispoof_minifasnet_v2_2.7_80x80.onnx",
    "4_0_0_80x80_MiniFASNetV1SE.pth": "antispoof_minifasnet_v1se_4.0_80x80.onnx",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Folder hasil clone Silent-Face-Anti-Spoofing")
    parser.add_argument("--out", default="models", help="Folder output .onnx")
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.repo))
    import torch
    from src.model_lib.MiniFASNet import MiniFASNetV1SE, MiniFASNetV2
    from src.utility import get_kernel, parse_model_name

    mapping = {"MiniFASNetV2": MiniFASNetV2, "MiniFASNetV1SE": MiniFASNetV1SE}
    os.makedirs(args.out, exist_ok=True)

    for pth_name, onnx_name in MODELS.items():
        pth_path = os.path.join(args.repo, "resources", "anti_spoof_models", pth_name)
        h, w, model_type, _ = parse_model_name(pth_name)
        model = mapping[model_type](conv6_kernel=get_kernel(h, w))

        state = torch.load(pth_path, map_location="cpu")
        state = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state.items())
        model.load_state_dict(state)
        model.eval()

        out_path = os.path.join(args.out, onnx_name)
        dummy = torch.randn(1, 3, h, w) * 255
        torch.onnx.export(
            model, dummy, out_path,
            input_names=["input"], output_names=["logits"],
            opset_version=11, do_constant_folding=True, dynamo=False,
        )

        # verifikasi: output OpenCV DNN harus sama dengan PyTorch
        import cv2
        net = cv2.dnn.readNetFromONNX(out_path)
        net.setInput(dummy.numpy())
        cv_out = net.forward()
        with torch.no_grad():
            pt_out = model(dummy).numpy()
        diff = float(np.max(np.abs(cv_out - pt_out)))
        print(f"{onnx_name}: max diff OpenCV vs PyTorch = {diff:.6f}")
        if diff > 1e-2:
            print("  PERINGATAN: selisih terlalu besar, cek versi opset/OpenCV")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
