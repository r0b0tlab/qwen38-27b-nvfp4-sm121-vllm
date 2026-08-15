import json, glob
from safetensors import safe_open

f = sorted(glob.glob("/c/model-00001-*.safetensors"))[0]
with safe_open(f, framework="pt") as st:
    keys = [k for k in st.keys() if "layers.0.linear_attn.in_proj_qkv" in k or "layers.0.mlp.gate_proj" in k or "layers.0.self_attn.q_proj" in k]
    for k in sorted(keys):
        s = st.get_slice(k)
        print(k, s.get_shape(), s.get_dtype())
q = json.load(open("/c/hf_quant_config.json"))
from collections import Counter
print("algos:", dict(Counter(v.get("quant_algo") for v in q["quantization"].get("quantized_layers", {}).values())))
