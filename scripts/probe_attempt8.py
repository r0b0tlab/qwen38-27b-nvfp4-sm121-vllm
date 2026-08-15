import json, glob
from safetensors import safe_open

f = sorted(glob.glob("/c/model-00001-*.safetensors"))[0]
with safe_open(f, framework="pt") as st:
    keys = [k for k in st.keys() if "layers.0.linear_attn" in k]
    for k in sorted(keys):
        s = st.get_slice(k)
        print(k, s.get_shape(), s.get_dtype())
