import numpy as np
x_enc_p = np.zeros((1, 46), dtype=np.int64)
x_length = 46*np.ones((1,), dtype=np.int64)
tone = np.zeros((1, 46), dtype=np.int64)
language = np.zeros((1, 46), dtype=np.int64)
bert_zh = np.random.randn(1, 1024, 46).astype(np.float32)
bert_ja = np.random.randn(1, 1024, 46).astype(np.float32)
bert_en = np.random.randn(1, 1024, 46).astype(np.float32)
g = np.random.rand(1, 256, 1).astype(np.float32)

inputs_enc_p = {
    "x":x_enc_p,
    "x_lengths":x_length,
    "tone": tone,
    "language": language,
    "bert": bert_zh,
    "ja_bert": bert_ja,
    "en_bert": bert_en,
    "g": g
}

x_sdp = np.random.rand(1, 192, 46).astype(np.float32)
x_dp = np.random.rand(1, 192, 46).astype(np.float32)
x_mask = np.random.rand(1, 1, 46).astype(np.float32)
g = np.random.rand(1, 256, 1).astype(np.float32)
noise_scale_w = 0.8*np.ones((1,), dtype=np.float32)

inputs_sdp = {
    "x":x_sdp,
    "x_mask":x_mask,
    "g":g,
    "noise_scale_w": noise_scale_w
}

inputs_dp = {
    "x":x_dp,
    "x_mask":x_mask,
    "g":g
}

# --data-loader-script

z_p = np.random.randn(1, 192, 201).astype(np.float32)
y_mask = np.random.randn(1, 1, 201).astype(np.float32)
inputs_flow = {
    "z_p":z_p,
    "y_mask": y_mask,
    "g":g
}

z_in = np.random.randn(1, 192, 304).astype(np.float32)
inputs_dec = {
    'z_in':z_in,
    "g":g
}


sid = np.zeros(1, dtype=np.int64)
inputs_emb = {
    'sid':sid
}

def loader():
    yield inputs_dp

# polygraphy run onnx_weights/G_122000/enc.onnx --trt --fp-to-fp16 --rtol=0.1 --atol=0.1 --fail-fast --trt-min-shapes x:[1,2] tone:[1,2] language:[1,2] bert:[1,1024,2] ja_bert:[1,1024,2] en_bert:[1,1024,2] --trt-opt-shapes x:[1,30] tone:[1,30] language:[1,30] bert:[1,1024,30] ja_bert:[1,1024,30] en_bert:[1,1024,30] --trt-max-shapes x:[1,200] tone:[1,200] language:[1,200] bert:[1,1024,200] ja_bert:[1,1024,200] en_bert:[1,1024,200] --save-engine xx.engine --data-loader-script inputs.py:loader


# 35, 43,27,23,15,61,41,67,81,71,77,105,

# polygraphy run _sdp_.onnx --trt --fp-to-fp16 --rtol=0.01 --atol=0.01 --fail-fast --trt-min-shapes x:[1,192,5] x_mask:[1,1,5] --trt-opt-shapes x:[1,192,35] x_mask:[1,1,35] --trt-max-shapes x:[1,192,200] x_mask:[1,1,200]  --save-engine _sdp_.engine --data-loader-script inputs.py:loader

# polygraphy run _sdp_.onnx --trt --fp-to-fp16 --rtol=0.01 --atol=0.01 --fail-fast --trt-min-shapes x:[1,192,5] x_mask:[1,1,5] --trt-opt-shapes x:[1,192,35] x_mask:[1,1,35] --trt-max-shapes x:[1,192,200] x_mask:[1,1,200]  --save-engine _sdp_.engine --data-loader-script inputs.py:loader

'''
polygraphy run onnx_exports/G_122000/flow_new.onnx --trt --fp-to-fp16 --rtol=0.01 --atol=0.01 --fail-fast \
    --trt-min-shapes z_p:[1,192,100] y_mask:[1,1,100] \
    --trt-opt-shapes z_p:[1,192,260] y_mask:[1,1,260] \
    --trt-max-shapes z_p:[1,192,1500] y_mask:[1,1,1500]  \
    --save-engine engines/_flow_new.engine --data-loader-script inputs.py:loader


polygraphy run onnx_exports/G_122000/dec.onnx --trt --fp-to-fp16 --rtol=0.01 --atol=0.01 --fail-fast \
    --trt-min-shapes z_in:[1,192,100] \
    --trt-opt-shapes z_in:[1,192,260] \
    --trt-max-shapes z_in:[1,192,1500] \
    --save-engine engines/_dec_.engine --data-loader-script inputs.py:loader
    
polygraphy run onnx_exports/G_122000/dp.onnx --trt --fp-to-fp16 --rtol=0.01 --atol=0.01 --fail-fast \
    --trt-min-shapes x:[1,192,5] x_mask:[1,1,5] \
    --trt-opt-shapes x:[1,192,30] x_mask:[1,1,30] \
    --trt-max-shapes x:[1,192,150] x_mask:[1,1,150]  \
    --save-engine engines/_dp_.engine --data-loader-script inputs.py:loader
'''