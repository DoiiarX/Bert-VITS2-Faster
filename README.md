# Bert-VITS2-Faster
0. original algorithm repository[Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)  
1. all to trtfp16 to get 9x faster.  
## Using
0. after you get your own trained model.  (train_ms.py)  
  0.1 prepare your own 'config.yml' as the 'default_config.yml'  
1. sh bert_to_onnx.sh  
2. python infer_torch_export_onnx.py (first comment line: g_model_name = None)  
3. see inputs.py to get trt.engine  
4. python infer_backend_mix.py (test trt infer)  

# You can do
1. change the training forward to merge the all 6 submodel to one model to get VITS2.engine. 
2. improve the performance of VITS2 on emotion.  
3. when some submodel severe performance loss, you can replece the trt to onnx. 