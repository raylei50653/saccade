import tensorrt as trt
import os

logger = trt.Logger(trt.Logger.ERROR)
runtime = trt.Runtime(logger)
engine_path = "models/yolo/yolo26n.engine"

if not os.path.exists(engine_path):
    print(f"❌ Engine not found: {engine_path}")
else:
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
        print(f"--- Engine Inspection: {engine_path} ---")
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            shape = engine.get_tensor_shape(name)
            dtype = engine.get_tensor_dtype(name)
            print(
                f"Tensor {i}: {name} | Mode: {mode} | Shape: {shape} | Dtype: {dtype}"
            )
