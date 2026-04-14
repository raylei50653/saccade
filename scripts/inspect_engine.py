import tensorrt as trt
logger = trt.Logger(trt.Logger.ERROR)
runtime = trt.Runtime(logger)
with open('models/embedding/google_siglip2-base-patch16-224.engine', 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())
    print(f"Input: {engine.get_tensor_name(0)} Shape: {engine.get_tensor_shape(engine.get_tensor_name(0))}")
    print(f"Output 1: {engine.get_tensor_name(1)} Shape: {engine.get_tensor_shape(engine.get_tensor_name(1))}")
    print(f"Output 2: {engine.get_tensor_name(2)} Shape: {engine.get_tensor_shape(engine.get_tensor_name(2))}")
