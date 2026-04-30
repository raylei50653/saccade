import onnx
import sys


def make_dynamic(input_path: str, output_path: str) -> None:
    print(f"🔄 Converting {input_path} to dynamic batch...")
    model = onnx.load(input_path)

    # 取得輸入節點
    inputs = model.graph.input
    for input_node in inputs:
        # 將第一個維度 (Batch) 設為具名變量 'batch'
        input_node.type.tensor_type.shape.dim[0].dim_param = "batch"
        print(f"✅ Input '{input_node.name}' is now dynamic.")

    # 取得輸出節點
    outputs = model.graph.output
    for output_node in outputs:
        output_node.type.tensor_type.shape.dim[0].dim_param = "batch"
        print(f"✅ Output '{output_node.name}' is now dynamic.")

    onnx.save(model, output_path)
    print(f"🎉 Saved dynamic ONNX to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python make_onnx_dynamic.py input.onnx output.onnx")
    else:
        make_dynamic(sys.argv[1], sys.argv[2])
