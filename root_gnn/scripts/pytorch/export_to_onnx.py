#!/usr/bin/env python
"""
Export PyTorch models (LSTM and LTC) to ONNX format
"""
import argparse
import torch
import torch.onnx
import os
from train_torch import RecurrentEncoder
from ncps.torch import LTC, CfC

def export_model_to_onnx(model_path, model_type, output_path, 
                        lstm_units_1_1=32, lstm_units_1_2=32, 
                        merge_dense_units_1=64, 
                        batch_size=1, sequence_length=10):
    """
    Export a trained PyTorch model to ONNX format
    
    Args:
        model_path: Path to the saved PyTorch model (.pt file)
        model_type: Type of model ('lstm', 'ltc', or 'cfc')
        output_path: Path where to save the ONNX model
        lstm_units_1_1: Units for first LSTM/LTC layer
        lstm_units_1_2: Units for second LSTM/LTC layer
        merge_dense_units_1: Units for merge dense layer
        batch_size: Batch size for the dummy input
        sequence_length: Sequence length for the dummy input
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define input shapes
    input_shape_1 = (sequence_length, 6)
    input_shape_2 = (6, 4)
    input_shape_3 = 8
    
    # Create the model with appropriate RNN block
    if model_type == "ltc":
        rnn_model = RecurrentEncoder(
            input_shape_1, input_shape_2, input_shape_3, 
            rnn_block=LTC, lnn_block=True,
            lstm_units_1_1=lstm_units_1_1, 
            lstm_units_1_2=lstm_units_1_2, 
            merge_dense_units_1=merge_dense_units_1
        )
    elif model_type == "cfc":
        rnn_model = RecurrentEncoder(
            input_shape_1, input_shape_2, input_shape_3, 
            rnn_block=CfC, lnn_block=True,
            lstm_units_1_1=lstm_units_1_1, 
            lstm_units_1_2=lstm_units_1_2, 
            merge_dense_units_1=merge_dense_units_1
        )
    else:  # default to LSTM
        rnn_model = RecurrentEncoder(
            input_shape_1, input_shape_2, input_shape_3,
            lstm_units_1_1=lstm_units_1_1, 
            lstm_units_1_2=lstm_units_1_2, 
            merge_dense_units_1=merge_dense_units_1
        )
    
    # Load the trained weights
    print(f"Loading model from {model_path}")
    rnn_model.load_state_dict(torch.load(model_path, map_location=device))
    rnn_model.to(device)
    rnn_model.eval()
    
    # Create dummy inputs with correct shapes
    dummy_track = torch.randn(batch_size, *input_shape_1, device=device)
    dummy_cluster = torch.randn(batch_size, *input_shape_2, device=device) 
    dummy_hlv = torch.randn(batch_size, input_shape_3, device=device)
    
    # Export to ONNX
    print(f"Exporting {model_type} model to ONNX format...")
    
    # Define input names and dynamic axes for flexibility
    input_names = ['track_input', 'hlv_input', 'cluster_input']
    output_names = ['output']
    
    dynamic_axes = {
        'track_input': {0: 'batch_size'},
        'hlv_input': {0: 'batch_size'},
        'cluster_input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
    
    # Export the model
    torch.onnx.export(
        rnn_model,
        (dummy_track, dummy_hlv, dummy_cluster),
        output_path,
        export_params=True,
        opset_version=11,  # You can adjust this based on your needs
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=True
    )
    
    print(f"Model exported successfully to {output_path}")
    
    # Verify the ONNX model (optional but recommended)
    try:
        import onnx
        import onnxruntime
        
        # Check that the model is well formed
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed!")
        
        # Test inference with ONNX Runtime
        ort_session = onnxruntime.InferenceSession(output_path)
        
        # Prepare inputs
        ort_inputs = {
            'track_input': dummy_track.cpu().numpy(),
            'hlv_input': dummy_hlv.cpu().numpy(),
            'cluster_input': dummy_cluster.cpu().numpy()
        }
        
        # Run inference
        ort_outputs = ort_session.run(None, ort_inputs)
        print(f"ONNX inference test passed! Output shape: {ort_outputs[0].shape}")
        
    except ImportError:
        print("WARNING: onnx or onnxruntime not installed. Skipping validation.")
        print("Install with: pip install onnx onnxruntime")
    except Exception as e:
        print(f"WARNING: ONNX validation failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX')
    parser.add_argument('model_path', help='Path to the saved PyTorch model (.pt file)')
    parser.add_argument('--model_type', '-t', choices=['lstm', 'ltc', 'cfc'], 
                       default='lstm', help='Type of model to export')
    parser.add_argument('--output', '-o', default=None, 
                       help='Output path for ONNX model (default: model_path with .onnx extension)')
    parser.add_argument('--lstm_units_1_1', default=32, type=int, 
                       help='Units for first LSTM/LTC layer')
    parser.add_argument('--lstm_units_1_2', default=32, type=int, 
                       help='Units for second LSTM/LTC layer')
    parser.add_argument('--merge_dense_units_1', default=64, type=int, 
                       help='Units for merge dense layer')
    parser.add_argument('--batch_size', default=1, type=int, 
                       help='Batch size for dummy input (can be dynamic)')
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output is None:
        base_name = os.path.splitext(args.model_path)[0]
        args.output = f"{base_name}_{args.model_type}.onnx"
    
    # Export the model
    export_model_to_onnx(
        model_path=args.model_path,
        model_type=args.model_type,
        output_path=args.output,
        lstm_units_1_1=args.lstm_units_1_1,
        lstm_units_1_2=args.lstm_units_1_2,
        merge_dense_units_1=args.merge_dense_units_1,
        batch_size=args.batch_size
    )