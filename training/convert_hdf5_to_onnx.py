#!/usr/bin/env python3

import argparse
import os
import sys

# Ensure TensorFlow logs are quiet
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

try:
    import tensorflow as tf
    # Use tf.keras to load models
    from tensorflow import keras
except Exception as e:
    print("ERROR: TensorFlow is required to convert Keras HDF5 to ONNX.")
    print(str(e))
    sys.exit(1)

try:
    import tf2onnx
    from tf2onnx import tf_loader
except Exception as e:
    print("ERROR: tf2onnx is required. Install with: pip install tf2onnx")
    print(str(e))
    sys.exit(1)


# --- Custom objects used by the training model ---
from typing import Any

import keras.backend as K
from keras.constraints import Constraint
from keras.layers import Input, Dense, GRU, concatenate
from keras.models import Model


def my_crossentropy(y_true, y_pred):
    return K.mean(2 * K.abs(y_true - 0.5) * K.binary_crossentropy(y_pred, y_true), axis=-1)


def mymask(y_true):
    return K.minimum(y_true + 1.0, 1.0)


def msse(y_true, y_pred):
    return K.mean(mymask(y_true) * K.square(K.sqrt(y_pred) - K.sqrt(y_true)), axis=-1)


def mycost(y_true, y_pred):
    return K.mean(
        mymask(y_true)
        * (
            10 * K.square(K.square(K.sqrt(y_pred) - K.sqrt(y_true)))
            + K.square(K.sqrt(y_pred) - K.sqrt(y_true))
            + 0.01 * K.binary_crossentropy(y_pred, y_true)
        ),
        axis=-1,
    )


def my_accuracy(y_true, y_pred):
    return K.mean(2 * K.abs(y_true - 0.5) * K.equal(y_true, K.round(y_pred)), axis=-1)


class WeightClip(Constraint):
    # Accept **kwargs to be compatible with Keras deserialization that may pass 'name' etc.
    def __init__(self, c=2, **kwargs):  # kwargs may include 'name'
        super().__init__()
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__, 'c': self.c}


CUSTOM_OBJECTS = {
    'my_crossentropy': my_crossentropy,
    'mymask': mymask,
    'msse': msse,
    'mycost': mycost,
    'my_accuracy': my_accuracy,
    'WeightClip': WeightClip,
}


def rebuild_model_with_states(training_model: Model) -> Model:
    """
    自动重建模型，添加GRU隐状态输入/输出端口。
    如果模型已经有GRU状态端口，直接返回原模型。
    """
    # 检查是否已有GRU状态端口
    if len(training_model.inputs) == 4 and len(training_model.outputs) == 5:
        print("  Model already has GRU state ports, skipping rebuild")
        return training_model
    
    print("  Rebuilding model with GRU state inputs/outputs...")
    
    # 新的推理输入（带状态）
    features_in = Input(shape=(None, 42), name='features')
    vad_state_in = Input(shape=(24,), name='vad_gru_state')
    noise_state_in = Input(shape=(48,), name='noise_gru_state')
    denoise_state_in = Input(shape=(96,), name='denoise_gru_state')

    # 复制训练模型的层配置并加载权重
    # 1) input_dense
    input_dense_src = training_model.get_layer('input_dense')
    input_dense = Dense(24, activation='tanh', name='input_dense_export',
                        kernel_constraint=input_dense_src.kernel_constraint,
                        bias_constraint=input_dense_src.bias_constraint)
    tmp_export = input_dense(features_in)
    input_dense.set_weights(input_dense_src.get_weights())

    # 2) vad_gru (return_sequences+return_state)
    vad_gru_src = training_model.get_layer('vad_gru')
    vad_gru_exp = GRU(24, activation='tanh', recurrent_activation='sigmoid',
                      return_sequences=True, return_state=True, name='vad_gru_export',
                      kernel_regularizer=vad_gru_src.kernel_regularizer,
                      recurrent_regularizer=vad_gru_src.recurrent_regularizer,
                      kernel_constraint=vad_gru_src.kernel_constraint,
                      recurrent_constraint=vad_gru_src.recurrent_constraint,
                      bias_constraint=vad_gru_src.bias_constraint)
    vad_seq, vad_state_out = vad_gru_exp(tmp_export, initial_state=vad_state_in)
    vad_gru_exp.set_weights(vad_gru_src.get_weights())

    # 3) vad_output
    vad_output_src = training_model.get_layer('vad_output')
    vad_output_exp_layer = Dense(1, activation='sigmoid', name='vad_output_export',
                                 kernel_constraint=vad_output_src.kernel_constraint,
                                 bias_constraint=vad_output_src.bias_constraint)
    vad_output_exp = vad_output_exp_layer(vad_seq)
    vad_output_exp_layer.set_weights(vad_output_src.get_weights())

    # 4) noise_gru 输入：concat([tmp_export, vad_seq, features_in])
    noise_in = concatenate([tmp_export, vad_seq, features_in], name='noise_concat_export')
    noise_gru_src = training_model.get_layer('noise_gru')
    noise_gru_exp = GRU(48, activation='relu', recurrent_activation='sigmoid',
                        return_sequences=True, return_state=True, name='noise_gru_export',
                        kernel_regularizer=noise_gru_src.kernel_regularizer,
                        recurrent_regularizer=noise_gru_src.recurrent_regularizer,
                        kernel_constraint=noise_gru_src.kernel_constraint,
                        recurrent_constraint=noise_gru_src.recurrent_constraint,
                        bias_constraint=noise_gru_src.bias_constraint)
    noise_seq, noise_state_out = noise_gru_exp(noise_in, initial_state=noise_state_in)
    noise_gru_exp.set_weights(noise_gru_src.get_weights())

    # 5) denoise_gru 输入：concat([vad_seq, noise_seq, features_in])
    denoise_in = concatenate([vad_seq, noise_seq, features_in], name='denoise_concat_export')
    denoise_gru_src = training_model.get_layer('denoise_gru')
    denoise_gru_exp = GRU(96, activation='tanh', recurrent_activation='sigmoid',
                          return_sequences=True, return_state=True, name='denoise_gru_export',
                          kernel_regularizer=denoise_gru_src.kernel_regularizer,
                          recurrent_regularizer=denoise_gru_src.recurrent_regularizer,
                          kernel_constraint=denoise_gru_src.kernel_constraint,
                          recurrent_constraint=denoise_gru_src.recurrent_constraint,
                          bias_constraint=denoise_gru_src.bias_constraint)
    denoise_seq, denoise_state_out = denoise_gru_exp(denoise_in, initial_state=denoise_state_in)
    denoise_gru_exp.set_weights(denoise_gru_src.get_weights())

    # 6) denoise_output
    denoise_output_src = training_model.get_layer('denoise_output')
    denoise_output_exp_layer = Dense(22, activation='sigmoid', name='denoise_output_export',
                                     kernel_constraint=denoise_output_src.kernel_constraint,
                                     bias_constraint=denoise_output_src.bias_constraint)
    denoise_output_exp = denoise_output_exp_layer(denoise_seq)
    denoise_output_exp_layer.set_weights(denoise_output_src.get_weights())

    export_model = Model(
        inputs=[features_in, vad_state_in, noise_state_in, denoise_state_in],
        outputs=[denoise_output_exp, vad_output_exp, vad_state_out, noise_state_out, denoise_state_out],
        name='rnnoise_export_with_states'
    )
    
    print("  ✓ Model rebuilt successfully with GRU state ports")
    return export_model


def convert(hdf5_path: str, onnx_path: str, opset: int = 13, auto_rebuild: bool = False) -> None:
    if not os.path.isfile(hdf5_path):
        raise FileNotFoundError(f"HDF5 model not found: {hdf5_path}")

    print(f"Loading Keras model from: {hdf5_path}")
    # Load with custom objects registered for deserialization
    model = keras.models.load_model(hdf5_path, custom_objects=CUSTOM_OBJECTS)
    
    # Auto-rebuild model with GRU states if needed
    if auto_rebuild:
        print("\n=== Auto-Rebuild Mode ===")
        print("  Checking if model needs GRU state ports...")
        model = rebuild_model_with_states(model)
        print("  Model ready for conversion with GRU state ports\n")

    # Check if the model has GRU state inputs/outputs
    num_inputs = len(model.inputs)
    num_outputs = len(model.outputs)
    
    print(f"Model has {num_inputs} input(s) and {num_outputs} output(s)")
    
    # Print input information
    for i, inp in enumerate(model.inputs):
        print(f"  Input {i}: {inp.name}, shape: {inp.shape}")
    
    # Print output information
    for i, out in enumerate(model.outputs):
        print(f"  Output {i}: {out.name}, shape: {out.shape}")
    
    # Check if this is a model with GRU states (4 inputs and 5 outputs)
    if num_inputs == 4 and num_outputs == 5:
        print("Detected model with GRU state inputs/outputs")
        # Build input signature for model with efficient state management
        input_specs = []
        for inp in model.inputs:
            inp_name = inp.name.split(':')[0]
            inp_shape = inp.shape.as_list()
            
            # Handle different input shapes
            if len(inp_shape) == 3:  # features: (None, None, 42)
                spec = tf.TensorSpec([None, None, inp_shape[2]], tf.float32, name=inp_name)
            elif len(inp_shape) == 2:  # GRU states: (None, hidden_size)
                spec = tf.TensorSpec([None, inp_shape[1]], tf.float32, name=inp_name)
            else:
                # Fallback: use dynamic shape
                spec = tf.TensorSpec([None] * len(inp_shape), tf.float32, name=inp_name)
            
            input_specs.append(spec)
        
        print(f"Converting to ONNX (opset {opset}) with GRU state inputs/outputs...")
        # Convert with all input signatures
        tf2onnx.convert.from_keras(model, input_signature=input_specs, output_path=onnx_path, opset=opset)
        
    elif num_inputs == 1:
        print("Detected standard model without GRU state ports")
        # Use a dynamic input signature (None, None, 42) to preserve time dimension flexibility
        input_name = model.inputs[0].name.split(':')[0]
        spec = (tf.TensorSpec([None, None, 42], tf.float32, name=input_name),)
        
        print(f"Converting to ONNX (opset {opset})...")
        # Convert directly from the Keras model
        tf2onnx.convert.from_keras(model, input_signature=spec, output_path=onnx_path, opset=opset)
    else:
        # Generic conversion for models with multiple inputs but unknown structure
        print(f"Converting to ONNX (opset {opset}) with {num_inputs} inputs...")
        input_specs = []
        for inp in model.inputs:
            inp_name = inp.name.split(':')[0]
            inp_shape = inp.shape.as_list()
            # Use dynamic shapes for flexibility
            spec = tf.TensorSpec([None] * len(inp_shape), tf.float32, name=inp_name)
            input_specs.append(spec)
        tf2onnx.convert.from_keras(model, input_signature=input_specs, output_path=onnx_path, opset=opset)

    print(f"Saved ONNX model to: {onnx_path}")
    

def main():
    parser = argparse.ArgumentParser(description='Convert Keras HDF5 model to ONNX for RNNoise.')
    parser.add_argument('--input', '-i', required=True, help='Path to Keras HDF5 model file')
    parser.add_argument('--output', '-o', required=False, help='Path to output ONNX file')
    parser.add_argument('--opset', type=int, default=13, help='ONNX opset version (default: 13)')
    parser.add_argument('--auto-rebuild', action='store_true', 
                        help='Automatically rebuild model with GRU state ports if missing')
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = args.output
    if not output_path:
        base, _ = os.path.splitext(input_path)
        output_path = base + '.onnx'
    output_path = os.path.abspath(output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    convert(input_path, output_path, opset=args.opset, auto_rebuild=args.auto_rebuild)

if __name__ == '__main__':
    main()


