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


def convert(hdf5_path: str, onnx_path: str, opset: int = 13) -> None:
    if not os.path.isfile(hdf5_path):
        raise FileNotFoundError(f"HDF5 model not found: {hdf5_path}")

    print(f"Loading Keras model from: {hdf5_path}")
    # Load with custom objects registered for deserialization
    model = keras.models.load_model(hdf5_path, custom_objects=CUSTOM_OBJECTS)

    # Use a dynamic input signature (None, None, 42) to preserve time dimension flexibility
    input_name = model.inputs[0].name.split(':')[0]
    spec = (tf.TensorSpec([None, None, 42], tf.float32, name=input_name),)

    print(f"Converting to ONNX (opset {opset})...")
    # Convert directly from the Keras model
    tf2onnx.convert.from_keras(model, input_signature=spec, output_path=onnx_path, opset=opset)

    print(f"Saved ONNX model to: {onnx_path}")
    

def main():
    parser = argparse.ArgumentParser(description='Convert Keras HDF5 model to ONNX for RNNoise.')
    parser.add_argument('--input', '-i', required=True, help='Path to Keras HDF5 model file')
    parser.add_argument('--output', '-o', required=False, help='Path to output ONNX file')
    parser.add_argument('--opset', type=int, default=13, help='ONNX opset version (default: 13)')
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = args.output
    if not output_path:
        base, _ = os.path.splitext(input_path)
        output_path = base + '.onnx'
    output_path = os.path.abspath(output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    convert(input_path, output_path, opset=args.opset)

if __name__ == '__main__':
    main()


