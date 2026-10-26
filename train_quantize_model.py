#!/usr/bin/env python3
"""
Train, INT8-quantize and export a tiny 4-class CNN for the ESP32-CAM sketch.

Dataset layout (any image format, any size):
    dataset/
      pigeon/*.jpg
      monkey/*.jpg
      human/*.jpg
      other/*.jpg

Usage:
    pip install tensorflow numpy
    python train_quantize_model.py --data_dir dataset --epochs 40
    python train_quantize_model.py --synthetic        # pipeline test, random data

Outputs:
    model.tflite   - full-integer INT8 model
    model_data.h   - copy this next to the .ino (replaces the placeholder)

Design notes (why the model looks like this):
  * Input 96x96x1, values = pixel / 255.0  (the sketch applies the same scaling).
  * Every conv uses stride 2, so activations shrink fast. Peak live tensors are
    conv1 (9,216 in + 9,216 out) and conv2 (9,216 in + 9,216 out) = 18,432 B,
    which fits the < 35 KB tensor arena with room for TFLM bookkeeping.
  * Only CONV_2D, RESHAPE, FULLY_CONNECTED and SOFTMAX ops are used, which
    every TFLite Micro version supports. BatchNorm is folded into the convs.
  * For best accuracy, collect training photos with the ESP32-CAM itself
    (same lens, same grayscale pipeline) and include lots of "other".
"""
import argparse
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

CLASS_NAMES = ["pigeon", "monkey", "human", "other"]   # order == sketch order
IMG = 96
MAX_MODEL_BYTES = 150 * 1024


def build_model(batch_size=None) -> keras.Model:
    inp = keras.Input(shape=(IMG, IMG, 1), batch_size=batch_size, name="input")
    x = inp
    for filters in (4, 16, 32, 64, 64):                # 96->48->24->12->6->3
        x = layers.Conv2D(filters, 3, strides=2, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(max_value=6.0)(x)              # ReLU6 quantizes well
    x = layers.Reshape((3 * 3 * 64,))(x)               # static reshape -> plain RESHAPE op
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(len(CLASS_NAMES), activation="softmax", name="probs")(x)
    return keras.Model(inp, out, name="esp32cam_tiny_cnn")


def load_data(data_dir, batch):
    common = dict(labels="inferred", label_mode="int", class_names=CLASS_NAMES,
                  color_mode="grayscale", image_size=(IMG, IMG), batch_size=batch,
                  validation_split=0.2, seed=42)
    train_raw = keras.utils.image_dataset_from_directory(data_dir, subset="training", **common)
    val_raw = keras.utils.image_dataset_from_directory(data_dir, subset="validation", **common)

    counts = np.zeros(len(CLASS_NAMES))
    for _, y in train_raw:
        counts += np.bincount(y.numpy(), minlength=len(CLASS_NAMES))
    print("Training images per class:", dict(zip(CLASS_NAMES, counts.astype(int))))
    weights = {i: float(counts.sum() / (len(CLASS_NAMES) * max(counts[i], 1)))
               for i in range(len(CLASS_NAMES))}

    augment = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.3),
    ])
    norm = lambda x, y: (x / 255.0, y)
    aug = lambda x, y: (tf.clip_by_value(augment(x, training=True), 0.0, 255.0) / 255.0, y)

    train_ds = train_raw.map(aug, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_raw.map(norm).prefetch(tf.data.AUTOTUNE)
    calib_ds = train_raw.map(norm)                     # un-augmented, for calibration
    return train_ds, val_ds, calib_ds, weights


def synthetic_data(batch):
    rng = np.random.default_rng(0)
    x = rng.random((256, IMG, IMG, 1), dtype=np.float32)
    y = rng.integers(0, len(CLASS_NAMES), 256)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch)
    return ds, ds, ds, None


def quantize(model, calib_ds, n_samples=300) -> bytes:
    def representative_dataset():
        n = 0
        for xb, _ in calib_ds:
            for i in range(xb.shape[0]):
                yield [xb[i:i + 1].numpy().astype(np.float32)]
                n += 1
                if n >= n_samples:
                    return

    # Re-build with a fixed batch of 1 so the exported graph has only static shapes
    # (otherwise the converter emits SHAPE/PACK/STRIDED_SLICE ops that older
    # TFLite Micro builds do not support).
    export_model = build_model(batch_size=1)
    export_model.set_weights(model.get_weights())

    conv = tf.lite.TFLiteConverter.from_keras_model(export_model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def evaluate_int8(tflite_bytes, val_ds):
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    print(f"Input  tensor: {inp['dtype'].__name__} shape={inp['shape']} "
          f"scale={in_scale:.6f} zero_point={in_zp}")
    print(f"Output tensor: {out['dtype'].__name__} shape={out['shape']} "
          f"scale={out['quantization'][0]:.6f} zero_point={out['quantization'][1]}")
    ops = sorted({d["op_name"] for d in interp._get_ops_details()})
    print("Ops used:", ", ".join(ops))

    correct = total = 0
    for xb, yb in val_ds:
        for i in range(xb.shape[0]):
            q = np.clip(np.round(xb[i:i + 1].numpy() / in_scale) + in_zp, -128, 127).astype(np.int8)
            interp.set_tensor(inp["index"], q)
            interp.invoke()
            correct += int(np.argmax(interp.get_tensor(out["index"])[0]) == int(yb[i]))
            total += 1
    print(f"INT8 validation accuracy: {correct / max(total, 1):.1%} ({total} images)")


def write_header(model_bytes: bytes, path: pathlib.Path):
    rows = []
    for i in range(0, len(model_bytes), 12):
        rows.append("  " + ", ".join(f"0x{b:02x}" for b in model_bytes[i:i + 12]))
    text = (
        "// Auto-generated by train_quantize_model.py - do not edit.\n"
        "// INT8 model, input [1,96,96,1] = pixel/255, output 4 classes: "
        + ", ".join(CLASS_NAMES) + "\n"
        "#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n"
        f"const unsigned int g_model_len = {len(model_bytes)};\n\n"
        "const unsigned char g_model[] __attribute__((aligned(16))) = {\n"
        + ",\n".join(rows) + "\n};\n\n#endif  // MODEL_DATA_H\n"
    )
    path.write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="model_data.h")
    ap.add_argument("--tflite", default="model.tflite")
    ap.add_argument("--synthetic", action="store_true",
                    help="random data, only to test the export pipeline")
    args = ap.parse_args()

    if args.synthetic:
        print("*** SYNTHETIC MODE: model will be meaningless (pipeline test only) ***")
        train_ds, val_ds, calib_ds, weights = synthetic_data(args.batch)
        args.epochs = min(args.epochs, 2)
    else:
        train_ds, val_ds, calib_ds, weights = load_data(args.data_dir, args.batch)

    model = build_model()
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, class_weight=weights,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                                       restore_best_weights=True)])

    tflite_bytes = quantize(model, calib_ds)
    pathlib.Path(args.tflite).write_bytes(tflite_bytes)
    evaluate_int8(tflite_bytes, val_ds)

    kb = len(tflite_bytes) / 1024
    print(f"Model size: {kb:.1f} KB" + ("" if len(tflite_bytes) < MAX_MODEL_BYTES
                                         else "  (!! exceeds 150 KB target)"))
    write_header(tflite_bytes, pathlib.Path(args.out))
    print(f"Wrote {args.tflite} and {args.out}. Copy {args.out} next to the .ino file.")


if __name__ == "__main__":
    main()
