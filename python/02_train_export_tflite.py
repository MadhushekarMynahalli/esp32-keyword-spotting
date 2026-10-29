"""Step 2: train a small depthwise-separable CNN, quantise to int8 TFLite, check accuracy."""
import argparse, json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from config import *


def load(n):
    return np.load(ART_DIR / f"X_{n}.npy"), np.load(ART_DIR / f"y_{n}.npy")


def ds_block(x, filters, stride):
    x = layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    return x


def build():
    inp = layers.Input((NFRAMES, NMEL, 1))
    x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = ds_block(x, 64, 1)
    x = ds_block(x, 96, 2)
    x = ds_block(x, 128, 2)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(len(LABELS), activation="softmax")(x)
    return models.Model(inp, out)


def eval_tflite(tfl, X, y):
    it = tf.lite.Interpreter(model_content=tfl); it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    s, z = i["quantization"]
    pred = []
    for k in range(len(X)):
        q = np.clip(np.round(X[k].astype(np.float32)[None, :, :, None] / s) + z, -128, 127).astype(np.int8)
        it.set_tensor(i["index"], q); it.invoke()
        pred.append(int(np.argmax(it.get_tensor(o["index"])[0])))
    return float(np.mean(np.array(pred) == y)), np.array(pred)


def confusion(y, p):
    n = len(LABELS); m = np.zeros((n, n), int)
    for a, b in zip(y, p): m[a, b] += 1
    print("confusion (rows=true, cols=pred):  " + " ".join(f"{l[:5]:>5}" for l in LABELS))
    for k, r in enumerate(m): print(f"{LABELS[k][:14]:>14} " + " ".join(f"{v:5d}" for v in r))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    tf.random.set_seed(0)
    Xtr, ytr = load("train"); Xva, yva = load("val"); Xte, yte = load("test"); Xta, yta = load("test_adc")

    def mk(X, y, train):
        d = tf.data.Dataset.from_tensor_slices((X, y))
        if train: d = d.shuffle(20000, seed=0)
        return d.batch(128).map(lambda x, t: (tf.cast(x, tf.float32)[..., None], t)).prefetch(2)

    model = build(); model.summary()
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(mk(Xtr, ytr, True), validation_data=mk(Xva, yva, False), epochs=a.epochs, callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
    print("float  test acc:", model.evaluate(mk(Xte, yte, False), verbose=0)[1],
          "| test_adc acc:", model.evaluate(mk(Xta, yta, False), verbose=0)[1])
    model.save(ART_DIR / "kws_float.keras")

    rng = np.random.default_rng(1)
    def rep():
        for k in rng.choice(len(Xtr), 500, replace=False):
            yield [Xtr[k].astype(np.float32)[None, :, :, None]]
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tfl = conv.convert()
    (ART_DIR / "kws_int8.tflite").write_bytes(tfl)
    print(f"int8 model: {len(tfl)/1024:.1f} KB")
    try:
        it = tf.lite.Interpreter(model_content=tfl)
        print("ops used (register all of these in the op resolver):",
              sorted({d["op_name"] for d in it._get_ops_details()}))
    except Exception as e:
        print("could not list ops:", e)

    acc, p = eval_tflite(tfl, Xte, yte); print("int8   test acc:", acc); confusion(yte, p)
    print("int8   test_adc acc:", eval_tflite(tfl, Xta, yta)[0])
