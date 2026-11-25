# Monkey Guard: ESP32-CAM TinyML Detector with Servo, Flasher, GSM Alerts and Repellent Audio

An offline, on-device **monkey detector** running a quantized INT8 CNN on an **AI-Thinker ESP32-CAM**,
using internal SRAM only (PSRAM is never touched). When a monkey is confirmed, a NodeMCU plays a repellent
sound, a flasher strobes and an SMS alert is sent. No cloud, no external APIs.

> **Status:** working prototype. Detection accuracy depends heavily on how close the training photos are to
> what your camera sees (see [Results and limitations](#results-and-limitations)).

---

## How it works

```
                 motion A / motion B (2 wires)
   ┌──────────────┐  ◄──────────────────────────  ┌────────────────────────────┐
   │  ESP32-CAM   │                               │          NodeMCU           │
   │  TinyML CNN  │  ──────────────────────────►  │  2x HC-SR04 (opposite      │
   │  pan servo   │        ALERT (1 wire)         │  sides), SIM800L (SMS),    │
   │  LED flasher │                               │  DFPlayer Mini (audio)     │
   └──────────────┘                               └────────────────────────────┘
```

| Mode | What happens |
|---|---|
| **SWEEP** | The servo pans the camera left and right; every frame is classified. |
| **LOOK** | An ultrasonic sensor sees something approaching on side A or B; the camera turns to that side. |
| **ALERT** | 2 consecutive monkey frames at 80% confidence or more: servo holds, flasher strobes, the NodeMCU loops the repellent sound and sends **one** SMS (5-minute cooldown). |

The alert clears 8 seconds after the last monkey frame, then the sweep resumes. While the flasher is
strobing, frames are only classified in the dark gaps between bursts, so the light does not wash out the picture.

---

## Target classes

1. `monkey`
2. `other` (background, people, other animals, generic objects)

The class list is read from `model_data.h`, so the firmware follows whatever folders you train with.
The guard logic needs a class named `monkey`.

---

## Project structure

```
.
├── train_quantize_model.py               Training, INT8 quantization, C-header export
├── esp32_tinyml_detector/
│   ├── esp32_tinyml_detector.ino         ESP32-CAM firmware (camera, inference, servo, flasher, link)
│   ├── model_data.h                      Generated model (INT8 byte array + class names)
│   └── selftest_data.h                   Optional: 4 known images to verify the on-device model
├── nodemcu_monkey_guard/
│   └── nodemcu_monkey_guard.ino          NodeMCU firmware (ultrasonics, SMS, audio)
└── WIRING.md                             Full pinout, power notes, bring-up order
```

The dataset is not included. See [Dataset layout](#dataset-layout).

---

## The model

| Item | Value |
|---|---|
| Input | 96×96 grayscale, `pixel / 255` |
| Architecture | 5 stride-2 Conv2D layers (4, 16, 32, 64, 64 filters) with BatchNorm + ReLU6, then a softmax layer |
| Parameters | about 60 K |
| Quantization | full-integer INT8 (representative dataset), INT8 input and output |
| Size | about 71 KB |
| Ops used | `CONV_2D`, `RESHAPE`, `FULLY_CONNECTED`, `SOFTMAX` (supported by every TFLite Micro version) |

Every convolution uses stride 2, so activation memory shrinks quickly: the peak is about 18 KB
(9,216 B in + 9,216 B out at the first layers), which is what lets the model fit in a 28 KB arena.

### Memory budget (no PSRAM)

| Memory | Size | Location |
|---|---|---|
| Frame buffer (96×96 grayscale) | 9,216 B | Internal DRAM |
| TFLite tensor arena | 28,672 B (28 KB, limit < 35 KB) | Internal SRAM (static) |
| Quantized model (`g_model`) | about 71 KB | Flash (`const`, 0 B of SRAM) |
| Free heap at boot | about 224 KB (measured on my board) | Internal DRAM |

The sketch prints `Arena used: ... of 28672 bytes` at boot so you can check your own model.

---

## Training

### Dataset layout

Each sub-folder is one class. Any image format and size works; images are resized to 96×96 grayscale.

```
dataset/
  monkey/*.jpg      the thing you want to detect
  other/*.jpg       everything else: empty scenes, walls, trees, people, pigeons, dogs...
```

Tips that made the biggest difference:
- **Photos taken with the ESP32-CAM itself** match the real input far better than web photos.
- Include the **real location** and distractors in `other`, or the model will call everything a monkey.
- Keep the classes reasonably balanced and delete wrong or blurry images.

### Run

```bash
pip install tensorflow numpy

python train_quantize_model.py --synthetic                          # pipeline test with random data
python train_quantize_model.py --data_dir dataset --epochs 80       # real training
```

Outputs `model.tflite` and `model_data.h`. Copy `model_data.h` into `esp32_tinyml_detector/`.
Options: `--out`, `--tflite`, `--batch`, `--epochs`. The script prints the INT8 accuracy on clean and
on degraded (blurred, noisy, dimmed) validation images, the ops used and the model size.

### Lesson learned: shortcut learning

My first model scored 100% validation accuracy but failed on the real camera. The monkey photos were
lower-resolution than the "other" photos, so the network learned "blurry means monkey". Its answers flipped
with a little blur or noise. The fix in the training script is to apply random blur, noise, brightness and
contrast to **every** training image regardless of class, and to report accuracy on degraded images as well.
Clean validation accuracy dropped, but behaviour on the real camera became sensible.

---

## ESP32-CAM firmware

### Arduino IDE setup

1. **Board package:** `esp32` by Espressif, version **2.0.17** (recommended). The TensorFlowLite_ESP32 library
   can fail to compile on core 3.x.
2. **Library:** `TensorFlowLite_ESP32` by Tanaka Masayuki (Library Manager).
3. **Board settings:** `AI Thinker ESP32-CAM` (or `ESP32 Wrover Module` with PSRAM Disabled),
   Partition Scheme `Huge APP (3MB No OTA/1MB SPIFFS)`, Serial Monitor 115200 baud.
4. Remove the SD card: its pins are used for the servo, flasher and link wires.

### Flashing wiring (FTDI)

| FTDI | ESP32-CAM | Notes |
|---|---|---|
| 5V | 5V | Use a reliable 5 V ≥ 2 A supply |
| GND | GND | Common ground |
| TX | U0R (GPIO 3) | Cross TX/RX |
| RX | U0T (GPIO 1) | Cross RX/TX |
| GND | GPIO 0 | **Only while flashing**; remove it, then press RESET |

### Options at the top of the sketch

| Setting | Default | Meaning |
|---|---|---|
| `MONKEY_CONFIDENCE` | 0.80 | Probability needed for a frame to count as a monkey |
| `CONFIRM_FRAMES` | 2 | Consecutive monkey frames before an alert |
| `SERVO_ANGLE_A` / `_B` | 20 / 160 | Directions watched by ultrasonic sensors A and B |
| `SWEEP_DWELL_MS`, `SERVO_STEP_DEG` | 1500 / 20 | Sweep speed and step |
| `CAM_VFLIP`, `CAM_HMIRROR` | 0 | Flip the image if the camera is mounted upside down or mirrored |
| `DEBUG_CAMERA_VIEW` | 0 | 1 = pixel statistics and an ASCII picture of what the camera sees |
| `RUN_SELFTEST` | 0 | 1 = at boot, feed 4 known images and compare with the Python results |

---

## NodeMCU firmware

Board: `NodeMCU 1.0 (ESP-12E Module)`, no extra library. Edit `PHONE_NUMBER` (with country code) before
uploading. **Never commit your real phone number to a public repository.**

Serial Monitor test keys: `d` print distances, `a` audio on, `s` audio off, `g` send a test SMS.

---

## Hardware and wiring

Full pinout, divider values, power notes and the bring-up order are in **[WIRING.md](WIRING.md)**. Summary:

**Link between the boards (3 wires + common GND)**

| NodeMCU | ESP32-CAM | Signal |
|---|---|---|
| D0 | GPIO13 | Motion on side A |
| D3 | GPIO15 | Motion on side B |
| D8 | GPIO2 | ALERT (monkey confirmed) |

**ESP32-CAM:** servo signal on GPIO14, external flasher on GPIO12 (through a transistor with a 10 kΩ base
pull-down), the on-board white LED on GPIO4 also strobes.

**NodeMCU:** both HC-SR04 TRIG pins on D1, ECHO A on D2, ECHO B on D5 (each through a 1 kΩ / 2 kΩ divider),
DFPlayer RX on D4 (through 1 kΩ), SIM800L TXD on D6 and RXD on D7 (through 1 kΩ).

**Power notes**
- Feed the ESP32-CAM, NodeMCU, sensors, DFPlayer and servo from a 5 V ≥ 3 A supply with 470-1000 µF capacitors.
  Give the servo its own supply if the board resets when it moves.
- The SIM800L needs its own 4.0 V supply (≥ 2 A burst). Never power it from the NodeMCU.
- All grounds must be joined.
- SIM800L is a **2G** module: Jio SIMs will not work; use an operator that still offers 2G in your area.

---

## Bring-up order

1. **NodeMCU alone:** check distances (`d`), audio (`a`/`s`) and a test SMS (`g`).
2. **ESP32-CAM alone** (link wires unplugged): the camera sweeps and prints `[SWEEP  80deg] ...`.
3. **Connect the link wires:** wave a hand at ultrasonic A; the camera turns to side A (`[LOOK  20deg]`).
4. **Show a monkey picture:** you get `[ALERT] MONKEY CONFIRMED`, the strobe, the sound and one SMS.

If the output never changes, set `RUN_SELFTEST` and `DEBUG_CAMERA_VIEW` to 1. The self-test tells you whether
the model runs correctly on the chip; the ASCII picture shows what the camera really sees.

---

## Results and limitations

| Metric | Value |
|---|---|
| Training images | [N] (monkey: [N], other: [N]) |
| INT8 validation accuracy, clean images | [88]% |
| INT8 validation accuracy, blurred/noisy/dim images | [33]% |
| Inference time on the ESP32-CAM | [3000] ms per frame |

- A tiny model trained on a small dataset will produce **false alarms and misses**. The 2-frame confirmation
  and the 80% threshold reduce them but do not remove them.
- Accuracy is best near the conditions of the training photos (distance, light, angle).
- Monkeys can get used to a repeated sound; consider several repellent sounds over time.

## Roadmap

- Capture training images with the ESP32-CAM itself and retrain.
- Add pigeon and human classes back with enough data.
- Log detections with time and confidence.

## Acknowledgements

TensorFlow / Keras and TensorFlow Lite Micro; the `TensorFlowLite_ESP32` Arduino library by Tanaka Masayuki.
Dataset sources: [kaggle.com]
