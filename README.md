# ESP32-CAM TinyML Image Classifier (Non-PSRAM Safe)

An on-device TinyML image classification system running entirely on the internal SRAM of an **AI-Thinker ESP32-CAM** without external PSRAM.

---

## 🎯 Target Categories
1. `pigeon`
2. `monkey`
3. `human`
4. `other` (background / generic objects)

---

## 📁 Project Structure
- [esp32_tinyml_detector.ino](file:///d:/dhanu/Miniproject/tinyml/esp32_tinyml_detector.ino) - Main Arduino sketch (camera driver, INT8 normalization, local inference loop).
- [model_data.h](file:///d:/dhanu/Miniproject/tinyml/model_data.h) - C header containing the quantized INT8 model byte array aligned to 16 bytes.
- [train_quantize_model.py](file:///d:/dhanu/Miniproject/tinyml/train_quantize_model.py) - Python script to train an ultra-lightweight Depthwise-Separable CNN, apply INT8 quantization, and export directly to `model_data.h`.

---

## ⚡ Memory Budget Breakdown (Non-PSRAM)
| Memory Segment | Size | Location | Safety Margin |
|---|---|---|---|
| **Frame Buffer (96x96 Grayscale)** | 9,216 bytes (~9.0 KB) | Internal DRAM (DMA) | Safe (< 10 KB) |
| **TFLite Tensor Arena** | 28,672 bytes (28.0 KB) | Internal SRAM (BSS) | Strictly < 35 KB |
| **Quantized Model (g_model)** | ~30 - 45 KB | Flash (mmap) | 0 bytes SRAM consumed |
| **Free Heap During Inference** | ~110 - 130 KB | Internal DRAM | Rock solid, zero leak |

---

## 🛠️ Arduino IDE Setup
1. **Board Package**: `esp32` by Espressif (version 2.0.14 or later).
2. **Library**: Open **Tools -> Manage Libraries...**, search for `TensorFlowLite_ESP32` by Tanaka Masayuki, and click **Install**.
   *(Alternatively: `EloquentTinyML` by Simone Salerno is supported by toggling `#define USE_ELOQUENT_TINYML 1` in the `.ino` file).*
3. **Board Settings**:
   - **Board**: `AI Thinker ESP32-CAM`
   - **CPU Frequency**: `240MHz (WiFi/BT)`
   - **Flash Frequency**: `80MHz`
   - **Partition Scheme**: `Huge APP (3MB No OTA/1MB SPIFFS)`
   - **Upload Speed**: `115200` or `921600`

---

## 🔌 Hardware Wiring for Flashing
| FTDI (USB-to-Serial) | ESP32-CAM | Notes |
|---|---|---|
| 5V | 5V | Use reliable 5V 1A+ supply |
| GND | GND | Common ground |
| TX | U0R (GPIO 3) | Cross RX/TX |
| RX | U0T (GPIO 1) | Cross TX/RX |
| GND | GPIO 0 | **Connect only during flashing!** Disconnect for normal boot |

---

## 🧠 Training & Exporting Custom Model
Run the Python pipeline locally or in Google Colab:
```bash
pip install tensorflow numpy
python train_quantize_model.py --epochs 20
```
This script will automatically calibrate, quantize to INT8, and overwrite `model_data.h`.
