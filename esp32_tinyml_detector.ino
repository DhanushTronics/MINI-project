/**
 * ============================================================================
 * ESP32-CAM TinyML Image Classifier  (internal SRAM only, no PSRAM)
 * Board : AI-Thinker ESP32-CAM (OV2640)
 * Input : 96x96 grayscale, INT8 full-integer TFLite Micro model
 * Classes: 0 pigeon | 1 monkey | 2 human | 3 other
 *
 * Library : "TensorFlowLite_ESP32" by Tanaka Masayuki (Library Manager)
 *           (EloquentTinyML is intentionally NOT used: it feeds float input
 *            and cannot drive a full-INT8 model.)
 *
 * Arduino IDE settings:
 *   Board            : "ESP32 Wrover Module" (PSRAM: Disabled)
 *                      or "AI Thinker ESP32-CAM"
 *   Partition Scheme : "Huge APP (3MB No OTA/1MB SPIFFS)"
 *   Serial Monitor   : 115200 baud
 * ============================================================================
 */

#include "esp_camera.h"
#include "WiFi.h"

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
// Some library versions ship version.h, others don't. Use it if present,
// otherwise fall back to the schema version every current .tflite uses (3).
#if defined(__has_include)
  #if __has_include("tensorflow/lite/version.h")
    #include "tensorflow/lite/version.h"
  #endif
#endif
#ifndef TFLITE_SCHEMA_VERSION
  #define TFLITE_SCHEMA_VERSION 3
#endif

#include "model_data.h"

// Set to 1 ONLY as a last resort for brownout resets. A proper 5 V / 2 A
// supply (and a short USB cable) is the real fix.
#define DISABLE_BROWNOUT_DETECTOR 0
#if DISABLE_BROWNOUT_DETECTOR
  #include "soc/soc.h"
  #include "soc/rtc_cntl_reg.h"
#endif

// ----------------------------------------------------------------------------
// AI-Thinker pin map
// ----------------------------------------------------------------------------
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#define FLASH_LED_PIN      4   // white flash LED (active HIGH) - kept OFF
#define STATUS_LED_PIN    33   // small red LED   (active LOW)

// ----------------------------------------------------------------------------
// Model / class configuration
// ----------------------------------------------------------------------------
constexpr int NUM_CLASSES   = 4;
constexpr int INPUT_WIDTH   = 96;
constexpr int INPUT_HEIGHT  = 96;
constexpr int INPUT_PIXELS  = INPUT_WIDTH * INPUT_HEIGHT;
const char* const CLASS_NAMES[NUM_CLASSES] = {"pigeon", "monkey", "human", "other"};

constexpr float CONFIDENCE_THRESHOLD = 0.60f;   // below this => "(low confidence)"

// Tensor arena: static, internal SRAM, must stay < 35 KB.
// The bundled CNN peaks at ~18.4 KB of activations (+ a few KB of overhead).
// If AllocateTensors() fails with your own model, raise to 34 KB at most.
constexpr int kTensorArenaSize = 28 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

static tflite::MicroErrorReporter micro_error_reporter;
static tflite::ErrorReporter*     error_reporter = &micro_error_reporter;
static const tflite::Model*       model          = nullptr;
static tflite::MicroInterpreter*  interpreter    = nullptr;
static TfLiteTensor*              model_input    = nullptr;
static TfLiteTensor*              model_output   = nullptr;

static float   g_in_mult = 1.0f;   // pixel -> quantized scaling factor
static int32_t g_in_zp   = 0;

// ----------------------------------------------------------------------------
// Camera
// ----------------------------------------------------------------------------
esp_err_t initCamera() {
  camera_config_t config = {};     // zero-init: avoids garbage in newer driver fields
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk  = XCLK_GPIO_NUM;
  config.pin_pclk  = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href  = HREF_GPIO_NUM;
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  config.pin_sccb_sda = SIOD_GPIO_NUM;     // ESP32 Arduino core 3.x
  config.pin_sccb_scl = SIOC_GPIO_NUM;
#else
  config.pin_sscb_sda = SIOD_GPIO_NUM;     // ESP32 Arduino core 2.x
  config.pin_sscb_scl = SIOC_GPIO_NUM;
#endif
  config.pin_pwdn  = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;

  config.frame_size   = FRAMESIZE_96X96;       // 9,216 bytes in grayscale
  config.pixel_format = PIXFORMAT_GRAYSCALE;
  config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location  = CAMERA_FB_IN_DRAM;     // never touch PSRAM
  config.fb_count     = 1;
  config.jpeg_quality = 12;                    // unused in grayscale

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[CAMERA] init failed: 0x%x\n", err);
    return err;
  }

  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    s->set_vflip(s, 1);        // set to 0 if your image is upside down
    s->set_hmirror(s, 0);
    s->set_brightness(s, 1);
    s->set_contrast(s, 1);
  }
  Serial.println("[CAMERA] OV2640 ready: 96x96 grayscale, 1 frame buffer in DRAM");
  return ESP_OK;
}

// ----------------------------------------------------------------------------
// TFLite Micro
// ----------------------------------------------------------------------------
static bool modelLooksValid() {
  if (g_model_len < 1024) return false;                 // placeholder stub
  return memcmp(g_model + 4, "TFL3", 4) == 0;           // flatbuffer identifier
}

bool initTinyML() {
  if (!modelLooksValid()) {
    Serial.println("[TFLITE] model_data.h still holds the PLACEHOLDER model.");
    Serial.println("         Train one with train_quantize_model.py and replace model_data.h.");
    return false;
  }

  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("[TFLITE] schema version %d != runtime %d\n",
                  (int)model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[TFLITE] AllocateTensors() failed (arena too small, or an op is");
    Serial.println("         unsupported by this library version - see messages above).");
    return false;
  }

  model_input  = interpreter->input(0);
  model_output = interpreter->output(0);

  // Validate the model matches what this sketch feeds it.
  if (model_input->type != kTfLiteInt8 ||
      model_input->dims->size != 4 ||
      model_input->dims->data[1] != INPUT_HEIGHT ||
      model_input->dims->data[2] != INPUT_WIDTH  ||
      model_input->dims->data[3] != 1) {
    Serial.println("[TFLITE] Expected int8 input [1,96,96,1]. Re-export with the provided script.");
    return false;
  }
  if (model_output->type != kTfLiteInt8 &&
      model_output->type != kTfLiteUInt8 &&
      model_output->type != kTfLiteFloat32) {
    Serial.println("[TFLITE] Unsupported output tensor type.");
    return false;
  }
  int out_n = model_output->dims->data[model_output->dims->size - 1];
  if (out_n != NUM_CLASSES) {
    Serial.printf("[TFLITE] Model has %d outputs, expected %d.\n", out_n, NUM_CLASSES);
    return false;
  }

  // Model was trained on pixel/255. real = pixel/255 ; q = real/scale + zp
  g_in_mult = 1.0f / (255.0f * model_input->params.scale);
  g_in_zp   = model_input->params.zero_point;

  Serial.printf("[TFLITE] Arena used: %u of %d bytes\n",
                (unsigned)interpreter->arena_used_bytes(), kTensorArenaSize);
  Serial.printf("[TFLITE] Input  scale=%.6f zp=%d | Output scale=%.6f zp=%d\n",
                model_input->params.scale, (int)model_input->params.zero_point,
                model_output->params.scale, (int)model_output->params.zero_point);
  return true;
}

// ----------------------------------------------------------------------------
// Arduino
// ----------------------------------------------------------------------------
void haltBlink(int period_ms) {   // never returns (infinite loop)
  while (true) {
    digitalWrite(STATUS_LED_PIN, LOW);  delay(period_ms);
    digitalWrite(STATUS_LED_PIN, HIGH); delay(period_ms);
  }
}

void setup() {
#if DISABLE_BROWNOUT_DETECTOR
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
#endif
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== ESP32-CAM TinyML classifier (no PSRAM) ===");

  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, HIGH);
  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);

  // Radios off: lower current spikes (fewer brownouts) and more free heap.
  WiFi.mode(WIFI_OFF);
  btStop();

  Serial.printf("[SYSTEM] Free heap at boot: %u bytes\n", (unsigned)ESP.getFreeHeap());

  if (initCamera() != ESP_OK) {
    Serial.println("[FATAL] Camera init failed. Check ribbon cable, power and board settings.");
    haltBlink(100);
  }
  if (!initTinyML()) {
    Serial.println("[FATAL] TinyML init failed. Halting (camera itself is OK).");
    haltBlink(500);
  }

  Serial.printf("[SYSTEM] Ready. Free heap: %u bytes\n\n", (unsigned)ESP.getFreeHeap());
}

void loop() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[CAMERA] Frame capture failed.");
    delay(100);
    return;
  }
  if (fb->width != INPUT_WIDTH || fb->height != INPUT_HEIGHT || fb->len < (size_t)INPUT_PIXELS) {
    Serial.printf("[CAMERA] Unexpected frame %dx%d len=%u\n",
                  (int)fb->width, (int)fb->height, (unsigned)fb->len);
    esp_camera_fb_return(fb);
    delay(500);
    return;
  }

  digitalWrite(STATUS_LED_PIN, LOW);
  const unsigned long t0 = millis();

  // Quantize camera pixels straight into the input tensor (no extra buffer).
  int8_t* dst = model_input->data.int8;
  for (int i = 0; i < INPUT_PIXELS; i++) {
    int32_t q = (int32_t)lroundf(fb->buf[i] * g_in_mult) + g_in_zp;
    if (q < -128) q = -128;
    if (q > 127)  q = 127;
    dst[i] = (int8_t)q;
  }
  esp_camera_fb_return(fb);                      // release the frame early

  float conf[NUM_CLASSES] = {0};
  int   best = 0;

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("[TFLITE] Invoke() failed.");
    digitalWrite(STATUS_LED_PIN, HIGH);
    delay(500);
    return;
  }

  const float   o_scale = model_output->params.scale;
  const int32_t o_zp    = model_output->params.zero_point;
  for (int i = 0; i < NUM_CLASSES; i++) {
    float p;
    switch (model_output->type) {
      case kTfLiteInt8:    p = (model_output->data.int8[i]  - o_zp) * o_scale; break;
      case kTfLiteUInt8:   p = (model_output->data.uint8[i] - o_zp) * o_scale; break;
      default:             p = model_output->data.f[i];                        break;
    }
    if (p < 0.0f) p = 0.0f;
    if (p > 1.0f) p = 1.0f;
    conf[i] = p;
    if (p > conf[best]) best = i;
  }

  const unsigned long dt = millis() - t0;
  digitalWrite(STATUS_LED_PIN, HIGH);

  Serial.printf(">>> %-7s %5.1f%%  (%lu ms)%s\n",
                CLASS_NAMES[best], conf[best] * 100.0f, dt,
                conf[best] < CONFIDENCE_THRESHOLD ? "  (low confidence)" : "");
  Serial.printf("    pigeon %.1f%% | monkey %.1f%% | human %.1f%% | other %.1f%% | heap %u\n",
                conf[0] * 100.0f, conf[1] * 100.0f, conf[2] * 100.0f, conf[3] * 100.0f,
                (unsigned)ESP.getFreeHeap());

  delay(100);
}
