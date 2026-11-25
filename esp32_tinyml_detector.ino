/**
 * ============================================================================
 * ESP32-CAM MONKEY GUARD  (TinyML classifier + pan servo + flasher + NodeMCU link)
 * Board : AI-Thinker ESP32-CAM (OV2640)
 * Input : 96x96 grayscale, INT8 full-integer TFLite Micro model
 * Classes: defined in model_data.h (written by train_quantize_model.py)
 *
 * What it does:
 *   SWEEP        servo pans the camera left <-> right, classifying every frame
 *   INVESTIGATE  NodeMCU reports motion on side A or B -> camera turns to that side
 *   ALERT        monkey confirmed -> servo holds, flasher strobes, ALERT line goes
 *                HIGH so the NodeMCU plays the repellent sound and sends the SMS
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
#include "driver/ledc.h"

// ---- Debug / camera options -------------------------------------------------
#define RUN_SELFTEST       0   // 1 = at boot feed 4 known images (needs selftest_data.h)
#define DEBUG_CAMERA_VIEW  0   // 1 = print pixel stats every frame + ASCII picture every 10th
#define CAM_VFLIP          0   // set to 1 if the ASCII picture is upside down
#define CAM_HMIRROR        0   // set to 1 if the ASCII picture is mirrored

#if RUN_SELFTEST && defined(__has_include)
  #if __has_include("selftest_data.h")
    #include "selftest_data.h"
    #define HAVE_SELFTEST 1
  #endif
#endif
#ifndef HAVE_SELFTEST
  #define HAVE_SELFTEST 0
#endif

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
// Guard hardware  (see WIRING.md).  Remove the SD card: these pins are shared with it.
// ----------------------------------------------------------------------------
#define SERVO_PIN          14   // pan servo signal (orange wire)
#define EXT_FLASHER_PIN    12   // external LED flasher via NPN/MOSFET (10k base->GND pull-down!)
#define MOTION_A_PIN       13   // input  <- NodeMCU D0  (HIGH = motion on side A)
#define MOTION_B_PIN       15   // input  <- NodeMCU D3  (HIGH = motion on side B)
#define ALERT_PIN           2   // output -> NodeMCU D8  (HIGH = monkey confirmed)
#define USE_ONBOARD_FLASH   1   // also strobe the on-board white LED (GPIO 4)

// ----------------------------------------------------------------------------
// Behaviour tuning
// ----------------------------------------------------------------------------
constexpr int      SERVO_MIN_DEG   = 20;      // sweep range
constexpr int      SERVO_MAX_DEG   = 160;
constexpr int      SERVO_STEP_DEG  = 20;      // sweep step (camera view is ~60 deg wide)
constexpr int      SERVO_ANGLE_A   = 20;      // direction watched by ultrasonic sensor A
constexpr int      SERVO_ANGLE_B   = 160;     // direction watched by ultrasonic sensor B
constexpr int      SERVO_US_MIN    = 600;     // pulse width at 0 deg   (SG90: ~500-600)
constexpr int      SERVO_US_MAX    = 2400;    // pulse width at 180 deg (SG90: ~2400)

constexpr float    MONKEY_CONFIDENCE = 0.80f; // a frame counts as "monkey" above this
constexpr uint8_t  CONFIRM_FRAMES    = 2;     // consecutive monkey frames needed for an alert
constexpr uint32_t SWEEP_DWELL_MS    = 1500;  // time spent looking at each sweep position
constexpr uint32_t INVESTIGATE_MS    = 8000;  // stay on a motion side this long after last motion
constexpr uint32_t ALERT_CLEAR_MS    = 8000;  // alert ends after this long without a monkey
constexpr uint32_t BOOT_GRACE_MS     = 5000;  // ignore motion lines right after power-up
constexpr uint32_t MOTION_DEBOUNCE_MS = 150;  // motion line must stay HIGH this long

// ----------------------------------------------------------------------------
// Model / class configuration
// ----------------------------------------------------------------------------
constexpr int MAX_CLASSES   = 8;                 // upper bound for the buffers below
#define NUM_CLASSES  (g_num_classes)             // class count/names come from model_data.h
#define CLASS_NAMES  (g_class_names)
constexpr int INPUT_WIDTH   = 96;
constexpr int INPUT_HEIGHT  = 96;
constexpr int INPUT_PIXELS  = INPUT_WIDTH * INPUT_HEIGHT;

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
    s->set_vflip(s, CAM_VFLIP);
    s->set_hmirror(s, CAM_HMIRROR);
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

  if (g_num_classes < 2 || g_num_classes > MAX_CLASSES) {
    Serial.println("[TFLITE] g_num_classes in model_data.h must be 2..8.");
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

// Quantize 96x96 8-bit pixels into the model input tensor (same maths as in training).
static void quantizeFrame(const uint8_t* src) {
  int8_t* dst = model_input->data.int8;
  for (int i = 0; i < INPUT_PIXELS; i++) {
    int32_t q = (int32_t)lroundf(src[i] * g_in_mult) + g_in_zp;
    if (q < -128) q = -128;
    if (q > 127)  q = 127;
    dst[i] = (int8_t)q;
  }
}

// Read the output tensor as probabilities. Returns index of the best class.
static int readOutput(float* conf) {
  const float   o_scale = model_output->params.scale;
  const int32_t o_zp    = model_output->params.zero_point;
  int best = 0;
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
  return best;
}

// Feed known images and compare with what Python computed. This separates
// "model/library problem" from "camera problem".
static void runSelfTest() {
#if RUN_SELFTEST && HAVE_SELFTEST
  if (g_model_len != selftest_model_len || g_num_classes != 2) {
    Serial.println("[SELFTEST] skipped (model_data.h is not the model this test was made for)");
    return;
  }
  Serial.println("[SELFTEST] 4 known images (expected = Python result for the same image):");
  bool match = true;
  float lo = 1.0f, hi = 0.0f;
  for (int k = 0; k < SELFTEST_COUNT; k++) {
    quantizeFrame(selftest_img[k]);
    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.println("[SELFTEST] Invoke() failed.");
      return;
    }
    float c[MAX_CLASSES] = {0};
    readOutput(c);
    Serial.printf("   %-6s  device: monkey %3.0f%% other %3.0f%%  | expected: monkey %3.0f%% other %3.0f%%\n",
                  selftest_label[k], c[0] * 100, c[1] * 100,
                  selftest_expected[k][0] * 100, selftest_expected[k][1] * 100);
    if (fabsf(c[0] - selftest_expected[k][0]) > 0.10f) match = false;
    if (c[0] < lo) lo = c[0];
    if (c[0] > hi) hi = c[0];
  }
  if (match) {
    Serial.println("[SELFTEST] PASS: model + library work on the ESP32. If live results are");
    Serial.println("           still wrong, the problem is the CAMERA IMAGE (see ASCII picture).");
  } else if (hi - lo < 0.05f) {
    Serial.println("[SELFTEST] FAIL: output does not react to the input at all.");
    Serial.println("           => library/kernel problem (try Chirale_TensorFlowLite).");
  } else {
    Serial.println("[SELFTEST] FAIL: output reacts but differs from Python.");
    Serial.println("           => library version mismatch (try Chirale_TensorFlowLite).");
  }
  Serial.println();
#endif
}

// Print what the camera sees: pixel statistics, and a small ASCII picture.
static void debugCameraView(const camera_fb_t* fb, bool picture) {
#if DEBUG_CAMERA_VIEW
  uint8_t mn = 255, mx = 0;
  uint32_t sum = 0;
  for (int i = 0; i < INPUT_PIXELS; i++) {
    uint8_t v = fb->buf[i];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    sum += v;
  }
  Serial.printf("[FRAME] pixels min %u max %u mean %u%s\n", mn, mx, (unsigned)(sum / INPUT_PIXELS),
                (mx - mn) < 25 ? "  <-- almost uniform: dark/saturated/blocked lens?" : "");
  if (picture) {
    static const char shades[] = " .:-=+*#%@";
    for (int y = 0; y < INPUT_HEIGHT; y += 4) {
      for (int x = 0; x < INPUT_WIDTH; x += 2) {
        Serial.print(shades[fb->buf[y * INPUT_WIDTH + x] * 9 / 255]);
      }
      Serial.println();
    }
  }
#endif
}

// ----------------------------------------------------------------------------
// Arduino
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Servo: raw LEDC (no library). Uses timer 2 / channel 4, the camera clock uses
// timer 0 / channel 0, so they cannot collide.
// ----------------------------------------------------------------------------
static int servoAngle = 90;

static void servoInit() {
  ledc_timer_config_t t = {};
  t.speed_mode      = LEDC_LOW_SPEED_MODE;
  t.duty_resolution = LEDC_TIMER_14_BIT;
  t.timer_num       = LEDC_TIMER_2;
  t.freq_hz         = 50;
  t.clk_cfg         = LEDC_AUTO_CLK;
  ledc_timer_config(&t);

  ledc_channel_config_t c = {};
  c.gpio_num   = SERVO_PIN;
  c.speed_mode = LEDC_LOW_SPEED_MODE;
  c.channel    = LEDC_CHANNEL_4;
  c.intr_type  = LEDC_INTR_DISABLE;
  c.timer_sel  = LEDC_TIMER_2;
  c.duty       = 0;
  c.hpoint     = 0;
  ledc_channel_config(&c);
}

static void servoWrite(int deg) {
  deg = constrain(deg, 0, 180);
  const uint32_t us   = SERVO_US_MIN + (uint32_t)(SERVO_US_MAX - SERVO_US_MIN) * deg / 180;
  const uint32_t duty = (uint32_t)(((uint64_t)us * 16384ULL) / 20000ULL);   // 14-bit, 20 ms period
  ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_4, duty);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_4);
}

// ----------------------------------------------------------------------------
// Flasher: runs in its own task so the strobe stays fast while inference runs.
// Pattern: 8 quick flashes, then a dark window (g_dark) in which loop() classifies,
// because a lit LED next to the lens would wash out the picture.
// ----------------------------------------------------------------------------
static volatile bool g_strobe = false;
static volatile bool g_dark   = true;

static void setFlasher(bool on) {
  digitalWrite(EXT_FLASHER_PIN, on ? HIGH : LOW);
#if USE_ONBOARD_FLASH
  digitalWrite(FLASH_LED_PIN, on ? HIGH : LOW);
#endif
}

static void strobeTask(void*) {
  for (;;) {
    if (!g_strobe) {
      setFlasher(false);
      g_dark = true;
      vTaskDelay(pdMS_TO_TICKS(40));
      continue;
    }
    g_dark = false;
    for (int i = 0; i < 8 && g_strobe; i++) {
      setFlasher(true);  vTaskDelay(pdMS_TO_TICKS(70));
      setFlasher(false); vTaskDelay(pdMS_TO_TICKS(70));
    }
    setFlasher(false);
    vTaskDelay(pdMS_TO_TICKS(200));   // let the camera exposure settle
    g_dark = true;
    vTaskDelay(pdMS_TO_TICKS(700));   // dark window: frames are classified now
  }
}

// ----------------------------------------------------------------------------
// Guard state machine
// ----------------------------------------------------------------------------
enum Mode { MODE_SWEEP, MODE_INVESTIGATE, MODE_ALERT };
static Mode     mode             = MODE_SWEEP;
static int      sweepDir         = +1;
static uint32_t settleUntil      = 0;   // camera is still moving until then
static uint32_t lastStepMs       = 0;
static uint32_t investigateUntil = 0;
static uint32_t lastMonkeyMs     = 0;
static uint8_t  monkeyHits       = 0;
static int      monkeyIdx        = -1;

static const char* modeName(int m) {   // int, not Mode: Arduino auto-prototypes sit above the enum
  return m == MODE_SWEEP ? "SWEEP" : (m == MODE_INVESTIGATE ? "LOOK " : "ALERT");
}

static void moveServo(int deg) {
  deg = constrain(deg, 0, 180);
  const int delta = abs(deg - servoAngle);
  servoWrite(deg);
  servoAngle = deg;
  settleUntil = millis() + 150 + (uint32_t)delta * 8;   // wait for the picture to calm down
}

static bool readMotion(int pin, uint32_t& highSince) {
  if (digitalRead(pin)) {
    if (highSince == 0) highSince = millis() | 1;
    return (millis() - highSince) >= MOTION_DEBOUNCE_MS;
  }
  highSince = 0;
  return false;
}

static void enterAlert(uint32_t now) {
  mode = MODE_ALERT;
  lastMonkeyMs = now;
  digitalWrite(ALERT_PIN, HIGH);
  g_strobe = true;
  Serial.println("[ALERT] MONKEY CONFIRMED -> NodeMCU alerted, flasher ON");
}

static void leaveAlert(uint32_t now) {
  digitalWrite(ALERT_PIN, LOW);
  g_strobe = false;
  mode = MODE_SWEEP;
  lastStepMs = now;
  monkeyHits = 0;
  Serial.println("[ALERT] cleared, resuming sweep");
}

// ----------------------------------------------------------------------------
// Arduino
// ----------------------------------------------------------------------------
void setup() {
#if DISABLE_BROWNOUT_DETECTOR
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
#endif
  // Outputs first, in a safe (off) state.
  pinMode(ALERT_PIN, OUTPUT);        digitalWrite(ALERT_PIN, LOW);
  pinMode(EXT_FLASHER_PIN, OUTPUT);  digitalWrite(EXT_FLASHER_PIN, LOW);
  pinMode(FLASH_LED_PIN, OUTPUT);    digitalWrite(FLASH_LED_PIN, LOW);
  pinMode(STATUS_LED_PIN, OUTPUT);   digitalWrite(STATUS_LED_PIN, HIGH);
  pinMode(MOTION_A_PIN, INPUT_PULLDOWN);
  pinMode(MOTION_B_PIN, INPUT_PULLDOWN);

  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== ESP32-CAM MONKEY GUARD ===");

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

  for (int i = 0; i < NUM_CLASSES; i++) {
    if (strcasecmp(CLASS_NAMES[i], "monkey") == 0) monkeyIdx = i;
  }
  if (monkeyIdx < 0) {
    Serial.println("[FATAL] The model has no class called 'monkey' (see g_class_names in model_data.h).");
    haltBlink(300);
  }

  servoInit();                         // after the camera: it owns LEDC timer 0
  servoWrite(servoAngle);
  settleUntil = millis() + 800;
  xTaskCreate(strobeTask, "strobe", 2048, nullptr, 1, nullptr);

  // Let auto-exposure / auto-gain settle: throw away the first frames.
  for (int i = 0; i < 8; i++) {
    camera_fb_t* w = esp_camera_fb_get();
    if (w) esp_camera_fb_return(w);
    delay(50);
  }

  runSelfTest();

  Serial.printf("[SYSTEM] Ready. Free heap: %u bytes. Monkey class index = %d\n\n",
                (unsigned)ESP.getFreeHeap(), monkeyIdx);
}

void loop() {
  static uint32_t frameCount = 0;
  static uint32_t aSince = 0, bSince = 0;
  uint32_t now = millis();

  // ---- 1. motion lines from the NodeMCU (debounced, ignored right after boot) ----
  const bool graceOver = now > BOOT_GRACE_MS;
  const bool motionA = graceOver && readMotion(MOTION_A_PIN, aSince);
  const bool motionB = graceOver && readMotion(MOTION_B_PIN, bSince);
  const bool motion  = motionA || motionB;
  const int  motionAngle = motionA ? SERVO_ANGLE_A : SERVO_ANGLE_B;

  // ---- 2. decide where the camera should point ----
  if (mode == MODE_ALERT) {
    // keep looking at the monkey; only turn away if it vanished and motion shows up elsewhere
    if (motion && (now - lastMonkeyMs) > 2500 && abs(servoAngle - motionAngle) > 2) {
      moveServo(motionAngle);
    }
  } else {
    if (motion) {
      mode = MODE_INVESTIGATE;
      investigateUntil = now + INVESTIGATE_MS;
      if (abs(servoAngle - motionAngle) > 2) {
        moveServo(motionAngle);
        monkeyHits = 0;
      }
    } else if (mode == MODE_INVESTIGATE && (int32_t)(now - investigateUntil) > 0) {
      mode = MODE_SWEEP;
      lastStepMs = now;
    }

    if (mode == MODE_SWEEP && (now - lastStepMs) >= SWEEP_DWELL_MS) {
      int next = servoAngle + sweepDir * SERVO_STEP_DEG;
      if (next >= SERVO_MAX_DEG) { next = SERVO_MAX_DEG; sweepDir = -1; }
      if (next <= SERVO_MIN_DEG) { next = SERVO_MIN_DEG; sweepDir = +1; }
      moveServo(next);
      lastStepMs = now;
      monkeyHits = 0;                  // new view => start confirming again
    }
  }

  // ---- 3. no useful picture while the camera moves or the flasher is lit ----
  if (millis() < settleUntil) { delay(10); return; }
  if (mode == MODE_ALERT && !g_dark) { delay(10); return; }

  // ---- 4. grab + classify a frame ----
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
  debugCameraView(fb, (frameCount++ % 10) == 0);

  digitalWrite(STATUS_LED_PIN, LOW);
  const unsigned long t0 = millis();
  quantizeFrame(fb->buf);
  esp_camera_fb_return(fb);

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("[TFLITE] Invoke() failed.");
    digitalWrite(STATUS_LED_PIN, HIGH);
    delay(500);
    return;
  }
  float conf[MAX_CLASSES] = {0};
  const int best = readOutput(conf);
  const unsigned long dt = millis() - t0;
  digitalWrite(STATUS_LED_PIN, HIGH);

  // ---- 5. confirm / clear the alert ----
  const bool monkey = (best == monkeyIdx) && (conf[monkeyIdx] >= MONKEY_CONFIDENCE);
  now = millis();
  if (monkey) {
    lastMonkeyMs = now;
    if (monkeyHits < 255) monkeyHits++;
    if (mode != MODE_ALERT) lastStepMs = now;      // do not sweep away while confirming
  } else {
    monkeyHits = 0;
  }

  if (mode != MODE_ALERT && monkeyHits >= CONFIRM_FRAMES) {
    enterAlert(now);
  } else if (mode == MODE_ALERT && (now - lastMonkeyMs) > ALERT_CLEAR_MS) {
    leaveAlert(now);
  }

  Serial.printf("[%s %3ddeg] %-7s %5.1f%% (%lu ms)%s%s\n",
                modeName(mode), servoAngle, CLASS_NAMES[best], conf[best] * 100.0f, dt,
                monkey ? "  <-- monkey frame" : "", motion ? "  [motion line]" : "");
}
