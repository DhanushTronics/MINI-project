/**
 * ============================================================================
 * NodeMCU (ESP8266) - MONKEY GUARD base station
 *
 *   - 2 x HC-SR04 ultrasonic sensors (facing OPPOSITE sides) detect something
 *     approaching. A rise in the "closer than usual" reading sets the MOTION_A /
 *     MOTION_B line HIGH so the ESP32-CAM turns its servo to that side.
 *   - When the ESP32-CAM confirms a monkey it sets the ALERT line HIGH. Then:
 *       * DFPlayer Mini plays the repellent sound (looped until the alert ends)
 *       * SIM800L sends ONE SMS (then waits SMS_COOLDOWN_MS before another)
 *
 * Board : "NodeMCU 1.0 (ESP-12E Module)"   (ESP8266 Arduino core, no extra library)
 * Serial Monitor (115200) shows distances/events and accepts test commands:
 *     a = start audio     s = stop audio     g = send test SMS
 *     d = toggle distance printing
 * ============================================================================
 */
#include <Arduino.h>
#include <SoftwareSerial.h>

// ============================ USER SETTINGS =================================
const char*    PHONE_NUMBER    = "+91XXXXXXXXXX";   // <- put YOUR number, with country code
const char*    SMS_TEXT        = "ALERT: Monkey detected! Repellent sound and flasher started.";
const uint32_t SMS_COOLDOWN_MS = 5UL * 60UL * 1000UL;   // at most one SMS per 5 minutes
const uint32_t SMS_RETRY_MS    = 60UL * 1000UL;         // after a failed SMS, retry after 1 min
const uint8_t  DF_VOLUME       = 25;                    // DFPlayer volume 0..30

// Ultrasonic motion detection
const float    DETECT_RANGE_CM = 250.0f;   // only react to things closer than this
const float    DELTA_CM        = 30.0f;    // ...and at least this much closer than "normal"
const uint8_t  HITS_NEEDED     = 2;        // consecutive readings needed (filters noise)
const uint32_t MOTION_HOLD_MS  = 4000;     // keep MOTION line HIGH this long after last hit

// ================================ PINS ======================================
#define PIN_MOTION_A  D0   // out -> ESP32-CAM GPIO13
#define PIN_TRIG      D1   // out -> TRIG of BOTH HC-SR04 (wired together)
#define PIN_ECHO_A    D2   // in  <- ECHO of sensor A (through 1k/2k divider!)
#define PIN_MOTION_B  D3   // out -> ESP32-CAM GPIO15
#define PIN_DF_TX     D4   // out -> DFPlayer RX (through 1k resistor)
#define PIN_ECHO_B    D5   // in  <- ECHO of sensor B (through 1k/2k divider!)
#define PIN_GSM_RX    D6   // in  <- SIM800L TXD
#define PIN_GSM_TX    D7   // out -> SIM800L RXD (through 1k resistor)
#define PIN_ALERT_IN  D8   // in  <- ESP32-CAM GPIO2

#ifndef IRAM_ATTR
  #define IRAM_ATTR ICACHE_RAM_ATTR
#endif

SoftwareSerial gsm(PIN_GSM_RX, PIN_GSM_TX);
SoftwareSerial df(-1, PIN_DF_TX);          // transmit-only

bool debugDistances = false;

// ============================ ULTRASONIC ====================================
// Both sensors share one TRIG pulse; their echo pulses are timed by interrupts.
volatile uint32_t riseA = 0, riseB = 0, widthA = 0, widthB = 0;
volatile bool gotA = false, gotB = false;

void IRAM_ATTR isrA() {
  if (digitalRead(PIN_ECHO_A)) riseA = micros();
  else { widthA = micros() - riseA; gotA = true; }
}
void IRAM_ATTR isrB() {
  if (digitalRead(PIN_ECHO_B)) riseB = micros();
  else { widthB = micros() - riseB; gotB = true; }
}

struct Sensor {
  float    base  = -1.0f;   // slowly adapting "normal" distance
  uint8_t  hits  = 0;
  uint32_t until = 0;
  bool     armed = false;
};
Sensor sA, sB;

void updateSensor(Sensor& s, float d, uint32_t now) {
  if (s.base < 0) { s.base = d; return; }
  const bool closer = (d < s.base - DELTA_CM) && (d < DETECT_RANGE_CM);
  if (closer) {
    if (s.hits < 255) s.hits++;
    if (s.hits >= HITS_NEEDED) { s.until = now + MOTION_HOLD_MS; s.armed = true; }
  } else {
    s.hits = 0;
  }
  // A new static object is absorbed into the baseline after a while.
  s.base += (closer ? 0.01f : 0.05f) * (d - s.base);
}

bool motionActive(const Sensor& s, uint32_t now) {
  return s.armed && (int32_t)(s.until - now) > 0;
}

void handleSensors(uint32_t now) {
  static uint32_t tTrig = 0;
  static bool waiting = false;

  if (!waiting && (now - tTrig) >= 100) {
    gotA = gotB = false;
    digitalWrite(PIN_TRIG, HIGH);
    delayMicroseconds(10);
    digitalWrite(PIN_TRIG, LOW);
    tTrig = now;
    waiting = true;
  } else if (waiting && (now - tTrig) >= 35) {
    waiting = false;
    float dA = gotA ? (widthA / 58.0f) : 400.0f;   // no echo = nothing within 4 m
    float dB = gotB ? (widthB / 58.0f) : 400.0f;
    if (dA > 400.0f) dA = 400.0f;
    if (dB > 400.0f) dB = 400.0f;
    updateSensor(sA, dA, now);
    updateSensor(sB, dB, now);
    if (debugDistances) {
      Serial.printf("[US] A %.0f cm (normal %.0f)   B %.0f cm (normal %.0f)\n", dA, sA.base, dB, sB.base);
    }
  }

  static bool lastA = false, lastB = false;
  const bool mA = motionActive(sA, now);
  const bool mB = motionActive(sB, now);
  digitalWrite(PIN_MOTION_A, mA ? HIGH : LOW);
  digitalWrite(PIN_MOTION_B, mB ? HIGH : LOW);
  if (mA != lastA) { lastA = mA; Serial.printf("[MOTION] side A %s\n", mA ? "DETECTED" : "clear"); }
  if (mB != lastB) { lastB = mB; Serial.printf("[MOTION] side B %s\n", mB ? "DETECTED" : "clear"); }
}

// ============================== DFPLAYER ====================================
void dfSend(uint8_t cmd, uint16_t param) {
  uint8_t f[10] = {0x7E, 0xFF, 0x06, cmd, 0x00, (uint8_t)(param >> 8), (uint8_t)(param & 0xFF), 0, 0, 0xEF};
  uint16_t sum = 0;
  for (int i = 1; i <= 6; i++) sum += f[i];
  const uint16_t chk = (uint16_t)(0 - sum);
  f[7] = chk >> 8;
  f[8] = chk & 0xFF;
  df.write(f, 10);
  delay(100);
}
void dfInit()      { delay(1000); dfSend(0x09, 2); delay(200); dfSend(0x06, DF_VOLUME); }
void audioStart()  { Serial.println("[AUDIO] repellent ON");  dfSend(0x08, 1); }   // loop track 0001
void audioStop()   { Serial.println("[AUDIO] repellent OFF"); dfSend(0x16, 0); }

// ================================ GSM =======================================
String gsmLast;

bool gsmWait(const char* expect, uint32_t timeoutMs) {
  String buf;
  const uint32_t t0 = millis();
  while (millis() - t0 < timeoutMs) {
    while (gsm.available()) {
      buf += (char)gsm.read();
      if (buf.length() > 200) buf.remove(0, 100);
    }
    if (buf.indexOf(expect) >= 0) { gsmLast = buf; return true; }
    delay(5);   // also feeds the watchdog
  }
  gsmLast = buf;
  return false;
}

void gsmClear() { while (gsm.available()) gsm.read(); }

bool gsmCmd(const char* cmd, const char* expect, uint32_t timeoutMs) {
  gsmClear();
  gsm.println(cmd);
  return gsmWait(expect, timeoutMs);
}

bool gsmInit() {
  bool alive = false;
  for (int i = 0; i < 6 && !alive; i++) alive = gsmCmd("AT", "OK", 1000);
  if (!alive) {
    Serial.println("[GSM] no answer. Check SIM800L power (4.0 V, 2 A burst), TX/RX wires, ground.");
    return false;
  }
  gsmCmd("ATE0", "OK", 1000);          // no echo
  gsmCmd("AT+CMGF=1", "OK", 1000);     // SMS text mode
  Serial.print("[GSM] waiting for network");
  for (int i = 0; i < 15; i++) {       // up to ~30 s
    gsmCmd("AT+CREG?", "+CREG:", 1000);
    if (gsmLast.indexOf(",1") >= 0 || gsmLast.indexOf(",5") >= 0) {
      Serial.println(" registered.");
      gsmCmd("AT+CSQ", "+CSQ:", 1000);
      gsmLast.trim();
      Serial.println("[GSM] " + gsmLast);
      return true;
    }
    Serial.print(".");
    delay(1000);
  }
  Serial.println(" NOT registered (SIM inserted? 2G supported by your operator? PIN lock off?).");
  return false;
}

bool gsmSendSMS(const char* number, const char* text) {
  Serial.println("[SMS] sending ...");
  gsmClear();
  gsm.println("AT+CMGF=1");
  if (!gsmWait("OK", 2000)) { Serial.println("[SMS] module not answering"); return false; }
  gsmClear();
  gsm.print("AT+CMGS=\"");
  gsm.print(number);
  gsm.println("\"");
  if (!gsmWait(">", 5000)) {
    gsm.write(27);   // ESC = cancel
    Serial.println("[SMS] no '>' prompt");
    return false;
  }
  gsm.print(text);
  gsm.write(26);     // Ctrl+Z = send
  const bool ok = gsmWait("+CMGS:", 25000);
  Serial.println(ok ? "[SMS] SENT" : "[SMS] FAILED (no network / no credit / wrong number?)");
  return ok;
}

// ========================= ALERT FROM THE ESP32-CAM =========================
uint32_t lastSmsMs = 0;
bool     smsEver   = false;

void trySendSms(uint32_t now) {
  if (smsEver && (now - lastSmsMs) < SMS_COOLDOWN_MS) {
    Serial.println("[SMS] skipped (cooldown)");
    return;
  }
  if (gsmSendSMS(PHONE_NUMBER, SMS_TEXT)) {
    smsEver = true;
    lastSmsMs = millis();
  } else {
    smsEver = true;   // retry later, not on every alert
    lastSmsMs = millis() - SMS_COOLDOWN_MS + SMS_RETRY_MS;
  }
}

void handleAlert(uint32_t now) {
  static bool alertOn = false, lastRaw = false;
  static uint32_t changedAt = 0;
  const bool raw = digitalRead(PIN_ALERT_IN);
  if (raw != lastRaw) { lastRaw = raw; changedAt = now; }
  if ((now - changedAt) >= 100 && raw != alertOn) {   // stable for 100 ms
    alertOn = raw;
    if (alertOn) {
      Serial.println("[ALERT] monkey confirmed by camera!");
      audioStart();                  // sound first, SMS second (SMS blocks for a few seconds)
      trySendSms(millis());
    } else {
      Serial.println("[ALERT] camera cleared the alert");
      audioStop();
    }
  }
}

// ============================ SERIAL TEST KEYS ==============================
void handleSerial() {
  while (Serial.available()) {
    const char c = Serial.read();
    if (c == 'a') audioStart();
    else if (c == 's') audioStop();
    else if (c == 'g') gsmSendSMS(PHONE_NUMBER, "Test SMS from NodeMCU monkey guard");
    else if (c == 'd') { debugDistances = !debugDistances; Serial.println(debugDistances ? "[US] printing ON" : "[US] printing OFF"); }
  }
}

// ================================ ARDUINO ===================================
void setup() {
  // Safe output states immediately (the ESP32-CAM reads these lines).
  pinMode(PIN_MOTION_A, OUTPUT); digitalWrite(PIN_MOTION_A, LOW);
  pinMode(PIN_MOTION_B, OUTPUT); digitalWrite(PIN_MOTION_B, LOW);
  pinMode(PIN_TRIG, OUTPUT);     digitalWrite(PIN_TRIG, LOW);
  pinMode(PIN_ECHO_A, INPUT);
  pinMode(PIN_ECHO_B, INPUT);
  pinMode(PIN_ALERT_IN, INPUT);

  Serial.begin(115200);
  delay(300);
  Serial.println("\n=== NodeMCU MONKEY GUARD base station ===");

  gsm.begin(9600);
  df.begin(9600);

  dfInit();
  Serial.println("[AUDIO] DFPlayer initialised (volume " + String(DF_VOLUME) + ")");
  const bool gsmOk = gsmInit();
  Serial.println(gsmOk ? "[GSM] ready" : "[GSM] NOT ready - SMS will fail until fixed");

  attachInterrupt(digitalPinToInterrupt(PIN_ECHO_A), isrA, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_ECHO_B), isrB, CHANGE);

  Serial.println("Ready. Keys: a=audio on, s=audio off, g=test SMS, d=distances");
}

void loop() {
  const uint32_t now = millis();
  handleSensors(now);
  handleAlert(now);
  handleSerial();
}
