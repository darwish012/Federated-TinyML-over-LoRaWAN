// Calibration helper for method 1: reads the Arduino Uno's AREF pin (the Uno's internal ~1.1 V
// reference while uno_daq runs) with the ESP32's eFuse-calibrated ADC.
// Wiring: Uno AREF -> LilyGO GPIO36 (VP); Uno GND -> LilyGO GND.
// One attenuation per build (ATTEN): this core keeps the calibration of the first attenuation a
// pin was read with, so switching at runtime converts with the wrong curve.
#ifndef ATTEN
#define ATTEN ADC_6db        // ~150-1750 mV: 1.1 V sits mid-range
#endif
void setup() {
  Serial.begin(115200);
  analogSetPinAttenuation(36, ATTEN);
}
void loop() {
  uint32_t sum = 0, raw = 0;
  for (int i = 0; i < 1024; ++i) { sum += analogReadMilliVolts(36); raw += analogRead(36); }
  Serial.printf("AREF atten=%d %.1f mV (raw %.0f)\n", (int)ATTEN, sum / 1024.0, raw / 1024.0);
  delay(500);
}
