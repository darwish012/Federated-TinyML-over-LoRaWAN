/*
 * Method 1 power DAQ (Arduino Uno): LilyGO current through a 1-ohm low-side shunt, plus the
 * LilyGO's phase markers, on ONE clock (the Uno's millis()).
 *
 * Wiring (the LilyGO's USB must stay UNPLUGGED: its ground would bypass the shunt):
 *   Uno 5V  -> LilyGO 5V pin
 *   LilyGO GND -> shunt leg A ; shunt leg B -> Uno GND
 *   Uno A0 -> shunt leg A ; Uno A1 -> shunt leg B   (both directly at the resistor leads)
 *   LilyGO GPIO13 (marker UART, 9600 8N1) -> Uno D8
 * Differential (Kelvin) reading: the current is (A0 - A1) / R, so resistance between shunt leg B
 * and the Uno's ground (wires, breadboard contacts) is not measured as shunt. With A0 alone the
 * reading changed x1.33 and x2.3 between re-wirings (2026-10-20).
 *
 * Output on USB serial (115200), one line per event:
 *   C <ms> <a0> <a1>      current sample: A0 and A1 counts, 0..1023 of the INTERNAL ~1.1 V reference
 *   M <ms> <marker line>  a LilyGO line (PHASE/ROUND/BATT/SCORE...), <ms> = arrival of its
 *                         first character (the marker is printed at the transition)
 *   V <ms> <adc>          supply check every 60 s: the internal 1.1 V bandgap read against
 *                         AVcc, so Vcc = Vref * 1023 / adc (Vref = the measured internal reference)
 * Conversion to mA (and the measured reference voltage) is done on the PC: capture_method1.py.
 */
#include <SoftwareSerial.h>

SoftwareSerial markers(8, 9);       // RX D8 (from LilyGO GPIO13), TX D9 unused
const unsigned long kPeriodUs = 5000; // 200 Hz current sampling

char line[96];
uint8_t lineLen = 0;
unsigned long lineStart = 0;
unsigned long nextSample = 0, nextVcc = 0;

// Bandgap (~1.1 V) against AVcc: ADMUX = REFS0 (AVcc) | MUX 1110. Then back to INTERNAL for A0.
// Takes ~15 ms, once a minute; the current sample due meanwhile is simply taken late.
void measureVcc() {
  ADMUX = _BV(REFS0) | 0x0E;
  delay(5);
  unsigned int sum = 0;
  for (int i = 0; i < 8; ++i) { ADCSRA |= _BV(ADSC); while (ADCSRA & _BV(ADSC)) {} sum += ADC; }
  Serial.print('V'); Serial.print(' '); Serial.print(millis()); Serial.print(' '); Serial.println(sum / 8.0, 2);
  analogReference(INTERNAL);
  analogRead(A0); delay(5); analogRead(A0);   // let AREF settle back to the internal reference
}

void setup() {
  Serial.begin(115200);
  markers.begin(9600);
  analogReference(INTERNAL);        // ~1.1 V full scale: ~1.07 mA per count on 1 ohm
  analogRead(A0);                   // first conversion after a reference change is discarded
  delay(10);
  Serial.println(F("# uno_daq 200Hz A0 internal-ref; lines: C <ms> <adc> | M <ms> <marker> | V <ms> <bandgap adc>"));
  measureVcc();
  nextSample = micros();
  nextVcc = millis() + 60000UL;
}

void loop() {
  if ((long)(millis() - nextVcc) >= 0) { nextVcc += 60000UL; measureVcc(); nextSample = micros(); }
  if ((long)(micros() - nextSample) >= 0) {
    nextSample += kPeriodUs;
    unsigned long ms = millis();
    int a0 = analogRead(A0);        // shunt leg A (LilyGO GND side)
    int a1 = analogRead(A1);        // shunt leg B (Uno GND side): current = (a0 - a1) / R
    Serial.print('C'); Serial.print(' '); Serial.print(ms); Serial.print(' '); Serial.print(a0);
    Serial.print(' '); Serial.println(a1);
  }
  while (markers.available()) {
    char c = markers.read();
    if (c == '\r') continue;
    if (c == '\n') {
      if (lineLen) {
        line[lineLen] = 0;
        Serial.print('M'); Serial.print(' '); Serial.print(lineStart); Serial.print(' '); Serial.println(line);
      }
      lineLen = 0;
      continue;
    }
    if (!lineLen) lineStart = millis() - 1;   // the start bit arrived ~1 ms before the byte completed
    if (lineLen < sizeof(line) - 1) line[lineLen++] = c;
  }
}
