#pragma once
// Offline only: no LMIC/radio. LittleFS format never touches tinyflmc NVS.
#include <Arduino.h>
#include <LittleFS.h>
#include "../common/tinyfl_core/tf_common.cpp"
#include "../common/tinyfl_nvs_keys.h"
bool provisionReady=false, provisionDone=false;
uint32_t provisionStart=0, provisionHello=0;
char provisionLine[180]={}; size_t provisionUsed=0; bool provisionOverflow=false;
uint8_t provisionEui[8]={};
void setup() {
  Serial.begin(115200); delay(500); uint64_t mac=ESP.getEfuseMac();
  for(int i=0;i<6;i++) provisionEui[i+2]=mac>>(8*(5-i));
  // arduino-esp32 3.x: format() before any begin() has no partition label (NULL) and fails;
  // begin() records the default "spiffs" label, which format() then uses.
  bool fsOk=LittleFS.begin(true); if(fsOk) LittleFS.end();
  if(!fsOk || !LittleFS.format() || !LittleFS.begin(false)) { Serial.println("TFPROV ERR littlefs_failed"); return; }
  LittleFS.end(); provisionReady=true; provisionStart=millis(); provisionHello=millis()-1000;
}
void loop() {
  if(!provisionReady || provisionDone) { delay(20); return; }
  if(millis()-provisionStart>=120000) { Serial.println("TFPROV ERR timeout"); provisionDone=true; return; }
  if(millis()-provisionHello>=1000) {
    Serial.print("TFPROV HELLO 42 ");
    for(int i=0;i<8;i++) Serial.printf("%02X",provisionEui[i]); Serial.println(); provisionHello=millis();
  }
  while(Serial.available()) {
    char c=Serial.read();
    if(c!='\n') {
      if(provisionUsed+1<sizeof(provisionLine)) provisionLine[provisionUsed++]=c;
      else provisionOverflow=true; continue;
    }
    provisionLine[provisionUsed]=0; tinyfl::v4::Keys keys{},readback{};
    if(provisionOverflow || !tinyfl::v4::parseKeys(provisionLine,keys)) Serial.println("TFPROV ERR bad_line");
    else if(!tinyfl::v4::storeKeys(keys,readback)) Serial.println("TFPROV ERR nvs_readback_failed");
    else { Serial.printf("TFPROV OK %08lX\n",(unsigned long)readback.crc()); provisionDone=true; }
    memset(provisionLine,0,sizeof(provisionLine)); memset(&keys,0,sizeof(keys)); memset(&readback,0,sizeof(readback));
    provisionUsed=0; provisionOverflow=false;
  } delay(1);
}
