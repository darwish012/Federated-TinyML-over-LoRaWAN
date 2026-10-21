#pragma once
#include <Arduino.h>
#include <FS.h>
#include <SD.h>
#include <SD_MMC.h>
#include <LittleFS.h>
#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <lmic.h>
#include <hal/hal.h>
#include <TinyFLMulticast.h>
#include <unistd.h>
// The SD card is accessed through SdFat on the boards. ESP-IDF's FATFS/VFS layer could not read
// the last cluster of some card files (cards 05 and 10: 1,028 bytes before the end, freshly
// written, FAT and sectors verified intact), while SdFat read the same files completely with a
// matching CRC (2026-10-04). Host tests keep the Arduino fs::FS path (no ESP32 macro there).
#if defined(ESP32) && !defined(TINYFL_NO_SDFAT)
#define TINYFL_SDFAT 1
#include <SdFat.h>
#else
#define TINYFL_SDFAT 0
#endif
#define TINYFL_SD_V4 1
#include "../common/tinyfl_core/tinyfl_core_all.cpp"
#include "../common/tinyfl_core/tf_protocol_v4.h"
#include "../common/tinyfl_core/tf_replay_v4.h"
#include "../common/tinyfl_nvs_keys.h"
// Sensor-only power build (TINYFL_SENSOR=1, with TINYFL_NO_SDFAT): no SD card. The board's
// identity comes from the build flags below, its history is sensed by an MPU6050 at boot
// (calibration), it keeps sensing every TINYFL_SENSE_INTERVAL_S during the run, and the final
// replay rescores the stored feature rows. Phase markers are mirrored on a second UART for
// the power DAQ, the OLED is off, and the battery voltage is logged to LittleFS.
#ifndef TINYFL_SENSOR
#define TINYFL_SENSOR 0
#endif
#if TINYFL_SENSOR
#if TINYFL_SDFAT
#error "the sensor build has no SD card: compile with -DTINYFL_NO_SDFAT"
#endif
#include "../common/tinyfl_mpu6050_source.h"
#include "../common/tinyfl_sensor_store.h"
#ifndef TINYFL_SENSOR_BOARD
#define TINYFL_SENSOR_BOARD 11
#endif
#ifndef TINYFL_SENSOR_POSITIONS
#define TINYFL_SENSOR_POSITIONS 0          // comma list, e.g. 0,1 for two machines on one sensor
#endif
#ifndef TINYFL_SENSOR_TRAIN
#define TINYFL_SENSOR_TRAIN 60             // history rows: train ...
#endif
#ifndef TINYFL_SENSOR_CAL
#define TINYFL_SENSOR_CAL 30               // ... then calibration (alarm threshold)
#endif
#ifndef TINYFL_SENSE_INTERVAL_S
#define TINYFL_SENSE_INTERVAL_S 60         // live acquisition period per machine
#endif
#ifndef TINYFL_BATT_LOG_S
#define TINYFL_BATT_LOG_S 60
#endif
#ifndef TINYFL_MARKER_TX_PIN
#define TINYFL_MARKER_TX_PIN 13            // free without an SD card; to the power DAQ (9600 8N1)
#endif
#define TINYFL_MARKER_RX_PIN 39            // input-only, unused (UART2's default RX 16 is flash on the PICO-D4)
#ifndef TINYFL_CPU_MHZ
#define TINYFL_CPU_MHZ 240                 // power-study option: 80 or 160 lowers the always-on CPU current
#endif
#ifndef TINYFL_LIGHT_SLEEP
#define TINYFL_LIGHT_SLEEP 0               // power-study option: ESP32 light sleep while only listening
#endif
#if TINYFL_LIGHT_SLEEP
#include "esp_sleep.h"
#include "driver/gpio.h"
#endif
#endif
#ifndef TINYFL_SD_MMC
// SPI by default: on the LilyGO T3 v1.6.1 bench boards SD_MMC times out (0x107) and the
// mount hangs into the task watchdog; SPI on the same pins is what worked.
#define TINYFL_SD_MMC 0
#endif
#ifndef TINYFL_PHASE_MARKERS
#define TINYFL_PHASE_MARKERS 1
#endif
#if !LMIC_ENABLE_class_c || !defined(CFG_eu868) || !defined(CFG_sx1276_radio)
#error "EU868 SX1276 Class C build configuration required"
#endif
static_assert(MAX_FRAME_LEN>=255,"LMIC long messages required");
const lmic_pinmap lmic_pins={18,LMIC_UNUSED_PIN,23,{26,33,32},0,10,8000000,nullptr};
namespace tinyfl_node {
using namespace tinyfl;
namespace p4=tinyfl::v4;
p4::Keys keys{};
uint8_t devEuiMsb[8]={};
Adafruit_SSD1306 display(128,64,&Wire,-1);
SPIClass sdSpi(HSPI); fs::FS* card=nullptr;
bool oled=false,radio=false,joined=false,fatal=false,txDone=false,joinFailed=false,allChecks=true;
uint16_t board=0,positions[8]={}; uint8_t machines=0,declaredStreams[8]={};
TfMachineInfo info[8];
const char* phaseName=nullptr; uint32_t nextTxAt=0,lastActivity=0,lastDisplay=0;
void displayRunState();
// Marker lines (PHASE/ROUND/BATT/SCORE) for "final lab/phase_energy.py"; the sensor build
// mirrors them to the power DAQ on Serial2 so current and markers share one clock.
void mark(const char* fmt,...) {
  char line[96]; va_list ap; va_start(ap,fmt); vsnprintf(line,sizeof(line),fmt,ap); va_end(ap);
  Serial.print(line);
#if TINYFL_SENSOR
  Serial2.print(line);
#endif
}
void phase(const char* name) {
  if(phaseName && !strcmp(phaseName,name)) return; phaseName=name;
#if TINYFL_PHASE_MARKERS
  mark("PHASE %s %lu\n",name,(unsigned long)millis());
#endif
}
void screen(const char* name,uint32_t done=0,uint32_t total=1) {
  if(!oled) return;
  display.clearDisplay(); display.setTextColor(SSD1306_WHITE); display.setTextSize(1);
  display.setCursor(0,0); display.printf("TinyFL v4 board %u\n",board);
  display.println(name); display.printf("%lu / %lu\n",(unsigned long)done,(unsigned long)total);
  displayRunState();
  display.drawRect(0,53,128,10,SSD1306_WHITE);
  display.fillRect(1,54,total?uint64_t(126)*min(done,total)/total:0,8,SSD1306_WHITE); display.display();
  lastDisplay=millis();
}
// Class C is switched on once and never off. MCCI LMIC 6.0.1 has a bug in
// externalRequestCb (lmic.c ~3294: `if (targetState == fEnabled) fEnabled = 0;
// else fEnabled = 1;`), so LMIC_enableClassC(1) while already on turns it OFF
// and LMIC_enableClassC(0) while on leaves it ON; only off->on behaves. LMIC
// itself copes with uplinks in Class C (engineUpdate starts continuous RX only
// while no TX is requested; TX and RX1/RX2 run as usual and continuous RX
// resumes afterwards), so nothing here ever requests "off", and "on" is
// requested only when it is neither on nor already requested.
void classCOn() {
  if(!LMIC.classC.flags.f.fEnabled && !LMIC.classC.requests.f.fStateChangeRq) LMIC_enableClassC(1);
}
void stop(const char* why) { Serial.printf("ERROR: %s\n",why); fatal=true;
  phase("idle"); screen(why); }
void copyDownlink(const uint8_t* bytes,size_t n);
void coreYield(void*) { feedLoopWDT(); if(radio) os_runloop_once(); static uint32_t last=0;
#if TINYFL_SENSOR
  // A live acquisition blocks loop() for ~4 s: keep draining the one-frame multicast buffer
  // into the downlink ring so no GLOBAL fragment is dropped meanwhile.
  if(radio) { uint8_t mc[221],port; size_t n; while(TinyFLMulticast::poll(mc,sizeof(mc),&n,&port)) if(port==10) copyDownlink(mc,n); }
#endif
  if(millis()-last>=10) { delay(1); last=millis(); } }
void progress(void*,const char* stage,uint32_t done,uint32_t total) {
#if TINYFL_SENSOR
  // The extractor reports "sd_read" per sample chunk; live samples are already in RAM.
  if(!strcmp(stage,"sd_read") || !strcmp(stage,"features")) { phase("features"); return; }
#endif
  if(!strcmp(stage,"sd_read") || !strcmp(stage,"features") || !strcmp(stage,"replay")) phase(stage);
  else if(millis()-lastDisplay>150 || done==total) screen(stage,done,total);
}
class ArduinoFile : public TfFile {
 public:
  ArduinoFile(File f,fs::FS* fsys,const String& path):f_(f),fs_(fsys),path_(path) {}
  ~ArduinoFile() override { f_.close(); }
  // A short read before EOF is retried from the same position. A failed read can leave the
  // stdio stream in its error state (every later read fails), so retries close and reopen
  // the file and seek back; only EOF or repeated failure returns fewer bytes.
  int64_t read(void* d,size_t n) override {
    uint8_t* p=(uint8_t*)d; size_t got=0; int tries=0;
    while(got<n) {
      // At most one 512-byte sector per call: FatFs then uses single-block reads (CMD17).
      // Card 02 fails a multi-block read (CMD18) at one spot every time, yet reads the whole
      // 41 MB file without an error in 512-byte pieces (sd_read_test, 2026-10-04).
      size_t want=n-got; if(want>512) want=512;
      size_t at=f_.position(); int r=f_.read(p+got,want);
      if(r>0) { got+=(size_t)r; continue; }
      size_t total=f_.size();
      if(at>=total || ++tries>6) {
        if(at<total) Serial.printf("WARN: read %s failed at %lu of %lu\n",path_.c_str(),(unsigned long)at,(unsigned long)total);
        break;
      }
      delay(20*tries); f_.close(); f_=fs_->open(path_,FILE_READ);
      if(!f_ || !f_.seek(at)) { delay(100); f_=fs_->open(path_,FILE_READ); if(!f_ || !f_.seek(at)) break; }
    }
    return (int64_t)got;
  }
  size_t write(const void* d,size_t n) override { return f_.write((const uint8_t*)d,n); }
  bool seek(uint64_t n) override { return n<=UINT32_MAX && f_.seek(n); }
  uint64_t size() override { return f_.size(); } uint64_t position() override { return f_.position(); }
  void flush() override { f_.flush(); }
 private: File f_; fs::FS* fs_; String path_;
};
// The SD card is used read-only: on the bench boards, files written to the card over SPI
// sometimes read back short. The card's own data (raw recordings, self-check references,
// board.txt, manifest) is read from SD; every file the core derives (.tmp, .dif, prepared .x)
// lives in LittleFS on the internal flash, which also holds the checkpoints.
inline bool onCard(const char* name) {
#if TINYFL_SENSOR
  return false;   // no card: the feature log, prepared rows and run state all live in LittleFS
#endif
  size_t n=strlen(name);
  auto ends=[&](const char* s){ size_t k=strlen(s); return n>=k && !strcmp(name+n-k,s); };
  return ends(".raw") || ends(".chk") || !strcmp(name,"board.txt") || !strcmp(name,"manifest.jsn");
}
// Unstick a card that a failed transfer or an ESP32 reset left inside a multi-block read (it
// keeps streaming and ignores CMD0 until power is cut): with CS low, clock out any pending data
// blocks, send CMD12 (stop transmission) and wait for not-busy; then 74+ idle clocks, CS high.
void sdUnstick() {
  sdSpi.beginTransaction(SPISettings(400000,MSBFIRST,SPI_MODE0));
  digitalWrite(13,LOW);
  for(int i=0;i<1100;i++) sdSpi.transfer(0xFF);
  const uint8_t cmd12[6]={0x4C,0,0,0,0,0x61};
  for(uint8_t b:cmd12) sdSpi.transfer(b);
  // Skip the stuff byte, wait for the R1 response (bit 7 clear), then for the end of busy (0xFF).
  sdSpi.transfer(0xFF);
  for(int i=0;i<16 && (sdSpi.transfer(0xFF)&0x80);i++) {}
  for(int i=0;i<2000 && sdSpi.transfer(0xFF)!=0xFF;i++) {}
  digitalWrite(13,HIGH);
  for(int i=0;i<20;i++) sdSpi.transfer(0xFF);
  sdSpi.endTransaction();
}
#if TINYFL_SDFAT
SdFs sdfat;
// After one failed transfer some cards stay wedged: every further read of ANY sector fails with
// the same error until the card is re-initialised (card 09 on board 9: a first read timeout, then
// "bad data token" on all retries, at a spot that moved between boots). Re-initialise only the
// card driver; the volume and every open file keep their state (the card content is unchanged).
bool sdCardReinit() {
  for(int i=0;i<4;i++) {
    sdUnstick();
    if(sdfat.card()->begin(SdSpiConfig(13,DEDICATED_SPI,SD_SCK_MHZ(4),&sdSpi))) return true;
    delay(200); feedLoopWDT();
  }
  return false;
}
class SdCardFile : public TfFile {
 public:
  bool openRead(const char* name) {
    strncpy(name_,name,sizeof(name_)-1); name_[sizeof(name_)-1]=0; return f_.open(&sdfat,name,O_RDONLY);
  }
  ~SdCardFile() override { f_.close(); }
  // A failed read is retried from the same offset; from the second retry on, the card is
  // re-initialised and the file reopened first, so a wedged card recovers.
  int64_t read(void* d,size_t n) override {
    uint8_t* p=(uint8_t*)d; size_t got=0; int tries=0;
    while(got<n) {
      uint64_t at=f_.curPosition(); int r=f_.read(p+got,n-got);
      if(r>0) { got+=(size_t)r; continue; }
      if(at>=f_.fileSize()) break;
      Serial.printf("WARN: SD read failed at %llu of %llu (SdFat error 0x%02x), retry %d\n",(unsigned long long)at,
                    (unsigned long long)f_.fileSize(),sdfat.sdErrorCode(),tries+1);
      if(++tries>8) break;
      delay(50*tries); feedLoopWDT();
      if(tries>=2) {
        bool ok=sdCardReinit(); f_.close();
        ok=ok && f_.open(&sdfat,name_,O_RDONLY);
        Serial.printf("WARN: SD card re-initialised %s\n",ok?"(file reopened)":"FAILED");
        if(!ok) continue;
      }
      if(!f_.seekSet(at)) continue;
    }
    return (int64_t)got;
  }
  size_t write(const void*,size_t) override { return 0; }  // the card is read-only
  bool seek(uint64_t n) override { return f_.seekSet(n); }
  uint64_t size() override { return f_.fileSize(); } uint64_t position() override { return f_.curPosition(); }
  void flush() override {}
 private: FsFile f_; char name_[16]={};
};
#endif
// Card access (read-only data files, board.txt, and the optional results log).
TfFile* cardOpenRead(const char* name) {
#if TINYFL_SDFAT
  SdCardFile* f=new(std::nothrow) SdCardFile; if(f && !f->openRead(name)) { delete f; f=nullptr; } return f;
#else
  if(!card) return nullptr;
  String path=String("/")+name; File f=card->open(path,FILE_READ); return f?new ArduinoFile(f,card,path):nullptr;
#endif
}
bool cardExists(const char* name) {
#if TINYFL_SDFAT
  return sdfat.exists(name);
#else
  return card && card->exists(String("/")+name);
#endif
}
bool cardRemove(const char* name) {
#if TINYFL_SDFAT
  return sdfat.remove(name);
#else
  return card && card->remove(String("/")+name);
#endif
}
bool cardRename(const char* from,const char* to) {
#if TINYFL_SDFAT
  return sdfat.rename(from,to);
#else
  return card && card->rename(String("/")+from,String("/")+to);
#endif
}
// Reads a small text file into out (NUL-terminated, at most cap-1 bytes).
bool cardReadText(const char* name,char* out,size_t cap) {
  TfFile* f=cardOpenRead(name); if(!f) return false;
  int64_t r=f->read(out,cap-1); delete f;
  if(r<0) return false; out[r]=0; return true;
}
class CardFs : public TfFileSystem {
 public:
  TfFile* open(const char* name,TfOpenMode mode) override {
    if(onCard(name)) return mode==kOpenRead?cardOpenRead(name):nullptr;  // the card is read-only
    String path=String("/")+name;
    File f=LittleFS.open(path,mode==kOpenRead?FILE_READ:(mode==kOpenWrite?FILE_WRITE:FILE_APPEND));
    return f?new ArduinoFile(f,&LittleFS,path):nullptr;
  }
  bool exists(const char* name) override { return onCard(name)?cardExists(name):LittleFS.exists(String("/")+name); }
  bool remove(const char* name) override { return onCard(name)?cardRemove(name):LittleFS.remove(String("/")+name); }
} cardFs;
inline bool hasBit(const uint8_t* a,int i) { return (a[i/8]>>(i%8))&1; }
inline void setbit(uint8_t* a,int i,bool value=true) {
  if(value) a[i/8]|=1<<(i%8); else a[i/8]&=~(1<<(i%8));
}
// Rev 4.3 schema: CRC-protected held GLOBAL and final-phase participation.
// A previous revision's differently sized checkpoint cannot be reused.
struct Progress {
  uint32_t magic=0x434c4654,generation=0,run=0,stateGeneration=0,archiveRun=0;
  uint16_t board=0,positions[8]={},heldSlot=0,seedSlot=0;
  uint8_t machines=0,phase=0,round=0,rounds=0,sent=0,complete=0;
  uint8_t finals[32]={},requested[32]={},done[8][32]={},doneRetry[8][32]={},current[32]={};
  uint8_t replayMachine=0,replayStarted=0,replayFinishing=0,replayStopped=0;
  uint32_t replayNext=0,csvOffset=0,totalAlarms=0,inputCrc[8]={};
  uint8_t base[p4::GlobalBytes]={};
  uint8_t outLen=0,outKind=0,outPhase=0,outMachine=0,outFrame[32]={};
  uint8_t participants[255]={}; // hosted machine bits for each final model
} saved;
void displayRunState() {
  display.printf("P%u R%u/%u\nAlarms %lu\n",saved.phase,saved.round,saved.rounds,(unsigned long)saved.totalAlarms);
}
p4::Assembly assembler,restartCandidate;
float baseWeights[kNumParams]; uint8_t updates[8][p4::UpdateBytes]={},validUpdates=0,repairMask[8]={};
uint8_t frame[221],down[4][221],downLen[4]={},head=0,tail=0;
uint32_t rxOverruns=0;
bool needReady=false,needStatus=false,waiting=false; uint16_t statusSlot=0; uint8_t statusMask=0;
enum TxKind:uint8_t { None,Hello,Check,Update,Repair,Ready,Status,Outbox,HelloReady,AdoptHello };
TxKind inFlight=None; uint8_t txMachine=0,txFrag=0,helloStage=0;
uint8_t activeMachines=0;
uint32_t heardRun=0; // deliberately volatile: must hear the held run since boot
uint32_t pendingAdoptHellos=0;
bool completedBootAnnounce=false; // reboot-only discovery of a completed board
uint8_t hostedMask(const p4::Model& g) {
  uint8_t mask=0;
  for(int m=0;m<machines;m++) if(p4::participates(g,positions[m])) mask|=1<<m;
  return mask;
}
TfAlarmState* states=nullptr; uint8_t replayPhases[255]={}; uint16_t nReplay=0;
#if TINYFL_SENSOR
using ReplayT=SensorReplay;      // rescoring of the stored live feature rows
Mpu6050Source sensor; TfExtractor sensorEx; bool sensorReady=false;
#else
using ReplayT=p4::FeatureReplay;
#endif
ReplayT* replay=nullptr; bool csvOn=false;
// Optional per-acquisition score log on the card (results.csv); saved.csvOffset is the length
// covered by the last checkpoint, so a reboot truncates any partial tail and appends from there.
#if TINYFL_SDFAT
FsFile csv;
bool csvBegin() {
  if(!sdfat.exists("results.csv")) {
    if(!csv.open(&sdfat,"results.csv",O_RDWR|O_CREAT|O_TRUNC)) return false;
    csv.print("position,index,phase,score,flag\n");
    if(!csv.sync()) { csv.close(); return false; }
    saved.csvOffset=csv.curPosition(); return true;
  }
  if(!csv.open(&sdfat,"results.csv",O_RDWR)) return false;
  if(csv.fileSize()<saved.csvOffset || !csv.truncate(saved.csvOffset) || !csv.seekEnd()) { csv.close(); return false; }
  return true;
}
void csvClose() { if(csv.isOpen()) csv.close(); }
void csvSync() { if(csvOn) { csv.sync(); saved.csvOffset=csv.curPosition(); } }
#else
File csv;
bool csvBegin() {
  if(!card) return false;   // sensor build: no card, no optional results log
  if(!card->exists("/results.csv")) {
    csv=card->open("/results.csv",FILE_WRITE); if(!csv) return false;
    csv.println("position,index,phase,score,flag"); csv.flush(); saved.csvOffset=csv.position(); csv.close();
  } else {
    File f=card->open("/results.csv",FILE_READ); bool ok=f && f.size()>=saved.csvOffset; f.close();
    if(!ok || truncate("/sdcard/results.csv",saved.csvOffset)!=0) return false;
  }
  csv=card->open("/results.csv",FILE_APPEND); return (bool)csv;
}
void csvClose() { if(csv) csv.close(); }
void csvSync() { if(csvOn) { csv.flush(); saved.csvOffset=csv.position(); } }
#endif
bool csvLine(const char* s) { size_t n=strlen(s); return csv.write((const uint8_t*)s,n)==n; }
bool blobRead(const char* path,void* data,size_t n) {
  File f=LittleFS.open(path,FILE_READ); uint8_t t[4];
  return f && f.size()==n+4 && f.read((uint8_t*)data,n)==n && f.read(t,4)==4 && get_u32(t)==tf_crc32(0,data,n);
}
bool blobWrite(const char* path,const void* data,size_t n) {
  String tmp=String(path)+".tmp"; File f=LittleFS.open(tmp,FILE_WRITE); uint8_t t[4];
  put_u32(t,tf_crc32(0,data,n)); bool ok=f && f.write((const uint8_t*)data,n)==n && f.write(t,4)==4;
  f.flush(); f.close(); return ok && LittleFS.rename(tmp,path);
}
bool checkpoint() {
  if(states) {
    saved.stateGeneration++;
    if(!blobWrite(saved.stateGeneration&1?"/rings1":"/rings0",states,nReplay*sizeof(*states))) {
      stop("RINGS SAVE FAIL"); return false;
    }
  }
  saved.generation++;
  if(!blobWrite(saved.generation&1?"/v4progress1":"/v4progress0",&saved,sizeof(saved))) {
    stop("STATE SAVE FAIL"); return false;
  } return true;
}
void releaseReplay() {
  delete replay; replay=nullptr; delete[] states; states=nullptr; nReplay=0; csvClose();
}
void modelPath(char* path,uint8_t phase) { snprintf(path,24,"/p%03u.model",phase); }
bool modelRead(uint8_t phase,p4::Model& model) {
  uint8_t bytes[p4::GlobalBytes]; char path[24]; modelPath(path,phase);
  return blobRead(path,bytes,sizeof(bytes)) && p4::decodeModel(bytes,sizeof(bytes),p4::GLOBAL,model) &&
         model.run==saved.run && model.phase==phase && (model.flags&p4::Final);
}
bool readCardManifest() {
  static char text[2048]; if(!cardReadText("board.txt",text,sizeof(text))) return false;
  unsigned b=0,v=0; char extra; char* at=text;
  auto nextLine=[&]() -> char* {
    if(!*at) return nullptr; char* l=at; char* e=strchr(at,'\n');
    if(e) { *e=0; at=e+1; } else at+=strlen(at);
    size_t n=strlen(l); while(n && (l[n-1]=='\r' || l[n-1]==' ')) l[--n]=0;
    return l;
  };
  char* line=nextLine();
  if(!line || sscanf(line,"board_id %u %c",&b,&extra)!=1 || !b || b>=65535) return false;
  line=nextLine(); if(!line || sscanf(line,"format_version %u %c",&v,&extra)!=1 || v!=2) return false;
  board=b; machines=0;
  while((line=nextLine())) {
    while(*line==' ') line++; if(!*line) continue;
    unsigned p=0,s=0; char machineId[96];
    if(machines>=8 || sscanf(line,"position %u machine_id %95s streams %u %c",&p,machineId,&s,&extra)!=3 ||
       p>=65535 || !s || s>kMaxStreams) return false;
    for(int i=0;i<machines;i++) if(positions[i]==p) return false;
    positions[machines]=p; declaredStreams[machines++]=s;
  }
  // Deterministic ascending-position update and repair queue.
  for(int i=0;i<machines;i++) for(int j=i+1;j<machines;j++) if(positions[j]<positions[i]) {
    uint16_t p=positions[i]; positions[i]=positions[j]; positions[j]=p;
    uint8_t s=declaredStreams[i]; declaredStreams[i]=declaredStreams[j]; declaredStreams[j]=s;
  } return machines>0;
}
bool loadProgress() {
  // Static, not on the stack: two ~1.6 KB checkpoints overflowed the 8 KB loopTask stack on hardware.
  static Progress a,b;
  bool av=blobRead("/v4progress0",&a,sizeof(a)) && a.magic==0x434c4654;
  bool bv=blobRead("/v4progress1",&b,sizeof(b)) && b.magic==0x434c4654;
  if(!av && !bv) return false;
  saved=!av || (bv && int32_t(b.generation-a.generation)>0)?b:a;
  if(saved.board!=board || saved.machines!=machines || memcmp(saved.positions,positions,2*machines) ||
     saved.outLen>32 || saved.outMachine>=machines || saved.replayMachine>machines || saved.phase==255 || saved.heldSlot>p4::MaxSlot ||
     saved.seedSlot>p4::MaxSlot || saved.replayNext>100000) stop("SAVED CARD/STATE INVALID");
  return !fatal;
}
void expectGlobal() { needReady=true; waiting=true; lastActivity=millis(); }
bool finishArchive() {
  if(!saved.archiveRun) return true;
  // The archive-pending identity is committed BEFORE rename; a reboot safely
  // resumes the rename, or observes it already finished. Never erase inputs.
  if(cardExists("results.csv")) {
    String archive=String("results_run")+saved.archiveRun+"_"+saved.generation+".csv";
    if(!cardRename("results.csv",archive.c_str())) Serial.println("WARN: results.csv archive failed (optional log)");
  }
  saved.archiveRun=0; return checkpoint();
}
void resetRun(uint32_t run) {
  releaseReplay();
  // Old model blobs are retained; run IDs/cleared final bits forbid reuse.
  uint32_t generation=saved.generation,oldRun=saved.run; static const Progress fresh; saved=fresh;
  saved.generation=generation; saved.run=run; saved.archiveRun=oldRun;
  activeMachines=0; heardRun=0; pendingAdoptHellos=0; completedBootAnnounce=false;
  saved.board=board; saved.machines=machines; memcpy(saved.positions,positions,2*machines);
  for(int i=0;i<machines;i++) saved.inputCrc[i]=info[i].inputCrc;
  validUpdates=0; txFrag=0; memset(repairMask,0,sizeof(repairMask)); assembler.reset(); restartCandidate.reset();
  needStatus=false; if(!checkpoint() || !finishArchive()) return; expectGlobal();
}
void restoreBase() {
  if(saved.heldSlot) {
    p4::Model g;
    if(!p4::decodeModel(saved.base,sizeof(saved.base),p4::GLOBAL,g) || g.run!=saved.run || g.phase!=saved.phase) {
      stop("BASE INVALID"); return;
    } activeMachines=hostedMask(g); tf_dequantise(g.codes,g.scales,baseWeights);
    for(int m=0;m<machines;m++) {
      char path[24]; snprintf(path,sizeof(path),"/v4update%u",m); p4::Model u;
      if(blobRead(path,updates[m],p4::UpdateBytes) && p4::decodeModel(updates[m],p4::UpdateBytes,p4::UPDATE,u) &&
         u.run==saved.run && u.position==positions[m] && u.phase==saved.phase &&
         u.round==saved.round && u.base<=saved.heldSlot) validUpdates|=1<<m;
    }
  }
  // Reboot exception (section 5): announce the stored completed run until
  // hearing a control or a CRC-valid GLOBAL, without resuming its work.
  if(saved.complete) {
    completedBootAnnounce=true; lastActivity=millis();
    needReady=false; needStatus=false; waiting=false; classCOn(); return;
  }
  if(!saved.replayStarted && (!saved.heldSlot || !saved.round || (saved.sent&activeMachines)==activeMachines))
    expectGlobal();
}
void acceptGlobal(const uint8_t* bytes,uint16_t slot) {
  p4::Model g;
  if(!slot || slot>p4::MaxSlot || !p4::decodeModel(bytes,p4::GlobalBytes,p4::GLOBAL,g) || !g.run) return;
  completedBootAnnounce=false;
  if(g.run!=saved.run) {
    if(saved.run && !saved.complete && g.round!=0) return;
    resetRun(g.run); if(fatal) return;
  } else if(saved.complete) return;
  heardRun=saved.run;
  if(slot<=saved.heldSlot) return;
  // A new slot is server authority; no v3 level/round formula or phase mask.
  bool sameRound=saved.heldSlot && saved.phase==g.phase && saved.round==uint8_t(g.round+1) &&
                 saved.seedSlot==slot && !(g.flags&p4::Final);
  releaseReplay(); saved.replayStarted=0; saved.replayFinishing=0;
  saved.phase=g.phase; saved.rounds=g.rounds; saved.round=g.round==g.rounds?0:uint8_t(g.round+1);
  saved.heldSlot=slot; saved.seedSlot=slot; saved.complete=0; saved.replayStopped=0;
  if(!sameRound) { saved.sent=0; validUpdates=0; txFrag=0; memset(repairMask,0,sizeof(repairMask)); }
  memcpy(saved.base,bytes,p4::GlobalBytes); tf_dequantise(g.codes,g.scales,baseWeights);
  activeMachines=hostedMask(g);
  waiting=false; needReady=false;
#if TINYFL_PHASE_MARKERS
  if(g.flags&p4::Final) mark("ROUND end %lu\n",(unsigned long)millis());
  else mark("ROUND run%lu-P%u:R%u %lu\n",(unsigned long)saved.run,g.phase,g.round+1,(unsigned long)millis());
#endif
  if(g.flags&p4::Final) {
    char path[24]; modelPath(path,g.phase);
    if(!blobWrite(path,bytes,p4::GlobalBytes)) { stop("FINAL MODEL SAVE FAIL"); return; }
    saved.participants[g.phase]=activeMachines;
    setbit(saved.finals,g.phase); expectGlobal(); // held_slot+1, never next phase guessed.
  }
  else if(!activeMachines) expectGlobal();
  checkpoint(); statusSlot=slot; statusMask=0; needStatus=true; phase("idle");
  char text[48]; snprintf(text,sizeof(text),"PHASE %u ROUND %u/%u",g.phase,g.round,g.rounds); screen(text,g.round,g.rounds);
}
void copyDownlink(const uint8_t* bytes,size_t n) {
  if(n>221) return; uint8_t next=(head+1)%4;
  if(next==tail) { rxOverruns++; return; } memcpy(down[head],bytes,n); downLen[head]=n; head=next;
}
void processDownlink(const uint8_t* f,size_t n) {
  p4::Header h; const uint8_t* b; if(!p4::parse(f,n,h,b)) return;
  if(h.type==p4::GLOBAL) {
    p4::Assembly& a=saved.heldSlot && h.slot<=saved.heldSlot?restartCandidate:assembler;
    int result=a.add(f,n); if(!result) return; lastActivity=millis();
    if(result==2) {
      if(h.slot==saved.heldSlot && !memcmp(a.bytes,saved.base,p4::GlobalBytes)) {
        p4::Model g;
        if(p4::decodeModel(a.bytes,p4::GlobalBytes,p4::GLOBAL,g) && g.run==saved.run) {
          completedBootAnnounce=false;
          if(!saved.complete) { heardRun=saved.run; needStatus=true; statusMask=0; statusSlot=h.slot; }
        }
      } else acceptGlobal(a.bytes,h.slot); a.reset();
    } return;
  }
  if(!p4::controlValid(h,b)) return;
  if(h.type==p4::CONTROL) {
    uint32_t run=get_u32(b); if(!run) return;
    completedBootAnnounce=false;
    if(b[4]==p4::ADOPT) {
      if(run==saved.run && saved.complete) return;
      if(run!=saved.run) resetRun(run);
      if(!fatal) pendingAdoptHellos=1; // one HELLO answers any number of ADOPTs
      return;
    }
    if(run!=saved.run) return;
    heardRun=run;
    if(b[4]==p4::END) {
      // Stop replay only after a durable checkpoint, retain all models/results.
      checkpoint(); releaseReplay(); saved.replayStopped=1; saved.complete=1; saved.outLen=0;
      checkpoint(); pendingAdoptHellos=0; waiting=false; needReady=false; needStatus=false;
      classCOn(); phase("idle"); screen("DONE"); return;
    }
    if(saved.complete) return;
    // Chunked REPLAY lists are accumulated; already completed pairs resend
    // DONE on repeat controls (unconfirmed uplinks may have been lost).
    for(int i=0;i<b[5];i++) if(hasBit(saved.finals,b[6+i])) {
      setbit(saved.requested,b[6+i]);
      for(int m=0;m<machines;m++) if(hasBit(saved.done[m],b[6+i])) setbit(saved.doneRetry[m],b[6+i]);
    }
    saved.complete=0; saved.replayStopped=0; waiting=false; needReady=false; checkpoint(); return;
  }
  if(h.type!=p4::REPAIR || get_u32(b)!=saved.run || !saved.heldSlot || !saved.round || saved.replayStarted) return;
  uint16_t seed=h.slot,position=get_u16(b+4); int m=0; while(m<machines && positions[m]!=position) m++;
  if(saved.complete || m==machines || !(activeMachines&(1<<m)) || seed<saved.seedSlot || seed<saved.heldSlot) return;
  if(seed>saved.seedSlot) {
    uint32_t round=uint32_t(saved.round)+(seed-saved.seedSlot);
    if(round>saved.rounds) return; saved.round=round; saved.seedSlot=seed;
    saved.sent=0; validUpdates=0; txFrag=0; memset(repairMask,0,sizeof(repairMask)); checkpoint();
  }
  // Rev 4.3 radio scheduling: the board keeps receiving until queue() can really
  // start the answering uplink (a repair, like everything else, may have to wait).
  repairMask[m]|=b[6]; waiting=false; needReady=false; phase("idle");
}
bool prepareUpdate(int m) {
  if(!(activeMachines&(1<<m))) return false;
  if(validUpdates&(1<<m)) return true; float* x=nullptr; TfMachineInfo current;
  phase("sd_read"); screen("SD TRAIN ROWS",m,machines);
  if(!tf_load_prepared(&cardFs,positions[m],&current,&x)) { stop(tf_last_error()); return false; }
  phase("train"); screen("TRAIN",m,machines); float local[kNumParams]; memcpy(local,baseWeights,sizeof(local));
  tf_local_train(x,current.nTrain,local); float loss=tf_mean_loss(local,x,current.nTrain); tf_free(x);
  if(!isfinite(loss)) { stop("TRAIN LOSS INVALID"); return false; }
  for(int i=0;i<kNumParams;i++) local[i]-=baseWeights[i]; p4::Model u{};
  u.run=saved.run; u.position=positions[m]; u.phase=saved.phase; u.round=saved.round;
  u.base=saved.heldSlot; u.rows=current.nTrain; u.loss=loss; tf_quantise(local,u.codes,u.scales);
  if(!p4::encodeModel(updates[m],p4::UPDATE,u)) { stop("UPDATE INVALID"); return false; }
  char path[24]; snprintf(path,sizeof(path),"/v4update%u",m);
  if(!blobWrite(path,updates[m],p4::UpdateBytes)) { stop("UPDATE SAVE FAIL"); return false; }
  validUpdates|=1<<m; phase("idle"); return true;
}
// Rev 4.3 radio scheduling. True only if LMIC would start an SF7 uplink right
// now, i.e. no EU868 duty-cycle wait: the same decision lmic.c makes in
// engineUpdate() (earliest feasible band via LMICeu868_nextTx, globalDutyAvail
// when it applies, TX_RAMPUP). Reading LMIC's own band timers keeps the board
// and the host build identical and also covers the delays LMIC adds itself
// (post-TX recovery, RX2 safety zone, DutyCycleReq).
// Signed difference of two LMIC tick values with defined 32-bit wrap-around.
inline ostime_t tickDiff(ostime_t a,ostime_t b) { return (ostime_t)((uint32_t)a-(uint32_t)b); }
bool uplinkReady() {
  const ostime_t now=os_getTime(),horizon=sec2osticks(3600);
  // LMIC measures time in signed 32-bit tick differences and uses now+8 h as
  // its "nothing available" sentinel, so a band timer more than ~1.5 h old
  // reads as unavailable and would stall LMIC for hours (a completed board is
  // silent for exactly that long). A real block lasts under a minute: reset
  // any timer further than 1 h from now.
  for(int b=0;b<MAX_BANDS;b++) {
    ostime_t d=tickDiff(LMIC.bands[b].avail,now); if(d<-horizon || d>horizon) LMIC.bands[b].avail=now;
  }
  { ostime_t d=tickDiff(LMIC.globalDutyAvail,now); if(d<-horizon || d>horizon) LMIC.globalDutyAvail=now; }
  ostime_t at=(ostime_t)((uint32_t)now+(uint32_t)sec2osticks(28800));
  for(int c=0;c<MAX_CHANNELS;c++) {
    if(LMIC.channelMap<(1u<<c)) break;
    if(!(LMIC.channelMap&(1u<<c)) || !(LMIC.channelDrMap[c]&(1u<<DR_SF7))) continue;
    ostime_t avail=LMIC.bands[LMIC.channelFreq[c]&3].avail; if(tickDiff(at,avail)>0) at=avail;
  }
  if((LMIC.globalDutyRate!=0 || (LMIC.opmode&OP_RNDTX)) && tickDiff(at,LMIC.globalDutyAvail)<0) at=LMIC.globalDutyAvail;
  return tickDiff(at,(ostime_t)((uint32_t)now+(uint32_t)TX_RAMPUP))<0;
}
// Every uplink goes through here. LMIC does not run Class C reception from the
// moment a TX is requested until its RX windows have passed, so a frame is handed
// over only when it can start at once. While the duty cycle forbids it nothing is
// requested: the board keeps receiving in Class C and work() retries next pass.
bool queue(size_t n,TxKind kind) {
  if((saved.complete && !(completedBootAnnounce && (kind==Hello || kind==Check || kind==HelloReady))) ||
     !n || n>221 || (LMIC.opmode&OP_TXRXPEND) || !uplinkReady()) return false;
  LMIC_setDrTxpow(DR_SF7,14);
  auto error=LMIC_setTxData2_strict(10,frame,n,0);
  if(error==LMIC_ERROR_TX_BUSY) return false;
  if(error!=LMIC_ERROR_SUCCESS) { stop("TX REJECTED"); return false; }
  if(kind==Update || kind==Repair) {
    char name[32]; snprintf(name,sizeof(name),"TX p%u frag %u/3",positions[txMachine],txFrag+1);
    screen(name,txFrag+1,3);
  }
  inFlight=kind; phase("idle"); return true;
}
bool pendingReplay() {
  for(int m=0;m<machines;m++) for(int p=0;p<255;p++)
    if(hasBit(saved.doneRetry[m],p) || (hasBit(saved.requested,p) && hasBit(saved.finals,p) &&
       (saved.participants[p]&(1<<m)) && !hasBit(saved.done[m],p))) return true;
  return false;
}
void finishTx() {
  TxKind done=inFlight; inFlight=None;
  if(saved.complete) {
    if(done==Hello) helloStage=1;
    if(done==Check) helloStage++;
    if(done==Hello || done==HelloReady) lastActivity=millis();
    classCOn(); phase("rx_classc"); return;
  }
  switch(done) {
    case Hello: helloStage=1; break;
    case AdoptHello:
      if(pendingAdoptHellos) pendingAdoptHellos--;
      if(helloStage==0) helloStage=1;
      if(waiting) needReady=false;
      lastActivity=millis(); classCOn(); phase("rx_classc"); return;
    case Check: helloStage++; break;
    case Update:
      if(++txFrag==3) { txFrag=0; saved.sent|=1<<txMachine; checkpoint();
        if((saved.sent&activeMachines)==activeMachines) expectGlobal(); }
      break;
    case Repair: {
      repairMask[txMachine]&=~(1<<txFrag); txFrag=0;
      if(!repairMask[txMachine]) saved.sent|=1<<txMachine; checkpoint();
      bool any=false; for(int m=0;m<machines;m++) any|=repairMask[m]!=0;
      if(!any) expectGlobal(); break;
    }
    case HelloReady: case Ready:
      needReady=false; waiting=true;
      if(LMIC.dn2Dr!=DR_SF7) { stop("RX2 NEEDS DR5"); return; }
      classCOn(); lastActivity=millis(); phase("rx_classc");
      screen("WAIT GLOBAL",__builtin_popcount(assembler.mask),3); return;
    case Status:
      // Rev 4.3: GLOBAL_STATUS is itself the board's "listening" report. A
      // board that was waiting for a GLOBAL (or is missing fragments) goes back
      // to Class C waiting, like after READY, with no extra READY/HELLO: such an
      // uplink would only keep it deaf through the duty-cycle wait during which
      // the server's prompt repair arrives. Periodic HELLO-for-READY still
      // applies when the idle timer next fires. A READY already owed (needReady
      // from expectGlobal) is unaffected.
      needStatus=false;
      if(waiting || statusMask) { waiting=true; lastActivity=millis(); }
      break;
    case Outbox:
      if(saved.outKind==2) setbit(saved.done[saved.outMachine],saved.outPhase);
      if(saved.outKind==3) setbit(saved.doneRetry[saved.outMachine],saved.outPhase,false);
      if(states) {
        for(int i=0;i<nReplay;i++) if(replayPhases[i]==saved.outPhase) {
          if(saved.outKind==1) states[i].sent=1;
        }
      }
      {
        bool wasDone=saved.outKind==2 || saved.outKind==3;
        saved.outLen=0; saved.outKind=0;
        if(wasDone && !pendingReplay()) {
          releaseReplay(); saved.replayStarted=0; saved.replayFinishing=0; saved.replayNext=0;
          expectGlobal(); classCOn();
        }
        checkpoint();
      } break;
    default: break;
  }
  // Rev 4.3: after any uplink the board listens again. If the next one is
  // blocked by the duty cycle (typically 37 s after a 221-byte fragment) it
  // waits in Class C (still enabled, see classCOn), not deaf in Class A.
  classCOn(); phase("rx_classc");
}
bool stageOutbox(uint8_t kind,uint8_t phaseIndex,size_t n) {
  if(saved.outLen || n>32 || !n) return false;
  saved.outLen=n; saved.outKind=kind; saved.outPhase=phaseIndex; memcpy(saved.outFrame,frame,n); return checkpoint();
}
bool startReplay() {
  if(!saved.replayStarted) {
    bool found=false;
    for(int m=0;m<machines && !found;m++) {
      memset(saved.current,0,32);
      for(int p=0;p<255;p++) if(hasBit(saved.requested,p) && hasBit(saved.finals,p) &&
         (saved.participants[p]&(1<<m)) && !hasBit(saved.done[m],p)) setbit(saved.current,p);
      for(int p=0;p<255;p++) if(hasBit(saved.current,p)) { saved.replayMachine=m; found=true; break; }
    }
    if(!found) { expectGlobal(); return false; }
    saved.replayStarted=1; saved.replayFinishing=0; saved.replayNext=0;
  }
  nReplay=0; for(int p=0;p<255;p++) if(hasBit(saved.current,p)) replayPhases[nReplay++]=p;
  if(!nReplay || saved.replayMachine>=machines) { stop("REPLAY STATE INVALID"); return false; }
  states=new(std::nothrow) TfAlarmState[nReplay]; if(!states) { stop("REPLAY HEAP LIMIT"); return false; }
  if(saved.replayNext || saved.replayFinishing) {
    if(!blobRead(saved.stateGeneration&1?"/rings1":"/rings0",states,nReplay*sizeof(*states))) {
      stop("REPLAY RINGS INVALID"); return false;
    }
  } else for(int i=0;i<nReplay;i++) tf_alarm_init(states+i);
  for(int i=0;i<nReplay;i++) { p4::Model g; if(!modelRead(replayPhases[i],g)) { stop("REPLAY MODEL INVALID"); return false; } }
#if TINYFL_SENSOR
  // Everything sensed so far is scored; sensing stops once a replay is requested (sensorTick),
  // so this count is stable across a reboot during the replay.
  info[saved.replayMachine].nAcq=sensorLogRows(&cardFs,positions[saved.replayMachine]);
#endif
  replay=new(std::nothrow) ReplayT;
  if(!replay || !replay->begin(&cardFs,info[saved.replayMachine],saved.replayNext)) { stop(tf_last_error()); return false; }
  // results.csv (per-acquisition scores) is an optional on-card log: the run's results travel
  // by radio (ALARM/REPLAY_DONE). SD writes are unreliable on the bench boards, so any CSV
  // failure only disables the log for this boot, it never stops the board.
  csvOn=csvBegin();
  if(!csvOn) Serial.println("WARN: results.csv disabled (SD write failed; optional log)");
  checkpoint(); return true;
}
// Fast boot: reuse the prepared m<PPPP>.x in LittleFS (features, standardisation and the
// self-check verdict of an earlier boot) when every raw stream's identity (header CRC +
// stored data CRC) equals the one recorded in the checkpoint. A changed or different card
// falls back to the full self-check, which then stops with RAW CARD CHANGED.
bool cachedMachine(int m) {
  if(!saved.inputCrc[m]) return false;
  TfMachineInfo cached;
  if(!tf_load_prepared(&cardFs,positions[m],&cached,nullptr) || cached.nStreams!=declaredStreams[m]) return false;
  uint32_t id=0;
  for(int s=0;s<cached.nStreams;s++) {
    char name[16]; tf_raw_name(name,positions[m],(uint8_t)s);
    TfSdReplaySource src; if(!src.open(&cardFs,name,false)) return false;
    uint8_t b[4]; put_u32(b,src.identityCrc()); id=tf_crc32(id,b,4);
  }
  if(id!=saved.inputCrc[m]) return false;
  cached.inputCrc=id; info[m]=cached; return true;
}
void replayStep() {
  if(!replay && !startReplay()) return;
  int m=saved.replayMachine;
  for(int i=0;i<nReplay;i++) if(states[i].latched && !states[i].sent) {
    char path[24]; snprintf(path,sizeof(path),"/v4alarm%u",replayPhases[i]);
    if(!blobRead(path,frame,27)) { stop("ALARM RECOVERY FAIL"); return; }
    stageOutbox(1,replayPhases[i],27); return;
  }
  if(saved.replayFinishing || saved.replayNext>=info[m].nAcq) {
    if(!saved.replayFinishing) {
      if(!replay->finish()) { stop("REPLAY CRC FAIL"); return; }
      saved.replayFinishing=1; csvSync(); checkpoint();
    }
    for(int i=0;i<nReplay;i++) if(!hasBit(saved.done[m],replayPhases[i])) {
      uint32_t v[]={saved.run,positions[m],replayPhases[i],info[m].nAcq,states[i].latched?1u:0u,
                    states[i].latched?states[i].firstAlarm:0xffffffffu};
      size_t n=p4::control(frame,p4::DONE,p4::NoSlot,v,nullptr,nullptr,0);
      char path[32]; snprintf(path,sizeof(path),"/done%u_%u",m,replayPhases[i]);
      if(!blobWrite(path,frame,n)) { stop("DONE SAVE FAIL"); return; }
      saved.outMachine=m; stageOutbox(2,replayPhases[i],n); return;
    }
    releaseReplay(); saved.replayStarted=0; saved.replayFinishing=0; saved.replayNext=0; checkpoint(); return;
  }
  float x[kFeat]; phase("replay"); if(!replay->step(x)) { stop(tf_last_error()); return; }
  phase("replay"); // extractor marked sd_read/features; scoring is a new phase.
  bool latched=false;
  for(int i=0;i<nReplay;i++) {
    p4::Model g; if(!modelRead(replayPhases[i],g)) { stop("MODEL READ FAIL"); return; }
    float w[kNumParams]; tf_dequantise(g.codes,g.scales,w); float score=tf_score_row(w,x);
    if(!isfinite(score) || score<0) { stop("REPLAY SCORE INVALID"); return; }
    auto ev=tf_alarm_step(states+i,info[m].part,saved.replayNext,score);
    char row[64]; snprintf(row,sizeof(row),"%u,%lu,%u,%.9g,%u\n",positions[m],
                           (unsigned long)saved.replayNext,replayPhases[i],score,ev.flag?1:0);
    if(csvOn && !csvLine(row)) {
      csvOn=false; csvClose(); Serial.println("WARN: results.csv disabled (SD write failed; optional log)");
    }
    if(ev.newlyLatched) {
      uint32_t v[]={saved.run,positions[m],replayPhases[i],saved.replayNext}; float f[]={score,(float)ev.threshold};
      size_t n=p4::control(frame,p4::ALARM,p4::NoSlot,v,f,nullptr,0);
      char path[24]; snprintf(path,sizeof(path),"/v4alarm%u",replayPhases[i]);
      if(!blobWrite(path,frame,n)) { stop("ALARM SAVE FAIL"); return; } latched=true; saved.totalAlarms++;
    } coreYield(nullptr);
  }
  saved.replayNext++;
  if(latched || saved.replayNext%32==0 || saved.replayNext==info[m].nAcq) {
    csvSync(); checkpoint();
  }
  if(millis()-lastDisplay>150) screen("REPLAY",saved.replayNext,info[m].nAcq);
}
size_t helloFrame() {
  // Flags: 1 SD mounted, 2 self-check passed, 4 IMU source (PROTOCOL_V4 HELLO), 8 run complete.
  uint32_t v[]={board,43,uint32_t((TINYFL_SENSOR?4:1)|(allChecks?2:0)|(saved.complete?8:0)),saved.run};
  return p4::control(frame,p4::HELLO,p4::NoSlot,v,nullptr,positions,machines);
}
void queueReady() {
  if(saved.run && heardRun==saved.run) queue(p4::ready(frame,saved.run,saved.heldSlot),Ready);
  else queue(helloFrame(),HelloReady);
}
void work() {
  if(!joined || fatal || inFlight!=None || (LMIC.opmode&OP_TXRXPEND) || int32_t(millis()-nextTxAt)<0) return;
  if(saved.complete && !completedBootAnnounce) return;
  if(pendingAdoptHellos) { queue(helloFrame(),AdoptHello); return; }
  if(helloStage==0) {
    queue(helloFrame(),Hello); return;
  }
  if(helloStage<=machines) {
    int m=helloStage-1; uint32_t v[]={positions[m],info[m].check.rows};
    float x[]={info[m].check.maxAbs,info[m].check.medAbs};
    queue(p4::control(frame,p4::CHECK,p4::NoSlot,v,x,nullptr,0),Check); return;
  }
  if(!allChecks) { stop("SELF-CHECK FAILED"); return; }
  if(saved.complete) {
    if(millis()-lastActivity>=180000) queue(helloFrame(),HelloReady);
    return;
  }
  // An in-flight outbox can be repeated after reset; restore ring state first
  // so acknowledging a pending ALARM also updates its persisted sent flag.
  if(saved.replayStarted && !states && !startReplay()) return;
  if(saved.outLen) { memcpy(frame,saved.outFrame,saved.outLen); queue(saved.outLen,Outbox); return; }
  for(int m=0;m<machines;m++) for(int p=0;p<255;p++) if(hasBit(saved.doneRetry[m],p)) {
    char path[32]; snprintf(path,sizeof(path),"/done%u_%u",m,p);
    if(!blobRead(path,frame,24) || get_u32(frame+8)!=saved.run) { stop("DONE RETRY INVALID"); return; }
    saved.outMachine=m; stageOutbox(3,p,24); return;
  }
  for(int m=0;m<machines;m++) if(repairMask[m]) {
    if(!prepareUpdate(m)) return; txMachine=m;
    for(txFrag=0;txFrag<3;txFrag++) if(repairMask[m]&(1<<txFrag)) break;
    queue(p4::fragment(frame,updates[m],p4::UPDATE,saved.seedSlot,txFrag),Repair); return;
  }
  if(needStatus) {
    uint32_t v[]={saved.run,statusSlot,statusMask}; queue(p4::control(frame,p4::STATUS,statusSlot,v,nullptr,nullptr,0),Status); return;
  }
  if(!saved.complete && !saved.replayStarted && saved.heldSlot && saved.round && !saved.replayStopped) {
    for(int m=0;m<machines;m++) if((activeMachines&(1<<m)) && !(saved.sent&(1<<m))) {
      if(!prepareUpdate(m)) return; txMachine=m;
      queue(p4::fragment(frame,updates[m],p4::UPDATE,saved.seedSlot,txFrag),Update); return;
    }
  }
  if(needReady) { queueReady(); return; }
  if(waiting) {
    if(millis()-lastActivity>=(assembler.mask?120000u:180000u)) {
      if(assembler.mask) { statusSlot=assembler.slot; statusMask=7&~assembler.mask; needStatus=true; }
      else needReady=true; // periodic uplink; queue() transmits only when the duty cycle allows
    } return;
  }
  if(!saved.complete && !saved.replayStopped) replayStep(); else phase("idle");
}
#if TINYFL_SENSOR
// ------------------------------------------------------------ sensor-only build
bool sensorManifest() {
  static const uint16_t list[]={TINYFL_SENSOR_POSITIONS};
  board=TINYFL_SENSOR_BOARD; machines=0;
  if(!board || board>=65535 || sizeof(list)/sizeof(list[0])>8) return false;
  for(uint16_t p:list) {
    if(p>=65535) return false;
    for(int i=0;i<machines;i++) if(positions[i]==p) return false;
    positions[machines]=p; declaredStreams[machines++]=1;
  }
  for(int i=0;i<machines;i++) for(int j=i+1;j<machines;j++) if(positions[j]<positions[i]) {
    uint16_t p=positions[i]; positions[i]=positions[j]; positions[j]=p;
  } return machines>0;
}
// One live acquisition -> 22 raw features. Phase markers: sense (MPU6050 FIFO), features.
bool senseOnce(float raw[kFeat]) {
  phase("sense"); if(!sensor.capture()) return false;
  phase("features"); return sensorEx.extract(sensor,raw);
}
// Boot calibration: the history the SD cards used to carry, sensed now (train, then calibration).
bool sensorCalibrate(int m) {
  char name[16]; sensorLogName(name,positions[m]); if(cardFs.exists(name)) cardFs.remove(name);
  const uint32_t rows=TINYFL_SENSOR_TRAIN+TINYFL_SENSOR_CAL;
  for(uint32_t r=0;r<rows;r++) {
    float raw[kFeat];
    if(!senseOnce(raw) || !sensorLogAppend(&cardFs,positions[m],raw)) { stop(tf_last_error()); return false; }
    if(r%10==9 || r+1==rows) Serial.printf("CALIB p=%u %lu/%lu\n",positions[m],(unsigned long)(r+1),(unsigned long)rows);
    screen("CALIBRATE",r+1,rows);
  }
  phase("prepare");
  if(!sensorPrepare(&cardFs,positions[m],TINYFL_SENSOR_TRAIN,TINYFL_SENSOR_CAL,&info[m])) { stop(tf_last_error()); return false; }
  return true;
}
// Reboot: reuse the prepared rows if the stored history is still the one the run trained on.
bool sensorCached(int m) {
  TfMachineInfo cached; uint32_t crc=0;
  if(!saved.inputCrc[m] || !tf_load_prepared(&cardFs,positions[m],&cached,nullptr) || cached.nStreams!=1 ||
     cached.nTrain!=TINYFL_SENSOR_TRAIN || cached.part.calCount!=TINYFL_SENSOR_CAL ||
     !sensorHistoryCrc(&cardFs,positions[m],cached.nAcq,&crc) || crc!=saved.inputCrc[m]) return false;
  cached.inputCrc=crc; info[m]=cached; return true;
}
// Battery voltage on GPIO35 through the T3's 1:2 divider (check once against a multimeter).
// Kept in /batt.log (u32 millis, u16 mV per record; millis 0xFFFFFFFF marks a boot).
void battRecord(uint32_t ms,uint16_t mv) {
  File f=LittleFS.open("/batt.log",FILE_APPEND); if(!f) return;
  uint8_t b[6]; put_u32(b,ms); put_u16(b+4,mv); f.write(b,6); f.close();
}
uint32_t sensorLastSense=0,sensorLastBatt=0,sleepMs=0,sleepWakes=0;
void logBattery() {
  uint32_t mv=analogReadMilliVolts(35)*2u;
  mark("BATT %lu %lu\n",(unsigned long)mv,(unsigned long)millis()); battRecord(millis(),(uint16_t)mv);
#if TINYFL_LIGHT_SLEEP
  mark("SLEEP %lu %lu %lu\n",(unsigned long)sleepMs,(unsigned long)sleepWakes,(unsigned long)millis());
#endif
}
#if TINYFL_LIGHT_SLEEP
// Light sleep while the board only listens. The SX1276 keeps receiving on its own (Class C
// continuous RX) and raises DIO0 on RxDone; DIO0 high wakes the ESP32 (GPIO level wake-up) and
// LMIC then polls the pin exactly as before. A timer bounds every sleep (<= 1 s, and never past
// our own next deadline), so periodic READY/STATUS, sensing and the battery log stay on time.
// Never: during a TX or its RX1/RX2 windows, with an LMIC job due, with a radio IRQ pending,
// with unprocessed downlinks, or before the join.
bool radioQuiet(uint32_t ms) {
  return radio && joined && !fatal && inFlight==None && !txDone && tail==head &&
         !(LMIC.opmode&(OP_TXRXPEND|OP_TXDATA|OP_JOINING|OP_POLL)) &&
         !os_queryTimeCriticalJobs(ms2osticks(ms+20)) &&
         !digitalRead(lmic_pins.dio[0]) && !digitalRead(lmic_pins.dio[1]);
}
void lightSleepMs(uint32_t ms) {
  Serial.flush(); Serial2.flush();               // an unflushed UART byte would be garbled
  esp_sleep_enable_timer_wakeup((uint64_t)ms*1000u);
  gpio_wakeup_enable((gpio_num_t)lmic_pins.dio[0],GPIO_INTR_HIGH_LEVEL);
  esp_sleep_enable_gpio_wakeup();
  uint32_t t=millis(); esp_light_sleep_start(); sleepMs+=millis()-t; sleepWakes++;
}
// Called at the end of loop(); true if it slept (then loop() skips its delay(1)).
bool maybeSleep() {
  if(!sensorReady || saved.replayStarted) return false;
  uint32_t now=millis(),ms=1000;
  auto until=[&](uint32_t last,uint32_t period) {
    if(!last) { ms=0; return; }                  // first sensing/battery run not done yet
    uint32_t due=last+period; int32_t left=(int32_t)(due-now); if(left<(int32_t)ms) ms=left>0?left:0;
  };
  until(sensorLastBatt,TINYFL_BATT_LOG_S*1000u);
  if(!saved.complete) until(sensorLastSense,TINYFL_SENSE_INTERVAL_S*1000u);
  if(ms<10 || !radioQuiet(ms)) return false;
  lightSleepMs(ms); return true;
}
// During a live acquisition: the MPU6050 FIFO (170 ms at 1 kHz) fills while the ESP32 sleeps.
void sensorIdle(uint32_t ms) { coreYield(nullptr); if(radioQuiet(ms)) lightSleepMs(ms); }
#endif
// Serial commands while running (the port is opened without reset): BATTLOG dumps the log.
void sensorCommands() {
  static char line[16]; static uint8_t n=0;
  while(Serial.available()) {
    char c=Serial.read();
    if(c!='\n' && c!='\r') { if(n<sizeof(line)-1) line[n++]=c; continue; }
    line[n]=0; n=0;
    if(!strcmp(line,"BATTLOG")) {
      File f=LittleFS.open("/batt.log",FILE_READ); uint8_t b[6]; uint32_t i=0;
      Serial.println("BATTLOG BEGIN");
      while(f && f.read(b,6)==6) Serial.printf("BATT_REC %lu %lu %u\n",(unsigned long)i++,(unsigned long)get_u32(b),get_u16(b+4));
      if(f) f.close(); Serial.println("BATTLOG END");
    } else if(!strcmp(line,"BATTCLEAR")) { LittleFS.remove("/batt.log"); Serial.println("BATTLOG CLEARED"); }
  }
}
// Live monitoring: one acquisition per machine every TINYFL_SENSE_INTERVAL_S, appended to the
// feature log and scored with the held GLOBAL. Never during a TX/RX window, never once a
// replay is requested (the replay then scores a fixed log).
void sensorTick() {
  uint32_t& lastSense=sensorLastSense; uint32_t& lastBatt=sensorLastBatt;
  if(!sensorReady) return;
  sensorCommands();
  if(!lastBatt || millis()-lastBatt>=TINYFL_BATT_LOG_S*1000u) { lastBatt=millis()|1; logBattery(); }
  if(fatal || !joined || saved.complete || saved.replayStarted || inFlight!=None || (LMIC.opmode&OP_TXRXPEND)) return;
  for(int i=0;i<32;i++) if(saved.requested[i]) return;
  if(lastSense && millis()-lastSense<TINYFL_SENSE_INTERVAL_S*1000u) return;
  lastSense=millis()|1;
  // Never during a TX/RX window (checked above), so afterwards the radio is either in Class C
  // continuous reception or idle. Restoring the previous marker instead mislabelled the time
  // after a live acquisition as "tx" when the last marker was a TX start (bench, 2026-10-14).
  const char* resume=LMIC.classC.flags.f.fEnabled?"rx_classc":"idle";
  for(int m=0;m<machines;m++) {
    float raw[kFeat];
    if(!senseOnce(raw) || !sensorLogAppend(&cardFs,positions[m],raw)) {
      Serial.printf("WARN: live acquisition p=%u failed: %s\n",positions[m],tf_last_error()); continue;
    }
    phase("score");
    if(saved.heldSlot) {
      float x[kFeat]; tf_std_apply(raw,info[m].std.centre[0],info[m].std.scale[0],x);
      mark("SCORE %u %lu %.6g %lu\n",positions[m],(unsigned long)(sensorLogRows(&cardFs,positions[m])-1),
           tf_score_row(baseWeights,x),(unsigned long)millis());
    }
  }
  phase(resume);
}
#endif
}
void os_getArtEui(u1_t* out) { tinyfl::v4::lmicEui(out,tinyfl_node::keys.join()); }
void os_getDevEui(u1_t* out) { tinyfl::v4::lmicEui(out,tinyfl_node::devEuiMsb); }
void os_getDevKey(u1_t* out) { memcpy(out,tinyfl_node::keys.app(),16); }
void onEvent(ev_t ev) {
  using namespace tinyfl_node;
  if(ev==EV_JOINED) joined=true;
  if(ev==EV_JOIN_FAILED || ev==EV_REJOIN_FAILED || ev==EV_TXCANCELED) joinFailed=true;
  if(ev==EV_TXCOMPLETE) txDone=true;
  if(ev==EV_TXSTART) phase("tx");
  if((ev==EV_TXCOMPLETE || ev==EV_RXCOMPLETE) && LMIC.dataLen && LMIC.dataBeg && LMIC.frame[LMIC.dataBeg-1]==10)
    copyDownlink(LMIC.frame+LMIC.dataBeg,LMIC.dataLen);
}
void setup() {
  using namespace tinyfl_node;
  Serial.begin(115200); delay(500); enableLoopWDT(); Wire.begin(21,22); oled=display.begin(SSD1306_SWITCHCAPVCC,0x3c);
#if TINYFL_SENSOR
  // A deployed sensor has no display: switch the panel off (SSD1306 sleep, <= 10 uA).
#if TINYFL_CPU_MHZ!=240
  // Before the UARTs are configured: their baud dividers follow the APB clock at begin().
  if(!setCpuFrequencyMhz(TINYFL_CPU_MHZ)) Serial.println("WARN: CPU frequency change refused");
  Serial.updateBaudRate(115200);
#endif
  Serial2.begin(9600,SERIAL_8N1,TINYFL_MARKER_RX_PIN,TINYFL_MARKER_TX_PIN);
  Serial.printf("CPU %lu MHz\n",(unsigned long)getCpuFrequencyMhz());
  if(oled) { display.ssd1306_command(SSD1306_DISPLAYOFF); oled=false; }
  Wire.setClock(400000);
#endif
  if(!p4::loadKeys(keys)) { stop("NOT PROVISIONED"); return; }
#if TINYFL_SENSOR
  tf_set_yield(coreYield,nullptr); tf_set_progress(progress,nullptr); phase("boot");
  if(!sensor.begin(Wire)) { stop(tf_last_error()); return; }
  Serial.printf("SENSOR MPU6050 who=0x%02x samples=%lu rate_hz=1000 channels=3 interval_s=%u\n",sensor.whoAmI(),
                (unsigned long)Mpu6050Source::kSamples,(unsigned)TINYFL_SENSE_INTERVAL_S);
  if(!sensorManifest()) { stop("SENSOR CONFIG INVALID"); return; }
#else
  tf_set_yield(coreYield,nullptr); tf_set_progress(progress,nullptr); phase("sd_read"); screen("SD");
#if TINYFL_SD_MMC
  pinMode(13,OUTPUT); digitalWrite(13,HIGH);
  if(!SD_MMC.begin("/sdcard",true,false,SDMMC_FREQ_DEFAULT)) { stop("SD MMC MOUNT FAIL"); return; } card=&SD_MMC;
#else
  // An ESP32 reset does not power-cycle the card: after a reset in the middle of a read the
  // card can still be busy and miss the first init. Clock it out with CS high and retry,
  // at 4 MHz.
  pinMode(13,OUTPUT); digitalWrite(13,HIGH); sdSpi.begin(14,2,15,13);
  bool sdOk=false;
  for(int attempt=0;attempt<8 && !sdOk;attempt++) {
#if TINYFL_SDFAT
    // sdfat.end() stops sdSpi, and SdFat's begin() then restarts it with the HSPI default pins
    // (MISO 12, MOSI 13 = our CS), so every retry failed: restart the bus on the card's pins.
    if(attempt) { sdfat.end(); sdSpi.begin(14,2,15,13); delay(250); }
#else
    if(attempt) { SD.end(); delay(250); }
#endif
    sdUnstick();
#if TINYFL_SDFAT
    // Fixed 4 MHz (more signal margin on these boards; the earlier failures at a moving spot on
    // card 09 were a wedged card, handled by sdCardReinit()).
    sdOk=sdfat.begin(SdSpiConfig(13,DEDICATED_SPI,SD_SCK_MHZ(4),&sdSpi));
    if(!sdOk) Serial.printf("WARN: SdFat mount attempt %d failed (error 0x%02x data 0x%02x)\n",attempt+1,
                            sdfat.sdErrorCode(),sdfat.sdErrorData());
#else
    sdOk=SD.begin(13,sdSpi,(attempt%2)?4000000:10000000,"/sdcard");
#endif
    feedLoopWDT();
  }
  if(!sdOk) { stop("SD SPI MOUNT FAIL"); return; }
#if !TINYFL_SDFAT
  card=&SD;
#endif
#endif
  if(!readCardManifest()) { stop("V4 CARD INVALID"); return; }
#endif
  if(!LittleFS.begin(false)) { stop("PROVISION LITTLEFS"); return; }
  bool restored=loadProgress(); if(fatal) return;
  if(restored && !finishArchive()) return;
  if(!restored) { saved.board=board; saved.machines=machines; memcpy(saved.positions,positions,2*machines); }
#if TINYFL_SENSOR
  battRecord(0xffffffffu,0);
  for(int m=0;m<machines;m++) {
    bool cached=restored && sensorCached(m);
    if(!cached && !sensorCalibrate(m)) return;
    // No reference features exist for live data: the check is that the sensor answered and
    // the whole history was sensed with finite features (CHECK reports -1/-1 = no reference).
    info[m].check.maxAbs=-1; info[m].check.medAbs=-1;
    Serial.printf("SELF_CHECK p=%u rows=%u sensor history PASS%s\n",positions[m],info[m].check.rows,cached?" cached":"");
    if(restored && saved.inputCrc[m]!=info[m].inputCrc) { stop("SENSOR HISTORY CHANGED"); return; }
    saved.inputCrc[m]=info[m].inputCrc;
  }
  sensorReady=true;
#if TINYFL_LIGHT_SLEEP
  sensor.idleHook=sensorIdle;
  Serial.println("LIGHT_SLEEP on: ESP32 sleeps while listening (wake: SX1276 DIO0 or timer)");
#endif
#else
  for(int m=0;m<machines;m++) {
    bool cached=restored && cachedMachine(m);
    if(!cached) {
      TfExtractor ex; phase("features"); screen("SELF-CHECK",m,machines);
      if(!tf_prepare_machine(&cardFs,positions[m],ex,&info[m]) || info[m].nStreams!=declaredStreams[m]) { stop(tf_last_error()); return; }
    }
    bool pass=info[m].check.compared && info[m].check.maxAbs<=1e-3f; allChecks&=pass;
    Serial.printf("SELF_CHECK p=%u rows=%u max=%.9g median=%.9g %s%s\n",positions[m],info[m].check.rows,
                  info[m].check.maxAbs,info[m].check.medAbs,pass?"PASS":"FAIL",cached?" cached":"");
    if(restored && saved.inputCrc[m]!=info[m].inputCrc) { stop("RAW CARD CHANGED"); return; }
    saved.inputCrc[m]=info[m].inputCrc;
  }
#endif
  checkpoint(); uint64_t mac=ESP.getEfuseMac(); uint32_t seed=0x811c9dc5;
  for(int i=0;i<6;i++) devEuiMsb[i+2]=mac>>(8*(5-i));
  Serial.print("DevEUI: "); for(int i=0;i<8;i++) { Serial.printf("%02X",devEuiMsb[i]); seed=(seed^devEuiMsb[i])*0x01000193; } Serial.println();
  if(!SPI.begin(5,19,27,18) || !os_init_ex(&lmic_pins)) { stop("RADIO INIT FAIL"); return; }
  radio=true; LMIC_reset(); LMIC_setClockError(MAX_CLOCK_ERROR*4u/1000u);
  Preferences p; char key[16]; snprintf(key,sizeof(key),"fcnt%08lX",(unsigned long)keys.devaddr());
  bool have=false; uint32_t counter=0;
  if(p.begin("tinyflmc",true)) { have=p.isKey(key); counter=p.getULong(key,0); p.end(); }
  TinyFLMulticast::begin(keys.devaddr(),keys.nwk(),keys.mcapp(),counter,have);
  nextTxAt=seed%40; phase("idle"); screen("JOIN"); LMIC_startJoining();
}
void loop() {
  using namespace tinyfl_node;
  if(fatal) { delay(20); return; } if(radio) os_runloop_once();
  if(joinFailed) { stop("JOIN/TX FAILURE"); return; }
  static bool configured=false;
  if(joined && !configured) {
    configured=true; LMIC_setLinkCheckMode(0); LMIC_setAdrMode(0); LMIC_setDrTxpow(DR_SF7,14);
    nextTxAt=millis()+1000*nextTxAt; restoreBase();
  }
  uint8_t multicast[221],port; size_t n;
  while(TinyFLMulticast::poll(multicast,sizeof(multicast),&n,&port)) if(port==10) copyDownlink(multicast,n);
  static bool haveCounter=false; static uint32_t lastCounter=0;
  if(TinyFLMulticast::hasCounter() && (!haveCounter || lastCounter!=TinyFLMulticast::frameCounter())) {
    Preferences p; char key[16]; snprintf(key,sizeof(key),"fcnt%08lX",(unsigned long)keys.devaddr());
    if(!p.begin("tinyflmc",false) || p.putULong(key,TinyFLMulticast::frameCounter())!=4) { stop("MC COUNTER SAVE FAIL"); return; }
    p.end(); haveCounter=true; lastCounter=TinyFLMulticast::frameCounter();
  }
  // Complete the old TX before applying a GLOBAL that may change its queue.
  if(txDone) { txDone=false; finishTx(); }
  while(tail!=head) { processDownlink(down[tail],downLen[tail]); tail=(tail+1)%4; }
#if TINYFL_SENSOR
  sensorTick();
#endif
  work();
#if TINYFL_SENSOR && TINYFL_LIGHT_SLEEP
  if(maybeSleep()) return;
#endif
  delay(1);
}
