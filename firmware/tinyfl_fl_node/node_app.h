#pragma once
#include <Arduino.h>
#include <FS.h>
#include <SD.h>
#include <SD_MMC.h>
#include <LittleFS.h>
#include <Preferences.h>
#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <lmic.h>
#include <hal/hal.h>
#include <TinyFLMulticast.h>
#include <unistd.h>
#include <esp_heap_caps.h>
#include "../common/tinyfl_core/tinyfl_core_all.cpp"
#include "../common/tinyfl_imu_source.h"

#if __has_include("lab_credentials.h")
#include "lab_credentials.h"
#elif __has_include("../tinyfl_node/lab_credentials.h")
#include "../tinyfl_node/lab_credentials.h"
#else
#include "lab_credentials.example.h"
#endif
#ifndef TINYFL_PHASE_MARKERS
#define TINYFL_PHASE_MARKERS 1
#endif
#ifndef TINYFL_SD_MMC
#define TINYFL_SD_MMC 1
#endif
#ifndef TINYFL_RUN_LEVEL_MASK
#define TINYFL_RUN_LEVEL_MASK 0x7f
#endif
#ifndef TINYFL_MULTICAST_DEVADDR
#define TINYFL_MULTICAST_DEVADDR 0
#endif
#if !LMIC_ENABLE_class_c || !defined(CFG_eu868) || !defined(CFG_sx1276_radio)
#error "Use this sketch's build_opt.h: EU868 SX1276 Class C is required"
#endif
static_assert(MAX_FRAME_LEN >= 255, "LMIC long messages required");
static_assert(TINYFL_RUN_LEVEL_MASK > 0 && TINYFL_RUN_LEVEL_MASK <= 127, "Invalid sweep mask");
const lmic_pinmap lmic_pins = {18, LMIC_UNUSED_PIN, 23, {26,33,32}, 0, 10, 8000000, nullptr};

namespace tinyfl_node {
using namespace tinyfl;
constexpr uint8_t kLevels[7] = {4,8,12,16,20,24,28};
static const u1_t PROGMEM joinEui[8] = TINYFL_JOIN_EUI;
static const u1_t PROGMEM appKey[16] = TINYFL_APP_KEY;
uint8_t devEui[8];
Adafruit_SSD1306 display(128,64,&Wire,-1);
SPIClass sdSpi(HSPI);
fs::FS* card = nullptr;
bool oled=false, radio=false, joined=false, txDone=false, joinFailed=false;
bool fatal=false, allChecks=true;
const char* phaseName=nullptr;
uint8_t board=0, machines=0, positions[3]={};
uint8_t active[7]={};
TfMachineInfo info[3];
uint32_t alarms=0, lastDisplay=0, lastActivity=0, nextTxAt=0;
uint8_t staggerSeconds=0;
char screenName[48]="SD";

void phase(const char* name) {
  if (phaseName && !strcmp(phaseName,name)) return;
  phaseName=name;
#if TINYFL_PHASE_MARKERS
  Serial.printf("PHASE %s %lu\n",name,(unsigned long)millis());
#endif
}
void screen(const char* name,uint32_t done=0,uint32_t total=1) {
  snprintf(screenName,sizeof(screenName),"%s",name);
  if (!oled) return;
  display.clearDisplay(); display.setTextColor(SSD1306_WHITE); display.setTextSize(1);
  display.setCursor(0,0); display.printf("TinyFL board %02u\n",board);
  display.drawFastHLine(0,11,128,SSD1306_WHITE); display.setCursor(0,15);
  display.println(screenName); display.printf("%lu / %lu\n",(unsigned long)done,(unsigned long)total);
  display.printf("Alarms %lu\n",(unsigned long)alarms);
  display.drawRect(0,53,128,10,SSD1306_WHITE);
  uint32_t w=total ? (uint64_t)126*min(done,total)/total : 0;
  display.fillRect(1,54,w,8,SSD1306_WHITE); display.display(); lastDisplay=millis();
}
void stop(const char* reason) {
  Serial.printf("ERROR: %s\n",reason); fatal=true;
  if (radio) LMIC_enableClassC(0);
  phase("idle"); screen(reason,0,1);
}
void coreYield(void*) {
  // Event callbacks only copy receive bytes/set flags; no core, flash or SD
  // work is re-entered while a portable-core operation is running.
  if (radio) os_runloop_once();
  static uint32_t lastYield=0;
  if(millis()-lastYield>=10) { delay(1); lastYield=millis(); }
}
void progress(void*,const char* stage,uint32_t done,uint32_t total) {
  if(!strcmp(stage,"sd_read") || !strcmp(stage,"features") || !strcmp(stage,"replay")) { phase(stage); return; }
  if (millis()-lastDisplay >= 150 || done==total) screen(stage,done,total);
}

class ArduinoFile : public TfFile {
 public:
  explicit ArduinoFile(File f): f_(f) {}
  ~ArduinoFile() override { f_.close(); }
  int64_t read(void* d,size_t n) override { return f_.read((uint8_t*)d,n); }
  size_t write(const void* d,size_t n) override { return f_.write((const uint8_t*)d,n); }
  bool seek(uint64_t p) override { return p<=UINT32_MAX && f_.seek((uint32_t)p); }
  uint64_t size() override { return f_.size(); }
  uint64_t position() override { return f_.position(); }
  void flush() override { f_.flush(); }
 private: File f_;
};
class CardFs : public TfFileSystem {
 public:
  TfFile* open(const char* name,TfOpenMode mode) override {
    String p=String("/")+name;
    File f=card->open(p,mode==kOpenRead?FILE_READ:(mode==kOpenWrite?FILE_WRITE:FILE_APPEND));
    return f ? new ArduinoFile(f) : nullptr;
  }
  bool exists(const char* name) override { return card->exists(String("/")+name); }
  bool remove(const char* name) override { return card->remove(String("/")+name); }
} cardFs;

// Two CRC-protected generation slots: a torn write leaves the previous slot.
// Includes base GLOBAL, machine queue, replay ring state and CSV offset.
struct Progress {
  uint32_t magic=0x334c4654, generation=0;
  uint16_t run=0;
  uint8_t board=0, nMachines=0, positions[3]={}, sweep=TINYFL_RUN_LEVEL_MASK;
  uint8_t finalMask=0, haveBase=0, baseSlot=0, targetRound=0, sentMachines=0;
  uint8_t replayMachine=0, replayDoneMask=0, replayStarted=0, complete=0;
  uint8_t resetPending=0;
  uint32_t replayNext=0, csvOffset=0, totalAlarms=0;
  uint32_t inputCrc[3]={};
  uint8_t base[kGlobalBytes]={};
  TfAlarmState alarm[7]={};
  // Durable radio outbox: replay stops until ALARM/REPLAY_DONE is transmitted.
  uint8_t pendingLen=0, pendingKind=0, pendingModel=0, pendingFrame[32]={};
} saved;
static_assert(sizeof(Progress)<4096,"Checkpoint unexpectedly large");
uint8_t level=0, waitSlot=0, statusSlot=0;
bool trainPending=false, waiting=false, needReady=false, needStatus=false, needAdvance=false;
int logRound=-1;
void roundMarker(int idx,int round) {
  int id=idx*13+round;
  if(id==logRound) return; logRound=id;
#if TINYFL_PHASE_MARKERS
  if(round==12) Serial.printf("ROUND end %lu\n",(unsigned long)millis());
  else Serial.printf("ROUND run%u-L%d:R%d %lu\n",saved.run,idx,round+1,(unsigned long)millis());
#endif
}
uint16_t statusMask=7;
TfGlobalReassembler assembler;
TfGlobalReassembler restartCandidate;
float baseWeights[kNumParams];
uint8_t updates[3][kUpdateBytes]; // retain all active machines' deltas for repairs
uint8_t updateValid=0, repairMask[3]={};
enum TxKind : uint8_t {None,Hello,Check,Update,Repair,Ready,Status,Outbox};
TxKind inFlight=None;
uint8_t txMachine=0, txFrag=0, helloStage=0;
uint8_t frame[kMaxFrame];
size_t frameLen=0;
uint8_t downlink[4][kMaxFrame], downLen[4]={};
uint8_t downHead=0, downTail=0;
uint32_t rxOverruns=0;
TfReplay* replay=nullptr;
TfReplayModel* replayModels=nullptr;
int replayModelCount=0;
File csv;
bool blobRead(const char* path,void* data,size_t n) {
  File f=LittleFS.open(path,FILE_READ); uint8_t tail[4];
  if (!f || f.size()!=n+4 || f.read((uint8_t*)data,n)!=n || f.read(tail,4)!=4) return false;
  return get_u32(tail)==tf_crc32(0,data,n);
}
bool blobWrite(const char* path,const void* data,size_t n) {
  String tmp=String(path)+".tmp"; File f=LittleFS.open(tmp,FILE_WRITE); uint8_t tail[4];
  put_u32(tail,tf_crc32(0,data,n));
  bool ok=f && f.write((const uint8_t*)data,n)==n && f.write(tail,4)==4;
  f.flush(); f.close();
  if (!ok) return false;
  // LittleFS rename is atomic, including replacement of an existing file.
  return LittleFS.rename(tmp,path);
}
bool checkpoint() {
  saved.generation++;
  const char* path=(saved.generation&1)?"/progress1":"/progress0";
  if (!blobWrite(path,&saved,sizeof(saved))) { stop("FLASH WRITE FAIL"); return false; }
  return true;
}
bool loadProgress() {
  Progress* a=new Progress; Progress* b=new Progress;
  bool av=blobRead("/progress0",a,sizeof(*a)) && a->magic==0x334c4654;
  bool bv=blobRead("/progress1",b,sizeof(*b)) && b->magic==0x334c4654;
  if (av || bv) saved=(!av || (bv && (int32_t)(b->generation-a->generation)>0))?*b:*a;
  delete a; delete b;
  if (!(av || bv)) return false;
  if(saved.pendingLen>32 || saved.replayMachine>machines || saved.targetRound>12 ||
     saved.finalMask>127 || saved.sentMachines>7 || saved.replayDoneMask>7 || saved.resetPending>1) { stop("BAD SAVED STATE"); return false; }
  if (saved.board!=board || saved.nMachines!=machines || memcmp(saved.positions,positions,machines) ||
      saved.sweep!=TINYFL_RUN_LEVEL_MASK) { stop("RUN/CARD CONFIG CHANGED"); return false; }
  alarms=saved.totalAlarms;
  return true;
}
void modelPath(char* p,uint8_t i) { snprintf(p,24,"/model_L%u",i); }
int firstLevel() { for(int i=0;i<7;i++) if(TINYFL_RUN_LEVEL_MASK&(1<<i)) return i; return -1; }
int nextLevel(int i) { for(++i;i<7;i++) if(TINYFL_RUN_LEVEL_MASK&(1<<i)) return i; return -1; }
void releaseReplay() {
  if(replay) delete replay; replay=nullptr;
  if(replayModels) delete[] replayModels; replayModels=nullptr;
  replayModelCount=0; if(csv) csv.close();
}
bool finishRunReset() {
  for(int i=0;i<7;i++) { char p[24]; modelPath(p,i); LittleFS.remove(p); }
  for(int i=0;i<7;i++) { char p[24]; snprintf(p,sizeof(p),"/alarm%u",i); LittleFS.remove(p); }
  for(int i=0;i<3;i++) { char p[24]; snprintf(p,sizeof(p),"/update%u",i); LittleFS.remove(p); }
  // Only the derived run CSV is replaced; input card files remain untouched.
  if(card->exists("/results.csv")) {
    String backup=String("/results_")+millis()+".csv";
    if(!card->rename("/results.csv",backup)) { stop("CSV ARCHIVE FAIL"); return false; }
  }
  saved.resetPending=0; return checkpoint();
}
void resetRun(uint16_t run) {
  uint32_t generation=saved.generation;
  logRound=-1;
  releaseReplay(); saved=Progress(); saved.generation=generation; alarms=0;
  saved.run=run; saved.board=board; saved.nMachines=machines;
  memcpy(saved.positions,positions,machines); saved.sweep=TINYFL_RUN_LEVEL_MASK;
  for(int m=0;m<machines;m++) saved.inputCrc[m]=info[m].inputCrc;
  // Commit the empty new-run identity BEFORE deleting the old run's files.
  // A reboot resumes idempotent cleanup, never a partially destroyed old run.
  saved.resetPending=1;
  if(!checkpoint() || !finishRunReset()) return;
  memset(repairMask,0,sizeof(repairMask)); updateValid=0; trainPending=false;
  waiting=false; needAdvance=false; needStatus=false; needReady=false; assembler.reset();
  restartCandidate.reset();
  checkpoint();
}
void expectGlobal(uint8_t slot) {
  waitSlot=slot; waiting=true; needReady=true; assembler.reset(); lastActivity=millis();
  screen("WAIT GLOBAL",0,3);
}

bool readCardManifest() {
  File f=card->open("/board.txt");
  if(!f) return false;
  unsigned b=0,v=0,n=0;
  String line=f.readStringUntil('\n'); if(sscanf(line.c_str(),"board %u",&b)!=1 || b<1 || b>10) return false;
  line=f.readStringUntil('\n'); if(sscanf(line.c_str(),"format_version %u",&v)!=1 || v!=1) return false;
  line=f.readStringUntil('\n'); if(sscanf(line.c_str(),"machines_hosted %u",&n)!=1 || n<1 || n>3) return false;
  board=b; machines=n;
  while(f.available()) {
    line=f.readStringUntil('\n'); line.trim(); if(!line.length() || line.startsWith("position,")) continue;
    unsigned p=0; if(sscanf(line.c_str(),"%u,",&p)!=1 || p>53 || p%10+1!=board) return false;
    unsigned index=0; while(index<3 && positions[index]!=255) ++index;
    if(index>=machines) return false;
    for(unsigned j=0;j<index;j++) if(positions[j]==p) return false;
    positions[index]=p;
  }
  for(int i=0;i<machines;i++) if(positions[i]==255 || (i && positions[i]<=positions[i-1])) return false;
  f.close(); f=card->open("/levels.csv"); if(!f) return false;
  line=f.readStringUntil('\n'); line.trim();
  if(line!="level,level_index,hardware,active_positions,n_active") return false;
  uint8_t seen=0;
  while(f.available()) {
    line=f.readStringUntil('\n'); line.trim(); if(!line.length()) continue;
    int cut[4],last=-1; for(int i=0;i<4;i++) { cut[i]=line.indexOf(',',last+1); if(cut[i]<0) return false; last=cut[i]; }
    unsigned L=0,idx=0,hw=0,count=0;
    if(sscanf(line.c_str(),"%u,%u,%u",&L,&idx,&hw)!=3) return false;
    if(idx>=7) { if(hw) return false; continue; }
    if(hw!=1 || L!=kLevels[idx] || (seen&(1<<idx))) return false;
    seen|=1<<idx; String names=line.substring(cut[2]+1,cut[3]);
    String suffix=line.substring(cut[3]+1); if(sscanf(suffix.c_str(),"%u",&count)!=1) return false;
    uint8_t mask=0; int at=0;
    while(at<(int)names.length()) {
      int end=names.indexOf(';',at); if(end<0) end=names.length();
      String token=names.substring(at,end); unsigned p=0;
      if(sscanf(token.c_str(),"%u",&p)!=1) return false;
      int m=0; while(m<machines && positions[m]!=p) ++m;
      if(m==machines || p>=L || (mask&(1<<m))) return false;
      mask|=1<<m; at=end+1;
    }
    if(__builtin_popcount(mask)!=count) return false;
    for(int m=0;m<machines;m++) if(bool(mask&(1<<m)) != (positions[m]<L)) return false;
    active[idx]=mask;
  }
  return seen==127;
}

void copyDownlink(const uint8_t* data,size_t n) {
  if(n>kMaxFrame) return;
  uint8_t next=(downHead+1)%4; if(next==downTail) { rxOverruns++; return; }
  memcpy(downlink[downHead],data,n); downLen[downHead]=n; downHead=next;
}
void restoreBase() {
  if(!saved.haveBase) { level=firstLevel(); expectGlobal(tf_slot(level,0)); return; }
  TfGlobalView g;
  if(!tf_decode_global(saved.base,&g) || !g.runId || g.runId!=saved.run ||
      tf_slot(g.levelIndex,g.round)!=saved.baseSlot) { stop("SAVED GLOBAL INVALID"); return; }
  level=g.levelIndex; tf_dequantise(g.codes,g.scale,baseWeights);
  if(saved.complete) {
    trainPending=false; waiting=false; needReady=false; needAdvance=false;
    screen("DONE",1,1); return;
  }
  if(g.round<12) roundMarker(level,saved.targetRound);
  trainPending = !saved.complete && g.round<12 &&
                 (saved.sentMachines&active[level])!=active[level];
  if(g.round==12 && !saved.complete) needAdvance=true;
  else if(!trainPending) expectGlobal(tf_slot(level,saved.targetRound+1));
  for(int m=0;m<machines;m++) {
    char path[24]; snprintf(path,sizeof(path),"/update%u",m); TfUpdateView u;
    if(blobRead(path,updates[m],kUpdateBytes) && tf_decode_update(updates[m],&u) &&
       u.runId==saved.run && u.position==positions[m] && u.levelIndex==level && u.round==saved.targetRound &&
       u.baseSlot>=tf_slot(level,0) && u.baseSlot<=saved.baseSlot)
      updateValid|=1<<m;
  }
}
void acceptGlobal(const uint8_t* payload,uint8_t slot) {
  TfGlobalView g;
  if(!tf_decode_global(payload,&g) || !g.runId || tf_slot(g.levelIndex,g.round)!=slot ||
     !(TINYFL_RUN_LEVEL_MASK&(1<<g.levelIndex))) { Serial.println("Reject invalid GLOBAL"); assembler.reset(); return; }
  if(g.runId==saved.run) {
    if(saved.haveBase && slot<=saved.baseSlot) { Serial.println("Ignore stale/duplicate GLOBAL"); return; }
    // Do not silently skip a final level model; request its missing fragments.
    if(g.levelIndex!=level && !(saved.finalMask&(1<<level))) { Serial.println("Missing final GLOBAL: keep requesting it"); return; }
  } else {
    Serial.printf("New run %u: discarding prior run models/progress\n",g.runId);
    resetRun(g.runId); if(fatal) return;
  }
  bool sameTrainingRound=saved.haveBase && g.levelIndex==level && g.round==saved.targetRound && g.round<12;
  level=g.levelIndex; saved.haveBase=1; saved.baseSlot=slot;
  roundMarker(level,g.round);
  memcpy(saved.base,payload,kGlobalBytes); tf_dequantise(g.codes,g.scale,baseWeights);
  if(!sameTrainingRound) {
    txFrag=0; saved.sentMachines=0; updateValid=0; memset(repairMask,0,sizeof(repairMask));
  }
  // A late full seed GLOBAL must not change already-cached UPDATE bytes:
  // partial retransmissions are immutable, including their actual base_slot.
  saved.targetRound=g.round;
  waiting=false; needReady=false;
  trainPending=g.round<12 && (saved.sentMachines&active[level])!=active[level];
  if(g.round<12 && !trainPending) expectGlobal(tf_slot(level,g.round+1));
  if(g.round==12) {
    char path[24]; modelPath(path,level);
    if(!blobWrite(path,payload,kGlobalBytes)) { stop("MODEL SAVE FAIL"); return; }
    saved.finalMask|=1<<level; trainPending=false; needAdvance=true;
  }
  checkpoint();
  // First acknowledge the received GLOBAL (status 0), then queue local work
  // or READY for the next selected level. Never compute in an LMIC callback.
  needStatus=true; statusMask=0; statusSlot=slot;
  phase("idle"); char text[48]; snprintf(text,sizeof(text),"LEVEL %u ROUND %u/12",kLevels[level],g.round);
  screen(text,g.round,12);
}
void processDownlink(const uint8_t* data,size_t n) {
  TfFrameHeader h; const uint8_t* body;
  if(!tf_parse_frame(data,n,&h,&body)) return;
  if(h.type==kMsgUpdateRepair) {
    uint8_t slot,p; uint16_t mask;
    if(!tf_parse_repair(data,n,&slot,&mask,&p) || !saved.haveBase || slot/13!=level ||
       slot%13<1 || slot%13>12 || saved.baseSlot%13==12) return;
    int m=0; while(m<machines && positions[m]!=p) m++;
    if(m==machines || !(active[level]&(1<<m))) return;
    uint8_t target=slot%13-1;
    if(target<saved.targetRound || saved.baseSlot>slot-1) return;
    if(target>saved.targetRound) {
      // A repair slot is an explicit server round indication. It permits
      // stale-copy training; a timeout alone is NOT a round clock.
      saved.targetRound=target; saved.sentMachines=0; updateValid=0; trainPending=true;
      txFrag=0;
      roundMarker(level,target);
      memset(repairMask,0,sizeof(repairMask)); checkpoint();
    }
    repairMask[m]|=mask; waiting=false; needReady=false;
    LMIC_enableClassC(0); phase("idle"); return;
  }
  if(h.type!=kMsgGlobalFragment || h.roundId>90) return;
  // Older-slot traffic cannot replace the active burst. A separate candidate
  // allows a restarted run's lower slots to arrive in ANY fragment order;
  // its run ID is trusted only once the entire payload passes its CRC.
  if(saved.haveBase && h.roundId<=saved.baseSlot) {
    if(h.index==0 && h.bodyLen>=4 && get_u16(body+2)==saved.run) {
      if(h.roundId==saved.baseSlot && h.bodyLen==214 && !memcmp(saved.base,body,214)) {
        needStatus=true; statusMask=0; statusSlot=h.roundId;
      }
      return;
    }
    auto result=restartCandidate.add(data,n);
    if(result==TfGlobalReassembler::kComplete) {
      TfGlobalView g;
      if(tf_decode_global(restartCandidate.payload(),&g) && g.runId && g.runId!=saved.run)
        acceptGlobal(restartCandidate.payload(),h.roundId);
      restartCandidate.reset();
    }
    return;
  }
  uint8_t oldMask=assembler.mask(); auto result=assembler.add(data,n);
  if(result==TfGlobalReassembler::kIgnored) { Serial.println("Rejected GLOBAL shape/conflicting duplicate"); return; }
  lastActivity=millis(); screen("WAIT GLOBAL",__builtin_popcount(assembler.mask()),3);
  if(result==TfGlobalReassembler::kComplete) acceptGlobal(assembler.payload(),h.roundId);
  else if(assembler.mask()==oldMask && result==TfGlobalReassembler::kDuplicate) Serial.println("Duplicate GLOBAL fragment");
}

bool prepareUpdate(int m) {
  if(updateValid&(1<<m)) return true;
  float* x=nullptr; TfMachineInfo current;
  phase("sd_read"); screen("SD TRAIN ROWS",m,machines);
  if(!tf_load_prepared(&cardFs,positions[m],&current,&x)) { stop(tf_last_error()); return false; }
  phase("train"); char text[48]; snprintf(text,sizeof(text),"LEVEL %u ROUND %u/12",kLevels[level],saved.targetRound+1);
  screen(text,m,__builtin_popcount(active[level]));
  float loss=tf_build_update(x,current.nTrain,baseWeights,positions[m],level,saved.targetRound,
                             saved.run,saved.baseSlot,updates[m]);
  tf_free(x); phase("idle");
  if(!isfinite(loss)) { stop("NONFINITE TRAIN LOSS"); return false; }
  char path[24]; snprintf(path,sizeof(path),"/update%u",m);
  if(!blobWrite(path,updates[m],kUpdateBytes)) { stop("UPDATE SAVE FAIL"); return false; }
  updateValid|=1<<m;
  Serial.printf("UPDATE p=%u level=%u round=%u base_slot=%u loss=%.9g free=%u largest=%u core_peak=%u\n",
                positions[m],level,saved.targetRound,saved.baseSlot,loss,ESP.getFreeHeap(),
                heap_caps_get_largest_free_block(MALLOC_CAP_8BIT),(unsigned)tf_heap_peak());
  return true;
}
bool queue(size_t n,TxKind kind) {
  if(!n || n>kMaxFrame || LMIC.opmode&OP_TXRXPEND) return false;
  LMIC_enableClassC(0); LMIC_setDrTxpow(DR_SF7,14);
  auto error=LMIC_setTxData2_strict(kFPort,frame,n,0);
  if(error==LMIC_ERROR_TX_BUSY) return false;
  if(error!=LMIC_ERROR_SUCCESS) { stop("LMIC TX REJECTED"); return false; }
  inFlight=kind; phase("idle"); return true;
}
void enterWait() {
  if(LMIC.dn2Dr!=DR_SF7) { stop("RX2 NEEDS DR5"); return; }
  LMIC_enableClassC(1); waiting=true; phase("rx_classc"); screen("WAIT GLOBAL",__builtin_popcount(assembler.mask()),3);
  lastActivity=millis();
}
void finishTx() {
  TxKind done=inFlight; inFlight=None;
  switch(done) {
    case Hello: helloStage=1; break;
    case Check: helloStage++; break;
    case Update:
      if(++txFrag==3) { txFrag=0; saved.sentMachines|=1<<txMachine; checkpoint(); }
      break;
    case Repair:
      repairMask[txMachine]&=~(1<<txFrag); txFrag=0;
      if(!repairMask[txMachine]) saved.sentMachines|=1<<txMachine;
      checkpoint();
      if(!(repairMask[0]|repairMask[1]|repairMask[2])) {
        waitSlot=tf_slot(level,saved.targetRound+1); waiting=true; needReady=true;
      }
      break;
    case Ready: needReady=false; enterWait(); return;
    case Status:
      needStatus=false;
      if(statusMask || waiting) { needReady=true; }
      break;
    case Outbox:
      if(saved.pendingKind==1) saved.alarm[saved.pendingModel].sent=1;
      if(saved.pendingKind==2) {
        saved.replayDoneMask|=1<<saved.replayMachine; saved.replayMachine++;
        saved.replayStarted=0; saved.replayNext=0; releaseReplay();
      }
      saved.pendingLen=0; saved.pendingKind=0; checkpoint(); break;
    default: break;
  }
  phase("idle");
}
bool stageOutbox(uint8_t kind,uint8_t m,size_t n) {
  if(n>sizeof(saved.pendingFrame) || saved.pendingLen) return false;
  saved.pendingKind=kind; saved.pendingModel=m; saved.pendingLen=n;
  memcpy(saved.pendingFrame,frame,n); return checkpoint();
}
bool startReplay() {
  if(saved.finalMask != TINYFL_RUN_LEVEL_MASK) { stop("MISSING FINAL MODEL"); return false; }
  int m=saved.replayMachine;
  if(m>=machines) { saved.complete=1; checkpoint(); phase("idle"); screen("DONE",1,1); return false; }
  releaseReplay(); replayModels=new TfReplayModel[7];
  for(int i=0;i<7;i++) if((TINYFL_RUN_LEVEL_MASK&(1<<i)) && (active[i]&(1<<m))) {
    char path[24]; modelPath(path,i); uint8_t payload[kGlobalBytes]; TfGlobalView g;
    if(!blobRead(path,payload,sizeof(payload)) || !tf_decode_global(payload,&g) ||
       g.runId!=saved.run || g.round!=12 || g.levelIndex!=i) { stop("REPLAY MODEL INVALID"); return false; }
    auto& rm=replayModels[replayModelCount++]; rm.levelIndex=i; tf_dequantise(g.codes,g.scale,rm.w);
  }
  if(!saved.replayStarted) {
    for(int i=0;i<7;i++) tf_alarm_init(&saved.alarm[i]); saved.replayNext=0;
    saved.replayStarted=1;
  }
  // A board/card's model membership is validated above from levels.csv.
  if(!replayModelCount) {
    size_t n=tf_encode_replay_done(frame,positions[m],0,0,0);
    stageOutbox(2,0,n); return false;
  }
  phase("sd_read"); screen("SD REPLAY VALIDATION",m,machines);
  replay=new TfReplay;
  if(!replay->begin(&cardFs,info[m],replayModels,replayModelCount,saved.alarm) ||
     (saved.replayNext && !replay->seekTo(saved.replayNext))) { stop(tf_last_error()); return false; }
  if(!card->exists("/results.csv")) {
    csv=card->open("/results.csv",FILE_WRITE);
    if(!csv) { stop("CSV OPEN FAIL"); return false; }
    csv.println("position,index,level,score,flag"); csv.flush(); saved.csvOffset=csv.position();
    if(!checkpoint()) return false;
    csv.close();
  }
  // Roll back rows written after the last durable checkpoint (32 rows max).
  if(truncate("/sdcard/results.csv",saved.csvOffset)!=0) { stop("CSV ROLLBACK FAIL"); return false; }
  csv=card->open("/results.csv",FILE_APPEND);
  if(!csv) { stop("CSV OPEN FAIL"); return false; }
  phase("replay"); return true;
}
void replayStep() {
  if(!replay) { startReplay(); return; }
  int m=saved.replayMachine;
  // On reboot or after a checkpoint, recover any latched but untransmitted
  // alarm from durable ring state. Server deduplicates (position, level).
  for(int i=0;i<replayModelCount;i++) if(saved.alarm[i].latched && !saved.alarm[i].sent) {
    // An outbox is always staged in the same checkpoint as the first latch.
    // This branch is only reached for the OTHER simultaneous latches. Read
    // their score from the durable alarm file saved before the checkpoint.
    char p[24]; snprintf(p,sizeof(p),"/alarm%u",i); uint8_t body[21];
    if(!blobRead(p,body,sizeof(body))) { stop("ALARM RECOVERY FAIL"); return; }
    memcpy(frame,body,sizeof(body)); stageOutbox(1,i,sizeof(body)); return;
  }
  if(replay->nextIndex()>=replay->total()) {
    int checked,bad; replay->finish(&checked,&bad);
    if(bad) { stop("REPLAY CRC FAIL"); return; }
    uint8_t mask=0,count=0;
    for(int i=0;i<replayModelCount;i++) { mask|=1<<replayModels[i].levelIndex; count+=saved.alarm[i].latched; }
    csv.flush(); saved.csvOffset=csv.position(); checkpoint();
    size_t n=tf_encode_replay_done(frame,positions[m],mask,replay->total(),count);
    stageOutbox(2,0,n); return;
  }
  phase("replay"); TfReplayResult result;
  if(!replay->step(&result) || result.badScore) { stop(tf_last_error()); return; }
  bool latched=false;
  for(int i=0;i<replayModelCount;i++) {
    if(!csv.printf("%u,%lu,%u,%.9g,%u\n",positions[m],(unsigned long)result.index,
                   replayModels[i].levelIndex,result.score[i],result.ev[i].flag?1:0)) { stop("CSV WRITE FAIL"); return; }
    if(result.ev[i].newlyLatched) {
      alarms++; saved.totalAlarms=alarms; latched=true;
      size_t n=tf_encode_alarm(frame,positions[m],replayModels[i].levelIndex,result.index,
                               result.score[i],(float)result.ev[i].threshold);
      char p[24]; snprintf(p,sizeof(p),"/alarm%u",i);
      if(!blobWrite(p,frame,n)) { stop("ALARM SAVE FAIL"); return; }
    }
  }
  saved.replayNext=replay->nextIndex();
  if(latched || (saved.replayNext%32)==0 || saved.replayNext==replay->total()) {
    csv.flush(); saved.csvOffset=csv.position(); checkpoint();
  }
  char text[48]; snprintf(text,sizeof(text),"REPLAY p%u acq %lu/%lu",positions[m],
                          (unsigned long)saved.replayNext,(unsigned long)replay->total());
  if(millis()-lastDisplay>150 || latched) screen(text,saved.replayNext,replay->total());
}
void work() {
  if(!joined || fatal || inFlight!=None || LMIC.opmode&OP_TXRXPEND) return;
  if((int32_t)(millis()-nextTxAt)<0) return;
  if(helloStage==0) { frameLen=tf_encode_hello(frame,board,machines,32,kHelloSdMounted|(allChecks?kHelloSelfCheckOk:0),saved.run); queue(frameLen,Hello); return; }
  if(helloStage<=machines) {
    int m=helloStage-1; auto& c=info[m].check;
    frameLen=tf_encode_self_check(frame,positions[m],c.rows,c.maxAbs,c.medAbs); queue(frameLen,Check); return;
  }
  if(!allChecks) { stop("SELF-CHECK FAILED"); return; }
  if(saved.pendingLen) { memcpy(frame,saved.pendingFrame,saved.pendingLen); queue(saved.pendingLen,Outbox); return; }
  // Drain repairs by ascending position; each cached UPDATE is immutable.
  for(int m=0;m<machines;m++) if(repairMask[m]) {
    if(!prepareUpdate(m)) return; txMachine=m;
    for(txFrag=0;txFrag<3;txFrag++) if(repairMask[m]&(1<<txFrag)) break;
    frameLen=tf_update_fragment(frame,updates[m],tf_slot(level,saved.targetRound+1),txFrag);
    char text[48]; snprintf(text,sizeof(text),"TX p%u frag %u/3",positions[m],txFrag+1); screen(text,txFrag+1,3);
    queue(frameLen,Repair); return;
  }
  if(needStatus) { frameLen=tf_encode_status(frame,statusSlot,statusMask); queue(frameLen,Status); return; }
  if(needAdvance) {
    needAdvance=false; int next=nextLevel(level);
    if(next>=0) { level=next; expectGlobal(tf_slot(level,0)); }
    else { waiting=false; needReady=false; LMIC_enableClassC(0); phase("replay"); }
  }
  if(trainPending) {
    LMIC_enableClassC(0);
    for(int m=0;m<machines;m++) if((active[level]&(1<<m)) && !(saved.sentMachines&(1<<m))) {
      if(!prepareUpdate(m)) return; txMachine=m;
      frameLen=tf_update_fragment(frame,updates[m],tf_slot(level,saved.targetRound+1),txFrag);
      char text[48]; snprintf(text,sizeof(text),"TX p%u frag %u/3",positions[m],txFrag+1); screen(text,txFrag+1,3);
      queue(frameLen,Update); return;
    }
    trainPending=false; expectGlobal(tf_slot(level,saved.targetRound+1));
  }
  if(needReady) { frameLen=tf_encode_ready(frame,waitSlot,DR_SF7,LMIC.dn2Dr); queue(frameLen,Ready); return; }
  if(waiting) {
    uint32_t timeout=assembler.mask()?120000:180000;
    if(millis()-lastActivity>=timeout) {
      needStatus=true; statusSlot=waitSlot; statusMask=assembler.missingMask(); LMIC_enableClassC(0); phase("idle");
    }
    return;
  }
  if(saved.finalMask==TINYFL_RUN_LEVEL_MASK && !saved.complete) replayStep();
  else if(saved.complete) { phase("idle"); }
}
}

void os_getArtEui(u1_t* b) { memcpy_P(b,tinyfl_node::joinEui,8); }
void os_getDevEui(u1_t* b) { memcpy(b,tinyfl_node::devEui,8); }
void os_getDevKey(u1_t* b) { memcpy_P(b,tinyfl_node::appKey,16); }
void onEvent(ev_t event) {
  using namespace tinyfl_node;
  if(event==EV_JOINED) joined=true;
  if(event==EV_JOIN_FAILED || event==EV_REJOIN_FAILED || event==EV_TXCANCELED) joinFailed=true;
  if(event==EV_TXCOMPLETE) txDone=true;
  if(event==EV_TXSTART) phase("tx");
  if((event==EV_TXCOMPLETE || event==EV_RXCOMPLETE) && LMIC.dataLen && LMIC.dataBeg &&
     LMIC.frame[LMIC.dataBeg-1]==tinyfl::kFPort) copyDownlink(LMIC.frame+LMIC.dataBeg,LMIC.dataLen);
}
void setup() {
  using namespace tinyfl_node;
  Serial.begin(115200); delay(500); phase("sd_read");
  Wire.begin(21,22); oled=display.begin(SSD1306_SWITCHCAPVCC,0x3c); screen("SD",0,1);
  tf_set_yield(coreYield,nullptr); tf_set_progress(progress,nullptr);
#if TINYFL_SD_MMC
  pinMode(13,OUTPUT); digitalWrite(13,HIGH); // DAT3/CS: do not select SPI mode
  if(!SD_MMC.begin("/sdcard",true,false,SDMMC_FREQ_DEFAULT)) { stop("SD MMC MOUNT FAILED"); return; }
  card=&SD_MMC;
#else
  sdSpi.begin(14,2,15,13);
  if(!SD.begin(13,sdSpi,20000000,"/sdcard")) { stop("SD SPI MOUNT FAILED"); return; }
  card=&SD;
#endif
  memset(positions,255,sizeof(positions));
  if(!readCardManifest()) { stop("CARD MANIFEST INVALID"); return; }
  // Never format automatically: preserve run/replay-counter data on a mount error.
  if(!LittleFS.begin(false)) { stop("PROVISION LITTLEFS"); return; }
  bool restored=loadProgress(); if(fatal) return;
  if(restored && saved.resetPending && !finishRunReset()) return;
  if(!restored) { saved.board=board; saved.nMachines=machines; memcpy(saved.positions,positions,machines); }
  // Recheck the card every boot; only one machine's buffers exist at a time.
  for(int m=0;m<machines;m++) {
    phase("features"); screen("SELF-CHECK",m,machines); TfExtractor ex;
    if(!tf_prepare_machine(&cardFs,positions[m],ex,&info[m])) { stop(tf_last_error()); return; }
    bool pass=info[m].check.compared && info[m].check.maxAbs<=1e-3f; allChecks=allChecks&&pass;
    Serial.printf("SELF_CHECK p=%u rows=%u max=%.9g median=%.9g %s core_peak=%u free=%u\n",
      positions[m],info[m].check.rows,info[m].check.maxAbs,info[m].check.medAbs,pass?"PASS":"FAIL",
      (unsigned)tf_heap_peak(),ESP.getFreeHeap());
    if(restored && saved.inputCrc[m]!=info[m].inputCrc) { stop("RAW CARD CONTENT CHANGED"); return; }
    saved.inputCrc[m]=info[m].inputCrc;
  }
  checkpoint(); phase("idle"); screen("JOIN",0,1);
  uint64_t mac=ESP.getEfuseMac(); uint8_t msb[8]={0,0}; uint32_t seed=0x811c9dc5;
  for(int i=0;i<6;i++) msb[i+2]=mac>>(8*(5-i));
  Serial.print("DevEUI: ");
  for(int i=0;i<8;i++) { devEui[i]=msb[7-i]; Serial.printf("%02X",msb[i]); seed=(seed^msb[i])*0x01000193; }
  Serial.println();
  if(!TINYFL_CREDENTIALS_CONFIGURED) { stop("CONFIGURE LAB KEYS"); return; }
  if(!SPI.begin(5,19,27,18) || !os_init_ex(&lmic_pins)) { stop("RADIO INIT FAILED"); return; }
  radio=true; LMIC_reset(); LMIC_setClockError((MAX_CLOCK_ERROR*4u)/1000u);
#if TINYFL_MULTICAST_DEVADDR
  static const uint8_t nwk[16]=TINYFL_MULTICAST_NWKSKEY, app[16]=TINYFL_MULTICAST_APPSKEY;
  Preferences p; char key[16]; snprintf(key,sizeof(key),"fcnt%08lX",(unsigned long)TINYFL_MULTICAST_DEVADDR);
  bool have=false; uint32_t counter=0;
  if(p.begin("tinyflmc",true)) { have=p.isKey(key); counter=p.getULong(key,0); p.end(); }
  TinyFLMulticast::begin(TINYFL_MULTICAST_DEVADDR,nwk,app,counter,have);
#endif
  staggerSeconds=seed%40; LMIC_startJoining();
}
void loop() {
  using namespace tinyfl_node;
  if(fatal) { delay(20); return; }
  if(radio) os_runloop_once();
  if(joinFailed) { stop("LMIC JOIN/TX FAILURE"); return; }
  static bool configured=false;
  if(joined && !configured) {
    configured=true; LMIC_setLinkCheckMode(0); LMIC_setAdrMode(0); LMIC_setDrTxpow(DR_SF7,14);
    nextTxAt=millis()+staggerSeconds*1000u;
    restoreBase(); screen("JOIN OK",1,1);
  }
#if TINYFL_MULTICAST_DEVADDR
  uint8_t multicast[tinyfl::kMaxFrame],port; size_t n;
  while(TinyFLMulticast::poll(multicast,sizeof(multicast),&n,&port)) if(port==tinyfl::kFPort) copyDownlink(multicast,n);
  // Persist BEFORE application dispatch; do not erase this namespace on a new
  // run. Namespace/key exactly matches the existing v2 firmware.
  static uint32_t lastCounter=0; static bool haveCounter=false;
  if(TinyFLMulticast::hasCounter() && (!haveCounter || lastCounter!=TinyFLMulticast::frameCounter())) {
    Preferences p; char key[16]; snprintf(key,sizeof(key),"fcnt%08lX",(unsigned long)TINYFL_MULTICAST_DEVADDR);
    if(!p.begin("tinyflmc",false) || p.putULong(key,TinyFLMulticast::frameCounter())!=4) { stop("MC COUNTER SAVE FAIL"); return; }
    p.end(); lastCounter=TinyFLMulticast::frameCounter(); haveCounter=true;
  }
#endif
  if(txDone) { txDone=false; finishTx(); }
  while(downTail!=downHead) { processDownlink(downlink[downTail],downLen[downTail]); downTail=(downTail+1)%4; }
  work(); delay(1);
}
