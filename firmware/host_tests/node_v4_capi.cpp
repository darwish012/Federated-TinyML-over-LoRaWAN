#include "../tinyfl_fl_node/node_v4.h"
#include <filesystem>
#include <fstream>
#define NODEAPI extern "C" __declspec(dllexport)
using namespace tinyfl_node;
NODEAPI void tfn_reset(int count) {
  releaseReplay(); saved=Progress(); LittleFS.files.clear(); SD_MMC.files.clear(); card=&SD_MMC;
  // Duty-cycle timing is off by default: the legacy tests drive frames
  // back to back without advancing time. tfn_duty(1) turns the EU868 model on.
  fakeMillis=100000; fakeTx.clear(); fakeTxLog.clear(); fakeDutyCycle=false; LMIC_reset(); fakeClassC=false;
  fakeClassCBadCalls=0; fakeRxResumeAt=0;
  fatal=false; joined=true; radio=false; allChecks=true; txDone=false; joinFailed=false;
  board=1000; machines=count; validUpdates=0; inFlight=None; txFrag=0; helloStage=0; nextTxAt=0;
  needReady=false; needStatus=false; waiting=false; assembler.reset(); restartCandidate.reset();
  heardRun=0; activeMachines=0; pendingAdoptHellos=0; completedBootAnnounce=false; lastActivity=fakeMillis;
  memset(repairMask,0,sizeof(repairMask)); memset(info,0,sizeof(info));
  for(int i=0;i<count;i++) positions[i]=300+i;
  saved.board=board; saved.machines=count; memcpy(saved.positions,positions,2*count);
  restoreBase(); checkpoint();
}
NODEAPI size_t tfn_next(uint8_t* out) {
  fakeTx.clear(); work(); if(fakeTx.empty()) return 0; memcpy(out,fakeTx.data(),fakeTx.size()); return fakeTx.size();
}
NODEAPI void tfn_finish() { LMIC.opmode=0; fakeRxResumeAt=fakeMillis+2000; finishTx(); } // RX1/RX2 stand-in: 2 s
NODEAPI void tfn_deliver(const uint8_t* f,size_t n) { processDownlink(f,n); }
NODEAPI int tfn_update(int m,const uint8_t* bytes) {
  p4::Model u; if(m<0 || m>=machines || !p4::decodeModel(bytes,598,p4::UPDATE,u) || u.position!=positions[m]) return 0;
  memcpy(updates[m],bytes,598); validUpdates|=1<<m;
  char path[24]; snprintf(path,sizeof(path),"/v4update%u",m); return blobWrite(path,bytes,598);
}
NODEAPI void tfn_state(uint32_t* out) {
  uint32_t values[]={saved.run,saved.heldSlot,saved.seedSlot,saved.phase,saved.round,saved.rounds,saved.sent,
    uint32_t(needReady),uint32_t(waiting),saved.complete,uint32_t(fatal),uint32_t(fakeClassC),saved.replayStarted};
  memcpy(out,values,sizeof(values));
}
NODEAPI int tfn_reboot() {
  releaseReplay(); saved=Progress(); validUpdates=0; inFlight=None; txFrag=0; helloStage=0;
  waiting=false; needStatus=false; needReady=false; LMIC_reset(); fakeClassC=false;
  heardRun=0; activeMachines=0; pendingAdoptHellos=0; completedBootAnnounce=false;
  memset(repairMask,0,sizeof(repairMask)); assembler.reset(); restartCandidate.reset();
  if(!loadProgress()) return 0; restoreBase(); return !fatal;
}
NODEAPI int tfn_final(int phase) { p4::Model g; return modelRead(phase,g); }
NODEAPI int tfn_requested(int phase) { return hasBit(saved.requested,phase); }
NODEAPI size_t tfn_alarm_state_bytes() { return sizeof(TfAlarmState); }
NODEAPI void tfn_advance(uint32_t ms) { fakeMillis+=ms; }
NODEAPI uint32_t tfn_active() { return activeMachines; }
NODEAPI uint32_t tfn_participants(int p) { return p>=0 && p<255?saved.participants[p]:0; }
NODEAPI int tfn_model_active(int p,int m) {
  p4::Model g;
  return p>=0 && p<255 && m>=0 && m<machines && modelRead(p,g) && p4::participates(g,positions[m]);
}
NODEAPI size_t tfn_csv(uint8_t* out,size_t capacity) {
  File f=card->open("/results.csv",FILE_READ);
  return f?f.read(out,min(capacity,size_t(f.size()))):0;
}
NODEAPI uint32_t tfn_heard() { return heardRun; }
// Radio-scheduling (duty-cycle) test hooks. The model is the mock's LMIC
// transcription; see mock/node_platform.h.
NODEAPI void tfn_duty(int on) { fakeDutyCycle=on!=0; if(on) LMIC_reset(); }
NODEAPI int tfn_uplink_ready() { return uplinkReady(); }
NODEAPI int tfn_receiving() { return fakeReceiving(); }          // Class C on and no TX requested/RX windows open
NODEAPI int tfn_classc_bad_calls() { return fakeClassCBadCalls; } // LMIC_enableClassC(0), or (1) while already on
NODEAPI size_t tfn_txlog_count() { return fakeTxLog.size(); }
NODEAPI void tfn_txlog(size_t i,uint64_t* out) { // queuedUs,startUs,airtimeUs,band,length,delayed
  const FakeTxRecord& r=fakeTxLog.at(i);
  out[0]=r.queuedUs; out[1]=r.startUs; out[2]=r.airtimeUs; out[3]=r.band; out[4]=r.length; out[5]=r.delayed;
}
NODEAPI void tfn_corrupt_latest() {
  auto& bytes=LittleFS.files[saved.generation&1?"/v4progress1":"/v4progress0"]->bytes;
  bytes.back()^=1;
}
NODEAPI int tfn_load_card(const char* root) {
  for(const auto& entry:std::filesystem::directory_iterator(root)) if(entry.is_regular_file()) {
    std::ifstream in(entry.path(),std::ios::binary);
    auto data=std::make_shared<FileData>();
    data->bytes.assign(std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>());
    card->files["/"+entry.path().filename().string()]=data;
  }
  for(int m=0;m<machines;m++) {
    TfExtractor ex;
    if(!tf_prepare_machine(&cardFs,positions[m],ex,&info[m])) return 0;
    saved.inputCrc[m]=info[m].inputCrc;
  }
  return checkpoint();
}
