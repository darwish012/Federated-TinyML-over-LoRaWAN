#pragma once
// Host-only hardware interfaces. Production node_v4.h is compiled unchanged;
// radios record queued frames, and SD/LittleFS live in memory. No devices open.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <memory>
#include <map>
#include <string>
#include <vector>
#include <new>
using std::min;
inline uint32_t fakeMillis=100000;
inline uint32_t millis() { return fakeMillis; }
inline void delay(uint32_t n) { fakeMillis+=n; }
inline void enableLoopWDT() {}
inline void feedLoopWDT() {}
inline void pinMode(int,int) {}
inline void digitalWrite(int,int) {}
#define OUTPUT 1
#define HIGH 1
#define LOW 0
#define HSPI 1
#define FILE_READ "r"
#define FILE_WRITE "w"
#define FILE_APPEND "a"
#define SDMMC_FREQ_DEFAULT 20000
#define SSD1306_WHITE 1
#define SSD1306_SWITCHCAPVCC 1
#define CFG_eu868 1
#define CFG_sx1276_radio 1
#define LMIC_ENABLE_class_c 1
#define MAX_FRAME_LEN 255
class String {
 public:
  std::string s;
  String()=default; String(const char* x):s(x) {} String(std::string x):s(x) {}
  String(uint32_t x):s(std::to_string(x)) {}
  const char* c_str() const { return s.c_str(); } size_t length() const { return s.size(); }
  void trim() { auto b=s.find_first_not_of(" \r\n\t"),e=s.find_last_not_of(" \r\n\t"); s=b==std::string::npos?"":s.substr(b,e-b+1); }
  friend String operator+(const String& a,const String& b) { return String(a.s+b.s); }
};
struct FileData { std::vector<uint8_t> bytes; };
class File {
 public:
  File()=default; File(std::shared_ptr<FileData> data,bool write,size_t at=0):data_(data),write_(write),at_(at) {}
  explicit operator bool() const { return bool(data_); }
  size_t read(uint8_t* out,size_t n) {
    if(!data_ || at_>data_->bytes.size()) return 0;
    n=min(n,data_->bytes.size()-at_); memcpy(out,data_->bytes.data()+at_,n); at_+=n; return n;
  }
  size_t write(const uint8_t* in,size_t n) {
    if(!data_ || !write_) return 0; if(at_+n>data_->bytes.size()) data_->bytes.resize(at_+n);
    memcpy(data_->bytes.data()+at_,in,n); at_+=n; return n;
  }
  size_t printf(const char* format,...) {
    char b[512]; va_list a; va_start(a,format); int n=vsnprintf(b,sizeof(b),format,a); va_end(a);
    return n>0 && n<int(sizeof(b))?write((const uint8_t*)b,n):0;
  }
  size_t println(const char* s) { return write((const uint8_t*)s,strlen(s))+write((const uint8_t*)"\n",1); }
  bool seek(uint32_t at) { if(!data_ || at>data_->bytes.size()) return false; at_=at; return true; }
  uint64_t size() const { return data_?data_->bytes.size():0; } size_t position() const { return at_; }
  bool available() const { return data_ && at_<data_->bytes.size(); }
  String readStringUntil(char delimiter) {
    std::string out; while(available()) { char c=data_->bytes[at_++]; if(c==delimiter) break; out+=c; } return String(out);
  }
  void flush() {} void close() { data_.reset(); }
 private: std::shared_ptr<FileData> data_; bool write_=false; size_t at_=0;
};
namespace fs {
class FS {
 public:
  std::map<std::string,std::shared_ptr<FileData>> files;
  File open(const String& path,const char* mode=FILE_READ) {
    auto key=path.s;
    if(strcmp(mode,FILE_READ)==0) { auto it=files.find(key); return it==files.end()?File():File(it->second,false); }
    auto& data=files[key]; if(!data) data=std::make_shared<FileData>();
    if(strcmp(mode,FILE_WRITE)==0) data->bytes.clear();
    return File(data,true,strcmp(mode,FILE_APPEND)==0?data->bytes.size():0);
  }
  bool exists(const String& path) const { return files.count(path.s)>0; }
  bool remove(const String& path) { return files.erase(path.s)>0; }
  bool rename(const String& from,const String& to) {
    auto it=files.find(from.s); if(it==files.end()) return false; files[to.s]=it->second; files.erase(it); return true;
  }
  bool begin(const char*,bool,bool,int) { return true; }
  bool begin(bool) { return true; }
  template<class S> bool begin(int,S&,int,const char*) { return true; }
  bool format() { files.clear(); return true; } void end() {}
};
}
inline fs::FS SD,SD_MMC,LittleFS;
inline int truncate(const char* path,long length) {
  std::string p=path; if(p.rfind("/sdcard",0)==0) p=p.substr(7);
  auto it=SD_MMC.files.find(p); if(it==SD_MMC.files.end() || length<0) return -1;
  it->second->bytes.resize(length); return 0;
}
#define MSBFIRST 1
#define SPI_MODE0 0
struct SPISettings { SPISettings(uint32_t,uint8_t,uint8_t) {} };
class SPIClass { public: SPIClass(int=0) {} bool begin(int,int,int,int) { return true; }
  void beginTransaction(SPISettings) {} uint8_t transfer(uint8_t) { return 0xFF; } void endTransaction() {} };
inline SPIClass SPI;
struct WireClass { void begin(int,int) {} }; inline WireClass Wire;
struct SerialClass {
  void begin(int) {} void print(const char*) {} void println() {} void println(const char*) {}
  void printf(const char*,...) {} int available() { return 0; } int read() { return 0; }
}; inline SerialClass Serial;
struct EspClass { uint64_t getEfuseMac() { return 0x0123456789ab; } }; inline EspClass ESP;
class Adafruit_SSD1306 {
 public:
  template<class T> Adafruit_SSD1306(int,int,T*,int) {}
  bool begin(int,int) { return true; } void clearDisplay() {} void setTextColor(int) {} void setTextSize(int) {}
  void setCursor(int,int) {} void printf(const char*,...) {} void println(const char*) {}
  void drawRect(int,int,int,int,int) {} void fillRect(int,int,int,int,int) {} void display() {}
};
using u1_t=uint8_t;
struct lmic_pinmap { int nss,rxtx,rst; int dio[3]; int rxactive,rssi; unsigned hz; void* ptr; };
#define LMIC_UNUSED_PIN -1
#define MAX_CLOCK_ERROR 65536
#define DR_SF7 5
#define OP_TXRXPEND 1
#define OP_RNDTX 2
enum ev_t { EV_JOINED,EV_JOIN_FAILED,EV_REJOIN_FAILED,EV_TXCANCELED,EV_TXCOMPLETE,EV_TXSTART,EV_RXCOMPLETE };
enum lmic_tx_error_t { LMIC_ERROR_SUCCESS,LMIC_ERROR_TX_BUSY,LMIC_ERROR_OTHER };
// EU868 duty-cycle model, a transcription of MCCI LMIC (lmic.c engineUpdate,
// lmic_eu868.c LMICeu868_nextTx, lmic_eu_like.c LMICeulike_updateTx and
// lmic.c calcAirTime) for the one configuration the node uses (SF7/125 kHz).
// Same field names and types as the real library (ostime_t ticks, 62500/s),
// so firmware reading LMIC.bands[].avail / globalDutyAvail runs unchanged.
using ostime_t=int32_t;
#define OSTICKS_PER_SEC 62500
#define us2osticks(us) ((ostime_t)(((int64_t)(us)*OSTICKS_PER_SEC)/1000000))
#define sec2osticks(sec) ((ostime_t)((int64_t)(sec)*OSTICKS_PER_SEC))
#define TX_RAMPUP (us2osticks(10000))
enum { MAX_CHANNELS=16, MAX_BANDS=4 };
enum { BAND_MILLI=0, BAND_CENTI=1, BAND_DECI=2, BAND_AUX=3 };
inline ostime_t os_getTime();
struct band_t { uint16_t txcap; int8_t txpow; uint8_t lastchnl; ostime_t avail; };
// Same member paths as lmic_class_c_t (flags.f.fEnabled, requests.f.fStateChangeRq).
struct FakeClassC {
  struct { struct { bool fEnabled=false,fRx2Active=false; } f; } flags;
  struct { struct { bool fStateChangeRq=false,fTargetState=false,fPending=false; } f; } requests;
};
struct FakeLMIC {
  int opmode=0,dn2Dr=5,dataLen=0,dataBeg=0; uint8_t frame[255]={};
  uint16_t channelMap=7; uint32_t channelFreq[MAX_CHANNELS]={}; uint16_t channelDrMap[MAX_CHANNELS]={};
  band_t bands[MAX_BANDS]={}; ostime_t globalDutyAvail=0; uint8_t globalDutyRate=0;
  FakeClassC classC;
  FakeLMIC() {
    // LMICeu868_initDefaultChannels(0) + LMIC_setupBand(): three default
    // channels in the 1 % sub-band, DR0..DR5 allowed, every band free now.
    const uint32_t freq[3]={868100000,868300000,868500000};
    for(int i=0;i<3;i++) { channelFreq[i]=freq[i]|BAND_CENTI; channelDrMap[i]=0x3f; }
    bands[BAND_MILLI]={1000,14,0,os_getTime()}; bands[BAND_CENTI]={100,14,0,os_getTime()};
    bands[BAND_DECI]={10,27,0,os_getTime()};
  }
};
inline FakeLMIC LMIC;
inline ostime_t os_getTime() { return (ostime_t)(uint32_t)(((uint64_t)fakeMillis*OSTICKS_PER_SEC)/1000); }
inline bool& fakeClassC=LMIC.classC.flags.f.fEnabled; // "Class C enabled", as LMIC reports it
inline std::vector<uint8_t> fakeTx;
// Misuse counter: any LMIC_enableClassC(0), or (1) while already enabled. Both are
// wrong with the real library (see LMIC_enableClassC below); host tests require 0.
inline int fakeClassCBadCalls=0;
// Class C is enabled, but LMIC receives continuously only when no uplink is
// requested or in progress and RX1/RX2 of the last one have passed (engineUpdate
// starts setupRxClassC only if !isTxRequested). fakeRxResumeAt is the harness's
// stand-in for "RX windows over" (set by the host test's tfn_finish).
inline uint32_t fakeRxResumeAt=0;
inline bool fakeReceiving() {
  return LMIC.classC.flags.f.fEnabled && !(LMIC.opmode&OP_TXRXPEND) && int32_t(fakeMillis-fakeRxResumeAt)>=0;
}
// One record per uplink handed to LMIC (host tests and the testbed read it).
struct FakeTxRecord { uint64_t queuedUs,startUs,airtimeUs; uint8_t band; uint16_t length; bool delayed; };
inline std::vector<FakeTxRecord> fakeTxLog;
inline bool fakeDutyCycle=true; // false: legacy tests that ignore radio time
inline void os_runloop_once() {}
inline bool os_init_ex(const lmic_pinmap*) { return true; }
// Transcription of MCCI LMIC 6.0.1 externalRequestCb (lmic.c ~3294), bug included:
// `if (targetState == fEnabled) fEnabled = 0; else fEnabled = 1;` - so (1) while on
// switches Class C OFF, (0) while on leaves it ON, only off->on works. Applied
// immediately (the real request completes at the next os_runloop_once).
inline void LMIC_enableClassC(int value) {
  if(!value || LMIC.classC.flags.f.fEnabled) fakeClassCBadCalls++;
  LMIC.classC.flags.f.fEnabled=(bool(value)==LMIC.classC.flags.f.fEnabled)?false:true;
}
inline void LMIC_setDrTxpow(int,int) {}
// LMIC calcAirTime for SF7, 125 kHz, CR 4/5, explicit header, CRC on; plen is
// the PHY payload (application bytes + 13 bytes of LoRaWAN framing).
inline ostime_t calcAirTimeSf7(unsigned plen) {
  int tmp=8*int(plen)-28+28+16; tmp=(tmp+27)/28; tmp*=5; tmp+=8; tmp=(tmp<<2)+49;
  return (((ostime_t)tmp<<2)*OSTICKS_PER_SEC+15625/2)/15625;
}
inline lmic_tx_error_t LMIC_setTxData2_strict(int,const uint8_t* data,size_t n,int) {
  if(LMIC.opmode&OP_TXRXPEND) return LMIC_ERROR_TX_BUSY;
  fakeTx.assign(data,data+n); LMIC.opmode|=OP_TXRXPEND;
  if(!fakeDutyCycle) return LMIC_ERROR_SUCCESS;
  // Tick arithmetic wraps at 32 bits exactly as on the target (done unsigned: defined behaviour).
  auto diff=[](ostime_t a,ostime_t b) { return (ostime_t)((uint32_t)a-(uint32_t)b); };
  auto add=[](ostime_t a,ostime_t b) { return (ostime_t)((uint32_t)a+(uint32_t)b); };
  ostime_t now=os_getTime(),at=add(now,sec2osticks(28800)); // LMICeu868_nextTx
  for(int c=0;c<MAX_CHANNELS;c++) {
    if(LMIC.channelMap<(1u<<c)) break; if(!(LMIC.channelMap&(1u<<c))) continue;
    if(!(LMIC.channelDrMap[c]&(1u<<DR_SF7))) continue;
    const band_t& b=LMIC.bands[LMIC.channelFreq[c]&3]; if(diff(at,b.avail)>0) at=b.avail;
  }
  int chosen=-1;
  for(int c=0;c<MAX_CHANNELS && chosen<0;c++) {
    if(!(LMIC.channelMap&(1u<<c)) || !(LMIC.channelDrMap[c]&(1u<<DR_SF7))) continue;
    if(diff(LMIC.bands[LMIC.channelFreq[c]&3].avail,at)<=0) chosen=c;
  }
  ostime_t txbeg=at; // engineUpdate
  if((LMIC.globalDutyRate!=0 || (LMIC.opmode&OP_RNDTX)) && diff(txbeg,LMIC.globalDutyAvail)<0) txbeg=LMIC.globalDutyAvail;
  if(diff(txbeg,add(now,TX_RAMPUP))<0) txbeg=now;
  LMIC.opmode&=~OP_RNDTX;
  ostime_t airtime=calcAirTimeSf7(unsigned(n)+13); // LMICeulike_updateTx
  uint8_t band=chosen<0?BAND_CENTI:uint8_t(LMIC.channelFreq[chosen]&3);
  LMIC.bands[band].avail=add(txbeg,(ostime_t)((uint32_t)airtime*LMIC.bands[band].txcap));
  if(LMIC.globalDutyRate!=0) LMIC.globalDutyAvail=add(txbeg,(ostime_t)((uint32_t)airtime<<LMIC.globalDutyRate));
  const uint64_t base=(uint64_t)fakeMillis*1000; const ostime_t wait=diff(txbeg,now);
  fakeTxLog.push_back({base,base+(uint64_t)((int64_t)wait*16),(uint64_t)airtime*16,band,uint16_t(n),wait>0});
  return LMIC_ERROR_SUCCESS;
}
inline void LMIC_reset() { LMIC=FakeLMIC(); }
inline void LMIC_setClockError(uint32_t) {} inline void LMIC_startJoining() {}
inline void LMIC_setLinkCheckMode(int) {} inline void LMIC_setAdrMode(int) {}
namespace TinyFLMulticast {
inline void begin(uint32_t,const uint8_t*,const uint8_t*,uint32_t,bool) {}
inline bool poll(uint8_t*,size_t,size_t*,uint8_t*) { return false; }
inline bool hasCounter() { return false; } inline uint32_t frameCounter() { return 0; }
}
