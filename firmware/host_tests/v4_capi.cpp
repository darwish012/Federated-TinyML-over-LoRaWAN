#include "../common/tinyfl_core/tf_protocol_v4.h"
#include "../common/tinyfl_core/tf_provision_v4.h"
#include "../common/tinyfl_core/tf_replay_v4.h"
#include "../common/tinyfl_nvs_keys.h"
using namespace tinyfl;
namespace p4=tinyfl::v4;
#define V4API extern "C" __declspec(dllexport)
V4API size_t tfv4_model(uint8_t* out,uint8_t type,const uint32_t* v,const float* scales,const int8_t* codes,float loss) {
  p4::Model m{}; m.run=v[0]; m.position=v[1]; m.phase=v[2]; m.round=v[3];
  m.base=v[4]; m.rows=v[5]>65535?65535:v[5]; m.rounds=v[6]; m.loss=loss;
  m.flags=p4::AllPositions;
  memcpy(m.scales,scales,16); memcpy(m.codes,codes,562); return p4::encodeModel(out,type,m);
}
V4API size_t tfv4_model_roundtrip(uint8_t* out,const uint8_t* in,size_t n,uint8_t type) {
  p4::Model m; if(!p4::decodeModel(in,n,type,m)) return 0; return p4::encodeModel(out,type,m);
}
V4API size_t tfv4_fragment(uint8_t* out,const uint8_t* in,uint8_t type,uint16_t slot,uint8_t i) {
  return p4::fragment(out,in,type,slot,i);
}
V4API size_t tfv4_control(uint8_t* out,uint8_t type,uint16_t slot,const uint32_t* v,const float* x,const uint16_t* list,uint8_t n) {
  return p4::control(out,type,slot,v,x,list,n);
}
V4API size_t tfv4_control_roundtrip(uint8_t* out,const uint8_t* in,size_t n) {
  p4::Header h; const uint8_t* b; if(!p4::parse(in,n,h,b) || !p4::controlValid(h,b)) return 0;
  uint32_t v[6]={}; float x[2]={}; uint16_t list[207]={}; uint8_t count=0;
  switch(h.type) {
    case p4::READY: v[0]=get_u32(b); v[1]=get_u16(b+4); break;
    case p4::STATUS: case p4::REPAIR: v[0]=get_u32(b); v[1]=get_u16(b+4); v[2]=b[6]; break;
    case p4::HELLO:
      v[0]=get_u16(b); v[1]=b[2]; v[2]=b[3]; v[3]=get_u32(b+4); count=b[8];
      for(int i=0;i<count;i++) list[i]=get_u16(b+9+2*i); break;
    case p4::ALARM:
      v[0]=get_u32(b); v[1]=get_u16(b+4); v[2]=b[6]; v[3]=get_u32(b+7);
      x[0]=get_f32(b+11); x[1]=get_f32(b+15); break;
    case p4::CHECK: v[0]=get_u16(b); v[1]=get_u16(b+2); x[0]=get_f32(b+4); x[1]=get_f32(b+8); break;
    case p4::DONE: v[0]=get_u32(b); v[1]=get_u16(b+4); v[2]=b[6]; v[3]=get_u32(b+7); v[4]=b[11]; v[5]=get_u32(b+12); break;
    case p4::CONTROL: v[0]=get_u32(b); v[1]=b[4]; count=b[5]; for(int i=0;i<count;i++) list[i]=b[6+i]; break;
    default: return 0;
  }
  return p4::control(out,h.type,h.slot,v,x,list,count);
}
V4API size_t tfv4_ready(uint8_t* out,uint32_t run,uint16_t held) { return p4::ready(out,run,held); }
V4API uint32_t tfv4_provision(const char* line,uint8_t* raw,uint8_t* joinLmic,uint32_t* addr,int* ok) {
  p4::Keys k{},readback{}; *ok=p4::parseKeys(line,k); if(!*ok) return 0;
  *ok=p4::storeKeys(k,readback); if(!*ok) return 0;
  memcpy(raw,readback.raw,60); p4::lmicEui(joinLmic,readback.join()); *addr=readback.devaddr(); return readback.crc();
}
V4API void tfv4_nvs_reset(int fault) {
  Preferences::values.clear(); Preferences::failWrite=fault==1; Preferences::corruptRead=fault==2;
  Preferences::values["tinyflmc/fcnt12345678"]={1,2,3,4};
}
V4API int tfv4_nvs_counter_untouched() { return Preferences::values["tinyflmc/fcnt12345678"]==std::vector<uint8_t>({1,2,3,4}); }
V4API int tfv4_nvs_get(int index,uint8_t* out) {
  if(index<0 || index>4) return 0;
  auto& bytes=Preferences::values[std::string("tinyflkeys/")+p4::KeyNames[index]];
  memcpy(out,bytes.data(),bytes.size()); return bytes.size();
}
V4API void tfv4_lmic_eui(uint8_t* out,const uint8_t* in) { p4::lmicEui(out,in); }
V4API int tfv4_assembly(const uint8_t* frames,const int* lens,int n,uint8_t* out,int* results) {
  p4::Assembly a; int offset=0;
  for(int i=0;i<n;i++) { results[i]=a.add(frames+offset,lens[i]); offset+=lens[i]; }
  if(a.mask==7) memcpy(out,a.bytes,p4::GlobalBytes); return a.mask;
}
