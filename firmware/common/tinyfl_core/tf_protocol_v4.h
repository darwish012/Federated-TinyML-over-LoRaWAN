// PROTOCOL_V4 rev 4.3; separate from the v3 regression codec.
#pragma once
#include "tf_common.h"
namespace tinyfl { namespace v4 {
constexpr size_t HeaderBytes=8, BodyBytes=213, MaxFrame=221, UpdateBytes=598, GlobalBytes=624;
constexpr uint8_t Final=1, AllPositions=2;
constexpr uint16_t NoSlot=0xffff, MaxSlot=0xfffe;
enum Type : uint8_t { UPDATE=1, READY=2, GLOBAL=3, STATUS=4, REPAIR=5, HELLO=6,
                     ALARM=7, CHECK=8, DONE=9, CONTROL=10 };
enum Command : uint8_t { REPLAY=1, END=2, ADOPT=3 };
struct Header { uint16_t slot; uint8_t type,index,count,len; };
inline size_t header(uint8_t* f,uint8_t type,uint16_t slot,uint8_t i,uint8_t n,uint8_t len) {
  f[0]=0xf1; f[1]=4; put_u16(f+2,slot); f[4]=i; f[5]=n; f[6]=len; f[7]=type; return 8;
}
inline bool parse(const uint8_t* f,size_t n,Header& h,const uint8_t*& b) {
  if(n<8 || n>221 || f[0]!=0xf1 || f[1]!=4 || f[6]>213 || n!=8u+f[6] ||
     f[7]<1 || f[7]>10 || f[4]>=f[5] || f[5]!=((f[7]==1 || f[7]==3)?3:1)) return false;
  h={get_u16(f+2),f[7],f[4],f[5],f[6]}; b=f+8; return true;
}
struct Model {
  uint32_t run; uint16_t position,base,rows;
  uint8_t phase,round,rounds,flags;
  float scales[4]; int8_t codes[kNumParams]; float loss;
  uint16_t maskBase; uint8_t mask[32];
};
inline bool participates(const Model& m,uint16_t position) {
  if(position==NoSlot) return false;
  if(m.flags&AllPositions) return true;
  if(position<m.maskBase) return false;
  uint32_t i=uint32_t(position)-m.maskBase;
  return i<256 && ((m.mask[i/8]>>(i%8))&1);
}
inline bool validQuant(const Model& m) {
  for(int i=0;i<4;i++) if(!isfinite(m.scales[i]) || m.scales[i]<=0) return false;
  for(int i=0;i<kNumParams;i++) if(m.codes[i]==-128) return false; return true;
}
inline size_t encodeModel(uint8_t* out,uint8_t type,const Model& m) {
  if(m.phase==255 || !validQuant(m)) return 0; size_t sc,co,crc;
  if(type==UPDATE) {
    if(m.position==0xffff || !m.round) return 0;
    put_u16(out,m.position); put_u32(out+2,m.run); out[6]=m.phase; out[7]=m.round;
    put_u16(out+8,m.base); put_u16(out+10,m.rows); sc=12; co=28; crc=594; put_f32(out+590,m.loss);
  } else if(type==GLOBAL) {
    if(!m.rounds || m.round>m.rounds || (m.flags&~3u)) return 0;
    put_u32(out,m.run); out[4]=m.phase; out[5]=m.round; out[6]=m.rounds;
    out[7]=(m.round==m.rounds?Final:0)|(m.flags&AllPositions); sc=8; co=24; crc=620;
    put_u16(out+586,m.maskBase); memcpy(out+588,m.mask,32);
  } else return 0;
  for(int i=0;i<4;i++) put_f32(out+sc+4*i,m.scales[i]);
  memcpy(out+co,m.codes,kNumParams); put_u32(out+crc,tf_crc32(0,out,crc)); return crc+4;
}
inline bool decodeModel(const uint8_t* in,size_t n,uint8_t type,Model& m) {
  if(n!=(type==UPDATE?UpdateBytes:GlobalBytes) || (type!=UPDATE && type!=GLOBAL) ||
     get_u32(in+n-4)!=tf_crc32(0,in,n-4)) return false;
  m=Model{}; size_t sc,co;
  if(type==UPDATE) {
    m.position=get_u16(in); m.run=get_u32(in+2); m.phase=in[6]; m.round=in[7];
    m.base=get_u16(in+8); m.rows=get_u16(in+10); m.loss=get_f32(in+590); sc=12; co=28;
    if(m.position==0xffff || !m.round) return false;
  } else {
    m.run=get_u32(in); m.phase=in[4]; m.round=in[5]; m.rounds=in[6]; m.flags=in[7]; sc=8; co=24;
    if(!m.rounds || m.round>m.rounds || (m.flags&~3u) || (m.flags&Final)!=(m.round==m.rounds?Final:0)) return false;
    m.maskBase=get_u16(in+586); memcpy(m.mask,in+588,32);
  }
  for(int i=0;i<4;i++) m.scales[i]=get_f32(in+sc+4*i);
  memcpy(m.codes,in+co,kNumParams); return m.phase!=255 && validQuant(m);
}
inline size_t fragment(uint8_t* f,const uint8_t* payload,uint8_t type,uint16_t slot,uint8_t index) {
  if(index>=3 || (type!=UPDATE && type!=GLOBAL)) return 0;
  size_t chunk=type==UPDATE?211:213, size=type==UPDATE?UpdateBytes:GlobalBytes;
  size_t offset=index*chunk,n=size-offset; if(n>chunk) n=chunk;
  uint8_t prefix=type==UPDATE?2:0; header(f,type,slot,index,3,n+prefix);
  if(prefix) memcpy(f+8,payload,2); memcpy(f+8+prefix,payload+offset,n); return 8+prefix+n;
}
class Assembly {
 public:
  uint16_t slot=0; uint8_t mask=0; uint8_t bytes[GlobalBytes]={};
  void reset() { slot=0; mask=0; }
  int add(const uint8_t* f,size_t n) {
    Header h; const uint8_t* b; if(!parse(f,n,h,b) || h.type!=GLOBAL) return 0;
    size_t off=h.index*213u,want=GlobalBytes-off; if(want>213) want=213;
    if(h.len!=want) return 0; if(h.slot!=slot) { mask=0; slot=h.slot; }
    uint8_t bit=1<<h.index; if(mask&bit) return memcmp(bytes+off,b,want)?0:3;
    memcpy(bytes+off,b,want); mask|=bit; return mask==7?2:1;
  }
};
// All post-update, repair and final-model READY paths use this single rule.
inline uint16_t awaitedSlot(uint16_t heldSlot) { return heldSlot?uint16_t(heldSlot+1):NoSlot; }
inline bool controlValid(const Header& h,const uint8_t* b) {
  switch(h.type) {
    case READY: return h.len==6 && get_u16(b+4)==h.slot;
    case STATUS: return h.len==7 && get_u16(b+4)==h.slot && !(b[6]&~7u);
    case REPAIR: return h.len==7 && get_u16(b+4)!=0xffff && b[6] && !(b[6]&~7u);
    case HELLO:
      if(h.len<9 || b[8]>8 || h.len!=9+2*b[8]) return false;
      for(int i=0;i<b[8];i++) {
        if(get_u16(b+9+2*i)==0xffff) return false;
        for(int j=0;j<i;j++) if(get_u16(b+9+2*i)==get_u16(b+9+2*j)) return false;
      } return true;
    case ALARM: return h.len==19 && get_u16(b+4)!=0xffff && b[6]!=255;
    case CHECK: return h.len==12 && get_u16(b)!=0xffff;
    case DONE: return h.len==16 && get_u16(b+4)!=0xffff && b[6]!=255 && b[11]<=1 &&
                      bool(b[11])==(get_u32(b+12)!=0xffffffffu);
    case CONTROL:
      if(h.len<6 || h.len!=6+b[5] || b[4]<1 || b[4]>3 ||
         (b[4]==REPLAY && !b[5]) || (b[4]==ADOPT && b[5])) return false;
      for(int i=0;i<b[5];i++) {
        if(b[6+i]==255) return false;
        for(int j=0;j<i;j++) if(b[6+i]==b[6+j]) return false;
      } return true;
    default: return false;
  }
}
// Scalars/list entries in contract order; production and host share encoder.
inline size_t control(uint8_t* f,uint8_t type,uint16_t slot,const uint32_t* v,
                      const float* x,const uint16_t* list,uint8_t count) {
  uint8_t* b=f+8; size_t n=0;
  switch(type) {
    case READY: put_u32(b,v[0]); put_u16(b+4,v[1]); n=6; break;
    case STATUS: case REPAIR: put_u32(b,v[0]); put_u16(b+4,v[1]); b[6]=v[2]; n=7; break;
    case HELLO:
      if(count>8) return 0; put_u16(b,v[0]); b[2]=v[1]; b[3]=v[2]; put_u32(b+4,v[3]); b[8]=count;
      for(int i=0;i<count;i++) put_u16(b+9+2*i,list[i]); n=9+2*count; break;
    case ALARM:
      put_u32(b,v[0]); put_u16(b+4,v[1]); b[6]=v[2]; put_u32(b+7,v[3]);
      put_f32(b+11,x[0]); put_f32(b+15,x[1]); n=19; break;
    case CHECK:
      put_u16(b,v[0]); put_u16(b+2,v[1]); put_f32(b+4,x[0]); put_f32(b+8,x[1]); n=12; break;
    case DONE:
      put_u32(b,v[0]); put_u16(b+4,v[1]); b[6]=v[2]; put_u32(b+7,v[3]); b[11]=v[4];
      put_u32(b+12,v[5]); n=16; break;
    case CONTROL:
      if(count>207) return 0; put_u32(b,v[0]); b[4]=v[1]; b[5]=count;
      for(int i=0;i<count;i++) { if(list[i]>254) return 0; b[6+i]=list[i]; } n=6+count; break;
    default: return 0;
  }
  header(f,type,slot,0,1,n); Header h; const uint8_t* body;
  return parse(f,8+n,h,body) && controlValid(h,body)?8+n:0;
}
inline size_t ready(uint8_t* f,uint32_t run,uint16_t held) {
  uint16_t slot=awaitedSlot(held); uint32_t v[]={run,slot}; return control(f,READY,slot,v,nullptr,nullptr,0);
}
} }
