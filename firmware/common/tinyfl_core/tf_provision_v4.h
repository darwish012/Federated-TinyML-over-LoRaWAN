#pragma once
#include "tf_common.h"
namespace tinyfl { namespace v4 {
struct Keys { uint8_t raw[60];
  const uint8_t* join() const { return raw; }
  const uint8_t* app() const { return raw+8; }
  const uint8_t* addr() const { return raw+24; }
  const uint8_t* nwk() const { return raw+28; }
  const uint8_t* mcapp() const { return raw+44; }
  uint32_t crc() const { return tf_crc32(0,raw,60); }
  uint32_t devaddr() const { return uint32_t(raw[24])<<24 | uint32_t(raw[25])<<16 |
                                  uint32_t(raw[26])<<8 | raw[27]; }
};
inline void lmicEui(uint8_t* out,const uint8_t* msb) { for(int i=0;i<8;i++) out[i]=msb[7-i]; }
inline int hexDigit(char c) {
  if(c>='0' && c<='9') return c-'0'; if(c>='a' && c<='f') return c-'a'+10;
  if(c>='A' && c<='F') return c-'A'+10; return -1;
}
inline bool parseKeys(const char* line,Keys& result) {
  const char prefix[]="TFPROV KEYS "; if(strncmp(line,prefix,sizeof(prefix)-1)) return false;
  const uint8_t lengths[]={8,16,4,16,16}; const char* p=line+sizeof(prefix)-1;
  Keys temp{}; int at=0;
  for(int k=0;k<5;k++) {
    for(int i=0;i<lengths[k];i++) {
      if(!p[0] || !p[1]) return false;
      int a=hexDigit(p[0]),b=hexDigit(p[1]); if(a<0 || b<0) return false;
      temp.raw[at++]=uint8_t((a<<4)|b); p+=2;
    }
    if(k<4) { if(*p!=' ') return false; while(*p==' ') p++; }
  }
  while(*p==' ' || *p=='\r' || *p=='\n') p++;
  if(*p) return false; result=temp; return true;
}
} }
