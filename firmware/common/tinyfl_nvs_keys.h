#pragma once
#include <Preferences.h>
#include "tinyfl_core/tf_provision_v4.h"
namespace tinyfl { namespace v4 {
static const char* const KeyNames[]={"joineui","appkey","mcaddr","mcnwk","mcapp"};
static const uint8_t KeySizes[]={8,16,4,16,16};
inline bool loadKeys(Keys& keys) {
  Preferences p; if(!p.begin("tinyflkeys",true)) return false; bool ok=true; int offset=0;
  for(int i=0;i<5;i++) {
    ok=ok && p.getBytesLength(KeyNames[i])==KeySizes[i] &&
       p.getBytes(KeyNames[i],keys.raw+offset,KeySizes[i])==KeySizes[i]; offset+=KeySizes[i];
  } p.end(); return ok;
}
inline bool storeKeys(const Keys& keys,Keys& readback) {
  Preferences p; if(!p.begin("tinyflkeys",false)) return false; bool ok=true; int offset=0;
  for(int i=0;i<5;i++) {
    if(p.putBytes(KeyNames[i],keys.raw+offset,KeySizes[i])!=KeySizes[i]) ok=false; offset+=KeySizes[i];
  } p.end(); return ok && loadKeys(readback) && !memcmp(keys.raw,readback.raw,60);
}
} }
