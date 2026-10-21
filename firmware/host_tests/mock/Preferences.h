#pragma once
// Host-only fake NVS. The production tinyfl_nvs_keys.h adapter is compiled
// unchanged against this interface, not reimplemented in Python.
#include <map>
#include <string>
#include <vector>
#include <cstring>
#include <cstdint>
class Preferences {
 public:
  inline static std::map<std::string,std::vector<uint8_t>> values;
  inline static bool failWrite=false,corruptRead=false;
  bool begin(const char* name,bool readOnly=false) { namespace_=name; readOnly_=readOnly; return true; }
  void end() {}
  bool isKey(const char* name) { return values.count(key(name)); }
  uint32_t getULong(const char* name,uint32_t fallback=0) {
    auto& b=values[key(name)]; uint32_t value; if(b.size()!=4) return fallback; memcpy(&value,b.data(),4); return value;
  }
  size_t putULong(const char* name,uint32_t value) { return putBytes(name,&value,4); }
  size_t getBytesLength(const char* name) { return values[key(name)].size(); }
  size_t getBytes(const char* name,void* out,size_t n) {
    auto& bytes=values[key(name)]; if(bytes.size()!=n) return 0; memcpy(out,bytes.data(),n);
    if(corruptRead && n) ((uint8_t*)out)[0]^=1; return n;
  }
  size_t putBytes(const char* name,const void* in,size_t n) {
    if(failWrite || readOnly_) return 0;
    values[key(name)]=std::vector<uint8_t>((const uint8_t*)in,(const uint8_t*)in+n); return n;
  }
 private:
  std::string namespace_; bool readOnly_=false;
  std::string key(const char* name) { return namespace_+"/"+name; }
};
