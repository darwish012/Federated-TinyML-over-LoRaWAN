#pragma once
#include "tf_machine.h"
namespace tinyfl { namespace v4 {
// Extract each acquisition only once; firmware can stream 255 models from
// flash one at a time without retaining 255 float32 weight arrays in RAM.
class FeatureReplay {
 public:
  bool begin(TfFileSystem* fs,const TfMachineInfo& info,uint32_t next) {
    info_=info; next_=next;
    for(int s=0;s<info.nStreams;s++) {
      char p[16]; tf_raw_name(p,info.position,s);
      if(!src_[s].open(fs,p)) return false;
      const auto& h=src_[s].header();
      if(h.position!=info.position || h.nStreams!=info.nStreams || h.streamIndex!=s ||
         h.nAcq!=info.nAcq || h.trainStart()!=info.part.trainStart || h.trainCount()!=info.part.trainCount ||
         h.calStart()!=info.part.calStart || h.calCount()!=info.part.calCount) return false;
      if(next && next<info.nAcq && !src_[s].seekAcquisition(next)) return false;
    } return next<=info.nAcq;
  }
  bool step(float* x) {
    if(next_>=info_.nAcq) return false;
    for(int s=0;s<info_.nStreams;s++) {
      float raw[kFeat],v[kFeat]; if(!ex_.extract(src_[s],raw)) return false;
      tf_std_apply(raw,info_.std.centre[s],info_.std.scale[s],v);
      for(int j=0;j<kFeat;j++) if(!s || v[j]>x[j]) x[j]=v[j];
    } next_++; return true;
  }
  bool finish() {
    for(int s=0;s<info_.nStreams;s++) {
      bool ok=false; if(src_[s].finishDataCrc(&ok) && !ok) return false;
    } return true; // seek/resume already preflighted the entire input CRC.
  }
  void releaseFFT() { ex_.release(); }
 private:
  TfMachineInfo info_{}; uint32_t next_=0;
  TfSdReplaySource src_[kMaxStreams]; TfExtractor ex_;
};
} }
