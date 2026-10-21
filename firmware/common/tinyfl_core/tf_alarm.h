// Alarm layer: research/healthy_only_causal_20260908/cadence.py dense_flags(
// scores, train_idx, calib_idx, window=21, k=13, multiplier=2.0), evaluated
// causally, one score at a time, with O(1) memory per (machine, model).
//
// What the reference does, exactly:
//   * stat[i] = the 13th largest of the 21 scores ending at i (-inf before 21).
//   * reference = max over the same statistic computed on the HISTORY rows
//     (train rows then calibration rows, concatenated, windowed as if adjacent)
//     that are finite; if the history is shorter than 21 rows it falls back to
//     the 0.995 quantile of the calibration scores.
//   * threshold = 2.0 * reference.
//   * flag[i] = stat[i] > threshold, and every flag up to and including the last
//     calibration index is forced False ("suppressed").
// The alarm LATCHES at the first flag after calibration; later flags are counted.
#pragma once

#include "tf_common.h"

namespace tinyfl {

struct TfPartition {
  uint32_t trainStart, trainCount, calStart, calCount;
  uint32_t calEnd() const { return calStart + calCount - 1u; }
  bool inTrain(uint32_t i) const { return i >= trainStart && i - trainStart < trainCount; }
  bool inCal(uint32_t i) const { return i >= calStart && i - calStart < calCount; }
};

struct TfAlarmState {
  float ring[kAlarmWindow];    // last 21 scores, every row
  float hring[kAlarmWindow];   // last 21 scores, history rows only
  float calScores[kAlarmWindow];
  double threshold;
  float refMax;
  uint32_t firstAlarm;
  uint32_t flagCount;
  uint32_t lastFlag;
  uint32_t nCal;
  uint8_t ringN, ringPos;
  uint8_t hN, hPos;
  uint8_t nCalStored;
  uint8_t haveRef;
  uint8_t thresholdReady;
  uint8_t latched;
  uint8_t sent;                // the ALARM uplink for this state has been queued
  uint8_t pad[3];
};

struct TfAlarmEvent {
  bool flag;           // stat > threshold and past calibration
  bool newlyLatched;   // first flag of this state
  float stat;          // the 13-of-21 statistic (-inf until 21 scores exist)
  double threshold;    // valid once the calibration has ended
  bool thresholdReady;
};

void tf_alarm_init(TfAlarmState* s);
// Feed the score of acquisition `index`; indices must arrive 0,1,2,... in order.
TfAlarmEvent tf_alarm_step(TfAlarmState* s, const TfPartition& part, uint32_t index, float score);

}  // namespace tinyfl
