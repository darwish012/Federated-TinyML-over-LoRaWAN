#include "tf_alarm.h"

namespace tinyfl {

void tf_alarm_init(TfAlarmState* s) { memset(s, 0, sizeof(*s)); }

// k-th largest of the 21 values (k = 13): index 21-13 = 8 of the ascending sort,
// exactly np.partition(w, window - k)[window - k].
static float kth_largest21(const float* v) {
  float t[kAlarmWindow];
  for (int i = 0; i < kAlarmWindow; ++i) {
    float x = v[i];
    int j = i;
    while (j > 0 && t[j - 1] > x) {
      t[j] = t[j - 1];
      --j;
    }
    t[j] = x;
  }
  return t[kAlarmWindow - kAlarmK];
}

// numpy.quantile(x, q) with the default 'linear' method.
static double quantile_linear(const float* v, int n, double q) {
  float t[kAlarmWindow];
  for (int i = 0; i < n; ++i) {
    float x = v[i];
    int j = i;
    while (j > 0 && t[j - 1] > x) {
      t[j] = t[j - 1];
      --j;
    }
    t[j] = x;
  }
  const double pos = q * (double)(n - 1);
  const int lo = (int)floor(pos);
  const int hi = (lo + 1 < n) ? lo + 1 : lo;
  const double g = pos - (double)lo;
  return (double)t[lo] + ((double)t[hi] - (double)t[lo]) * g;
}

TfAlarmEvent tf_alarm_step(TfAlarmState* s, const TfPartition& part, uint32_t index, float score) {
  TfAlarmEvent ev;
  ev.flag = false;
  ev.newlyLatched = false;
  ev.stat = -INFINITY;
  ev.threshold = s->threshold;
  ev.thresholdReady = s->thresholdReady != 0;

  // all-rows window
  s->ring[s->ringPos] = score;
  s->ringPos = (uint8_t)((s->ringPos + 1) % kAlarmWindow);
  if (s->ringN < kAlarmWindow) ++s->ringN;

  // history window (train + calibration rows)
  const bool inCal = part.inCal(index);
  if (part.inTrain(index) || inCal) {
    s->hring[s->hPos] = score;
    s->hPos = (uint8_t)((s->hPos + 1) % kAlarmWindow);
    if (s->hN < kAlarmWindow) ++s->hN;
    if (s->hN == kAlarmWindow) {
      const float hs = kth_largest21(s->hring);
      if (!s->haveRef || hs > s->refMax) s->refMax = hs;
      s->haveRef = 1;
    }
  }
  if (inCal) {
    if (s->nCalStored < kAlarmWindow) s->calScores[s->nCalStored++] = score;
    ++s->nCal;
  }
  if (index == part.calEnd() && !s->thresholdReady) {
    double ref;
    if (s->haveRef)
      ref = (double)s->refMax;
    else
      ref = quantile_linear(s->calScores, s->nCalStored, 0.995);
    s->threshold = kAlarmMultiplier * ref;
    s->thresholdReady = 1;
  }

  ev.thresholdReady = s->thresholdReady != 0;
  ev.threshold = s->threshold;
  if (s->ringN == kAlarmWindow) ev.stat = kth_largest21(s->ring);
  if (s->thresholdReady && index > part.calEnd() && (double)ev.stat > s->threshold) {
    ev.flag = true;
    ++s->flagCount;
    s->lastFlag = index;
    if (!s->latched) {
      s->latched = 1;
      s->firstAlarm = index;
      ev.newlyLatched = true;
    }
  }
  return ev;
}

}  // namespace tinyfl
