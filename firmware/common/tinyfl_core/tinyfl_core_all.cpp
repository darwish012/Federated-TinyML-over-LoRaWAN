// Single translation unit for the Arduino build: arduino-cli compiles only files
// inside the sketch folder and libraries, not ../common, so the sketch includes
// this file.  Host builds compile the .cpp files individually instead.
#include "tf_common.cpp"
#include "tf_fft.cpp"
#include "tf_features.cpp"
#include "tf_rawfile.cpp"
#include "tf_model.cpp"
#include "tf_alarm.cpp"
#include "tf_protocol.cpp"
#include "tf_machine.cpp"
