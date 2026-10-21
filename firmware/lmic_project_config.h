#pragma once

// Copy these settings into the MCCI LMIC library project configuration, or
// use build_firmware.ps1 which supplies the same definitions as build flags.
#define CFG_eu868 1
#define CFG_sx1276_radio 1
#define LMIC_ENABLE_class_c 1
#define LMIC_ENABLE_long_messages 1

#define DISABLE_BEACONS 1
#define DISABLE_PING 1
#define LMIC_ENABLE_DeviceTimeReq 0
