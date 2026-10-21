#pragma once
// Copy this to lab_credentials.h locally; never commit real keys. The sketch
// also accepts the existing ../tinyfl_node/lab_credentials.h without copying it.
#define TINYFL_CREDENTIALS_CONFIGURED 0
#define TINYFL_JOIN_EUI {0,0,0,0,0,0,0,0} // LMIC little-endian JoinEUI
#define TINYFL_APP_KEY {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
#define TINYFL_MULTICAST_DEVADDR 0u
#define TINYFL_MULTICAST_NWKSKEY {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
#define TINYFL_MULTICAST_APPSKEY {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
// Bit i selects the CANONICAL level index, not the subset's ordinal index.
// Must match server sweep configuration before flashing (default all 7).
#ifndef TINYFL_RUN_LEVEL_MASK
#define TINYFL_RUN_LEVEL_MASK 0x7f
#endif
#ifndef TINYFL_SD_MMC
#define TINYFL_SD_MMC 1 // ESP32 fixed CLK14 CMD15 D0=2, 1-bit; 0 => SPI
#endif
#ifndef TINYFL_PHASE_MARKERS
#define TINYFL_PHASE_MARKERS 1
#endif
