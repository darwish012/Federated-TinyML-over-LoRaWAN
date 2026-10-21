#include "TinyFLMulticast.h"

#include <string.h>

#include <lmic.h>
#include <lmic/oslmic.h>
#include <lmic/lorabase.h>
// os_aes() is a static inline in lmic_aes_api.h, which also pulls in the
// AESkey/AESaux buffers. Declaring it extern instead would shadow the inline
// and leave an undefined symbol at link time.
#include <aes/lmic_aes_api.h>

#if !defined(LMIC_ENABLE_event_logging) || !LMIC_ENABLE_event_logging
#error "TinyFLMulticast needs -DLMIC_ENABLE_event_logging=1; without it LMIC never calls the hook this library relies on."
#endif

#if !defined(LMIC_ENABLE_class_c) || !LMIC_ENABLE_class_c
#error "TinyFLMulticast needs -DLMIC_ENABLE_class_c=1; a Class A node is not listening when the multicast frame is sent."
#endif

namespace {

// The exact literal LMIC passes at the point it discards a foreign-address
// downlink. See decodeFrame() in lmic/lmic.c.
const char kWrongAddressMessage[] = "decodeFrame: wrong address";

bool gEnabled = false;
uint32_t gDevAddr = 0;
uint8_t gNwkSKey[16];
uint8_t gAppSKey[16];
uint32_t gSeqnoDn = 0;
bool gHaveSeqno = false;

uint8_t gPayload[TinyFLMulticast::kMaxPayload];
size_t gPayloadLen = 0;
uint8_t gPayloadPort = 0;
volatile bool gPayloadReady = false;

uint32_t gAccepted = 0;
uint32_t gRejected = 0;
uint32_t gOverrun = 0;

// Mirrors micB0() in se/drivers/default/lmic_se_default.c.
void micB0(u4_t devaddr, u4_t seqno, int dndir, int len) {
  os_clearMem(AESaux, 16);
  AESaux[0] = 0x49;
  AESaux[5] = dndir ? 1 : 0;
  AESaux[15] = len;
  os_wlsbf4(AESaux + 6, devaddr);
  os_wlsbf4(AESaux + 10, seqno);
}

bool verifyMic(const uint8_t *key, u4_t devaddr, u4_t seqno, int dndir,
               const uint8_t *pdu, int len) {
  micB0(devaddr, seqno, dndir, len);
  os_copyMem(AESkey, key, 16);
  return os_aes(AES_MIC, const_cast<uint8_t *>(pdu), len) ==
         os_rmsbf4(pdu + len);
}

// Mirrors aes_cipher() in the default secure element driver.
void decryptPayload(const uint8_t *key, u4_t devaddr, u4_t seqno, int dndir,
                    uint8_t *payload, int len) {
  if (len <= 0) {
    return;
  }
  os_clearMem(AESaux, 16);
  AESaux[0] = AESaux[15] = 1;  // mode=cipher, block counter=1
  AESaux[5] = dndir ? 1 : 0;
  os_wlsbf4(AESaux + 6, devaddr);
  os_wlsbf4(AESaux + 10, seqno);
  os_copyMem(AESkey, key, 16);
  os_aes(AES_CTR, payload, len);
}

// Decode the frame LMIC is about to discard. Returns true if it belonged to
// our multicast group and was accepted.
bool tryDecodeMulticast() {
  if (!gEnabled) {
    return false;
  }
  const uint8_t *d = LMIC.radio.pFrame;
  const int dlen = static_cast<int>(LMIC.dataLen);
  if (d == nullptr || dlen < static_cast<int>(OFF_DAT_OPTS) + 4) {
    return false;
  }

  const uint8_t hdr = d[0];
  if ((hdr & HDR_MAJOR) != HDR_MAJOR_V1) {
    return false;
  }
  // LoRaWAN forbids confirmed downlinks to a multicast address: there is no
  // single device that could acknowledge them.
  if ((hdr & HDR_FTYPE) != HDR_FTYPE_DADN) {
    return false;
  }

  const u4_t addr = os_rlsbf4(&d[OFF_DAT_ADDR]);
  if (addr != gDevAddr) {
    return false;  // some other network's traffic; stay silent about it
  }

  // From here on the frame is addressed to us, so a rejection is worth counting.
  const int fct = d[OFF_DAT_FCT];
  // Multicast frames carry no MAC commands and acknowledge nothing.
  if ((fct & FCT_OPTLEN) != 0 || (fct & FCT_ACK) != 0) {
    ++gRejected;
    return false;
  }

  const int poff = OFF_DAT_OPTS;
  const int pend = dlen - 4;
  if (pend <= poff) {
    ++gRejected;  // no FPort/payload
    return false;
  }

  // Extend the 16-bit counter to 32 bits the same way LMIC does; the full
  // value feeds the MIC, so it has to agree with the network server exactly.
  const u4_t seqno16 = os_rlsbf2(&d[OFF_DAT_SEQNO]);
  u4_t seqno;
  if (!gHaveSeqno) {
    seqno = seqno16;  // first frame of the session sets the epoch
  } else {
    const u2_t seqnoDiff = static_cast<u2_t>(seqno16 - gSeqnoDn);
    seqno = (seqnoDiff == 0xFFFFu) ? gSeqnoDn - 1 : gSeqnoDn + seqnoDiff;
  }

  if (!verifyMic(gNwkSKey, addr, seqno, 1, d, pend)) {
    ++gRejected;
    return false;
  }
  // Only now that the MIC proves authenticity is it safe to trust the counter.
  if (gHaveSeqno && seqno < gSeqnoDn) {
    ++gRejected;  // replay
    return false;
  }

  int payloadOffset = poff;
  const uint8_t port = d[payloadOffset++];
  if (port == 0) {
    ++gRejected;  // FPort 0 is MAC traffic, never valid on a multicast group
    return false;
  }

  const int payloadLen = pend - payloadOffset;
  if (payloadLen < 0 ||
      static_cast<size_t>(payloadLen) > TinyFLMulticast::kMaxPayload) {
    ++gRejected;
    return false;
  }
  if (gPayloadReady) {
    ++gOverrun;  // loop() has not drained the previous frame
    return false;
  }

  // Decrypt in place. LMIC discards this buffer immediately afterwards.
  decryptPayload(gAppSKey, addr, seqno, 1,
                 const_cast<uint8_t *>(&d[payloadOffset]), payloadLen);

  memcpy(gPayload, &d[payloadOffset], static_cast<size_t>(payloadLen));
  gPayloadLen = static_cast<size_t>(payloadLen);
  gPayloadPort = port;
  gSeqnoDn = seqno + 1;
  gHaveSeqno = true;
  ++gAccepted;
  gPayloadReady = true;
  return true;
}

}  // namespace

// LMIC calls these when built with LMIC_ENABLE_event_logging=1. They must exist
// for the link to succeed, and they are called from many places, so everything
// except the one message we care about has to stay cheap.
extern "C" void LMICOS_logEvent(const char *pMessage) { (void)pMessage; }

extern "C" void LMICOS_logEventUint32(const char *pMessage, uint32_t datum) {
  (void)datum;
  if (pMessage == nullptr) {
    return;
  }
  if (strcmp(pMessage, kWrongAddressMessage) != 0) {
    return;
  }
  tryDecodeMulticast();
}

namespace TinyFLMulticast {

void begin(uint32_t devAddr, const uint8_t nwkSKey[16],
           const uint8_t appSKey[16], uint32_t restoredCounter,
           bool haveRestoredCounter) {
  gDevAddr = devAddr;
  memcpy(gNwkSKey, nwkSKey, sizeof(gNwkSKey));
  memcpy(gAppSKey, appSKey, sizeof(gAppSKey));
  // Carry the replay window across the reboot. Clearing it here is what used
  // to let a captured burst be replayed at the node after a power cycle, and
  // what broke MIC verification once the server counter passed 65535.
  gSeqnoDn = haveRestoredCounter ? restoredCounter : 0;
  gHaveSeqno = haveRestoredCounter;
  gPayloadReady = false;
  gPayloadLen = 0;
  gEnabled = true;
}

void end() { gEnabled = false; }

bool isEnabled() { return gEnabled; }

bool poll(uint8_t *out, size_t outSize, size_t *outLen, uint8_t *port) {
  if (!gPayloadReady) {
    return false;
  }
  const size_t length = gPayloadLen < outSize ? gPayloadLen : outSize;
  if (out != nullptr) {
    memcpy(out, gPayload, length);
  }
  if (outLen != nullptr) {
    *outLen = length;
  }
  if (port != nullptr) {
    *port = gPayloadPort;
  }
  gPayloadReady = false;
  return true;
}

uint32_t acceptedCount() { return gAccepted; }
uint32_t rejectedCount() { return gRejected; }
uint32_t overrunCount() { return gOverrun; }
uint32_t frameCounter() { return gSeqnoDn; }

bool hasCounter() { return gHaveSeqno; }

}  // namespace TinyFLMulticast
