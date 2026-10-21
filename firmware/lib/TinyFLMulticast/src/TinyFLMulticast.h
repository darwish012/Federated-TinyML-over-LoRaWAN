/// \file TinyFLMulticast.h
///
/// LoRaWAN Class C multicast reception as a companion to MCCI LMIC.
///
/// MCCI LMIC 6.0.1 has no multicast session: its only Class C entry point is
/// LMIC_enableClassC(), which operates on the device's own unicast session, and
/// decodeFrame() discards any downlink whose DevAddr is not LMIC.devaddr.
///
/// Rather than fork the stack, this library hooks the seam LMIC already
/// provides at precisely that discard point. When built with
/// LMIC_ENABLE_event_logging=1, LMIC calls the application-supplied
/// LMICOS_logEventUint32("decodeFrame: wrong address", addr) while the raw
/// frame is still intact in LMIC.radio.pFrame. This library implements that
/// function, and for frames addressed to the configured multicast group it
/// verifies the MIC and decrypts the payload using LMIC's own AES engine.
///
/// The stock MCCI LMIC installation is not modified in any way.
///
/// Build requirements:
///   -DLMIC_ENABLE_event_logging=1   (enables the hook)
///   -DLMIC_ENABLE_class_c=1         (the node must be listening continuously)
///
/// Received frames are buffered, not dispatched from inside the LMIC call
/// stack. Call TinyFLMulticast::poll() from loop() after os_runloop_once();
/// calling back into LMIC from inside decodeFrame() would be re-entrant.

#pragma once

#include <Arduino.h>
#include <lmic.h>

namespace TinyFLMulticast {

/// Largest application payload this library will buffer. EU868 DR4/DR5 allow
/// 222 application bytes; 242 covers every EU868 data rate.
constexpr size_t kMaxPayload = 242;

/// Provision the multicast session. DevAddr and both session keys must match
/// the multicast group defined on the network server.
///
/// The restored counter is not optional. A multicast session has static keys
/// and a static DevAddr, so a node that starts with no counter accepts the
/// first authentic frame it hears whatever its counter says -- which means a
/// burst captured off the air can be replayed at the node after any reboot,
/// and every frame still passes the MIC. Persist the value returned by
/// frameCounter() (see persistFrameCounter() in the sketch) and hand it back
/// here so the window survives a power cycle.
///
/// Restoring the full 32-bit value also keeps the counter epoch correct once
/// the server passes 65535: the 16 bits on the wire are extended relative to
/// the counter held here, so starting from zero would compute the wrong
/// 32-bit sequence number and fail every MIC check.
///
/// \param devAddr              multicast DevAddr, as a 32-bit value (not byte-reversed)
/// \param nwkSKey              16-byte network session key, in network byte order
/// \param appSKey              16-byte application session key, in network byte order
/// \param restoredCounter      last accepted 32-bit downlink counter, from NVS
/// \param haveRestoredCounter  false only for a genuinely new session, i.e.
///                             when the group's keys and counter have both
///                             been rotated on the network server
void begin(uint32_t devAddr, const uint8_t nwkSKey[16], const uint8_t appSKey[16],
           uint32_t restoredCounter, bool haveRestoredCounter);

/// Stop accepting multicast frames.
void end();

/// True once begin() has been called.
bool isEnabled();

/// Collect one received multicast payload, if any is waiting.
///
/// \param out      destination buffer
/// \param outSize  capacity of \p out
/// \param outLen   set to the number of bytes written
/// \param port     set to the FPort the frame arrived on
/// \return true if a frame was returned, false if none was waiting
///
/// Call this from loop(), not from an LMIC callback.
bool poll(uint8_t *out, size_t outSize, size_t *outLen, uint8_t *port);

/// Frames accepted and decrypted since boot.
uint32_t acceptedCount();

/// Frames addressed to the multicast group but rejected: bad MIC, replayed
/// frame counter, or a shape the LoRaWAN spec forbids for multicast.
uint32_t rejectedCount();

/// Frames dropped because a previous payload had not been polled yet. A
/// non-zero value means loop() is not draining fast enough.
uint32_t overrunCount();

/// Current multicast downlink frame counter: the lowest value a frame may
/// carry and still be accepted. Persist this and feed it back to begin().
uint32_t frameCounter();

/// True once a counter epoch is established, either restored through begin()
/// or set by the first accepted frame of a genuinely new session.
bool hasCounter();

}  // namespace TinyFLMulticast
