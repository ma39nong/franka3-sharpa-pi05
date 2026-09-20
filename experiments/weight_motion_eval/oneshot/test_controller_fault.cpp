#include "controller_fault.hpp"
#include <cassert>
#include <iostream>
#include <thread>

int main() {
  using pi05::Fault;
  constexpr std::int64_t created = 5'000'000'000, expiry = created + 20'000'000;
  assert(pi05::deadline_fault(created, expiry, expiry - 1, true) == Fault::none);
  assert(pi05::deadline_fault(created, expiry, expiry, true) == Fault::previous_command_expired);
  assert(pi05::deadline_fault(created, expiry, expiry, false) == Fault::current_command_expired);
  assert(pi05::deadline_fault(created, expiry, created - 1, false) == Fault::future_command);
  assert(pi05::deadline_fault(created, created + 21'000'000, created, false) == Fault::invalid_window);
  assert(pi05::update_fault(created, created - 1, false, 0, 0) == Fault::clock_reversed);
  assert(pi05::update_fault(created, created + 50'000'001, false, 0, 0) == Fault::update_gap);
  assert(pi05::identity_fault(false, true, 504, 505) == Fault::sequence_mismatch);
  assert(pi05::identity_fault(false, false, 504, 504) == Fault::run_id_mismatch);
  assert(pi05::identity_fault(true, false, 0, 0) == Fault::none);
  assert(pi05::identity_fault(true, false, 0, 1) == Fault::sequence_mismatch);
  // Replay the suspected failure: old frame expires before the next arrives.
  pi05::FaultLatch latch;
  auto reason = pi05::update_fault(expiry - 1'000'000, expiry, true, created, expiry);
  assert(latch.capture({reason, expiry, created, expiry, 1'000'000, 504, 503}));
  assert(pi05::deadline_fault(created + 10'000'000, expiry + 10'000'000, expiry + 500'000, false) == Fault::none);
  assert(latch.faulted()); // A fresh next frame must NOT revive the stream.
  assert(!latch.capture({Fault::sequence_mismatch, expiry + 500'000, 0, 0, 0, 504, 505}));
  pi05::FaultRecord first;
  assert(latch.read(first) && first.reason == Fault::previous_command_expired && first.received == 503);
  // Concurrent RT/callback capture keeps one complete, immutable snapshot.
  for (int i = 0; i < 100; ++i) {
    pi05::FaultLatch concurrent;
    std::thread a([&] { concurrent.capture({Fault::update_gap, 11, 12, 13, 14, 15, 16}); });
    std::thread b([&] { concurrent.capture({Fault::decode_failed, 21, 22, 23, 24, 25, 26}); });
    a.join(); b.join();
    assert(concurrent.read(first));
    assert((first.reason == Fault::update_gap && first.now == 11 && first.received == 16) ||
           (first.reason == Fault::decode_failed && first.now == 21 && first.received == 26));
  }
  std::cout << pi05::status_json(latch, expiry + 1'000'000, std::string(32, 'a'), 503);
}
