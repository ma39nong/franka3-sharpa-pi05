// First-fault capture shared by the realtime loop and ROS callback.
// No allocation, logging, middleware calls or waiting on the realtime path.
#pragma once
#include <atomic>
#include <cstdint>
#include <sstream>
#include <string>

namespace pi05 {
enum class Fault {
  none, update_gap, clock_reversed, previous_command_expired,
  current_command_expired, future_command, invalid_window, decode_failed,
  sequence_mismatch, run_id_mismatch
};
inline const char* fault_name(Fault reason) {
  switch (reason) {
    case Fault::none: return "none";
    case Fault::update_gap: return "update_gap";
    case Fault::clock_reversed: return "clock_reversed";
    case Fault::previous_command_expired: return "previous_command_expired";
    case Fault::current_command_expired: return "current_command_expired";
    case Fault::future_command: return "future_command";
    case Fault::invalid_window: return "invalid_window";
    case Fault::decode_failed: return "decode_failed";
    case Fault::sequence_mismatch: return "sequence_mismatch";
    case Fault::run_id_mismatch: return "run_id_mismatch";
  }
  return "unknown";
}
inline Fault deadline_fault(std::int64_t created, std::int64_t expiry,
                            std::int64_t now, bool previous) {
  if (created <= 0 || expiry <= created || expiry - created > 20'000'001)
    return Fault::invalid_window;
  if (now < created) return Fault::future_command;
  if (now >= expiry) return previous ? Fault::previous_command_expired : Fault::current_command_expired;
  return Fault::none;
}
inline Fault update_fault(std::int64_t last, std::int64_t now, bool has_command,
                          std::int64_t created, std::int64_t expiry) {
  if (last != 0 && now < last) return Fault::clock_reversed;
  if (last != 0 && now - last > 50'000'000) return Fault::update_gap;
  return has_command ? deadline_fault(created, expiry, now, true) : Fault::none;
}
inline Fault identity_fault(bool first, bool same_run, std::uint64_t expected, std::uint64_t received) {
  if (!first && !same_run) return Fault::run_id_mismatch;
  return expected != received ? Fault::sequence_mismatch : Fault::none;
}
struct FaultRecord {
  Fault reason{Fault::none};
  std::int64_t now{0}, created{0}, expiry{0}, update_gap{0};
  std::uint64_t expected{0}, received{0};
};
class FaultLatch {
 public:
  bool capture(const FaultRecord& record) {
    if (record.reason == Fault::none) return false;
    unsigned expected = 0;
    if (!state_.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) return false;
    record_ = record;
    state_.store(2, std::memory_order_release);
    return true;
  }
  bool faulted() const { return state_.load(std::memory_order_acquire) != 0; }
  bool read(FaultRecord& out) const {
    if (state_.load(std::memory_order_acquire) != 2) return false;
    out = record_;
    return true;
  }
 private:
  static_assert(std::atomic<unsigned>::is_always_lock_free, "Fault latch must be lock-free");
  std::atomic<unsigned> state_{0};
  FaultRecord record_;
};
// Only called by the non-realtime ROS status timer. run is a decoded hex ID.
inline std::string status_json(const FaultLatch& latch, std::int64_t stamp,
                               const std::string& run, std::uint64_t sequence) {
  FaultRecord record;
  const bool ready = latch.read(record);
  const bool fault = ready || latch.faulted();
  std::ostringstream out;
  out << "{\"version\":1,\"stamp_ns\":" << stamp
      << ",\"run_id\":\"" << run << "\",\"sequence\":" << sequence
      << ",\"faulted\":" << (fault ? "true" : "false")
      << ",\"reason\":\"" << (ready ? fault_name(record.reason) : (fault ? "fault_pending" : "none"))
      << "\",\"fault_time_ns\":" << record.now
      << ",\"command_created_ns\":" << record.created
      << ",\"command_expiry_ns\":" << record.expiry
      << ",\"expected_sequence\":" << record.expected
      << ",\"received_sequence\":" << record.received
      << ",\"update_gap_ns\":" << record.update_gap
      << ",\"command_age_ms\":" << (record.created > 0 ? (record.now - record.created) / 1e6 : 0)
      << ",\"deadline_remaining_ms\":" << (record.expiry > 0 ? (record.expiry - record.now) / 1e6 : 0)
      << "}";
  return out.str();
}
}  // namespace pi05
