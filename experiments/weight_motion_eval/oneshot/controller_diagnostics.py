"""Add first-fault capture/status to the Pi05 overlay without relaxing guards.

Also called by prepare_overlay so a newly generated overlay has the same code.
Running this module upgrades the existing source; build_overlay.sh must follow.
"""

from pathlib import Path
import shutil


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Unexpected overlay source; diagnostic patch is ambiguous")
    return text.replace(old, new, 1)


def instrument(header, source):
    if 'pi05_controller_fault.hpp' in header:
        if 'reportPi05Status_' not in source:
            raise ValueError("Incomplete controller diagnostics patch")
        source = source.replace('get_node()->create_publisher<std_msgs::msg::String>("~/pi05_status", 1)',
                                'rclcpp::create_publisher<std_msgs::msg::String>(get_node(), "~/pi05_status", rclcpp::QoS(1))')
        return header, source
    header = once(header, '#include <atomic>', '#include <atomic>\n#include <std_msgs/msg/string.hpp>\n#include "franka_fr3_arm_controllers/pi05_controller_fault.hpp"')
    header = once(header, '    std::int64_t expiry_ns{0};', '    std::int64_t expiry_ns{0};\n    std::uint64_t envelope_sequence{0};')
    header = once(header, '  std::atomic<bool> pi05_fault_{false};', '''  pi05::FaultLatch pi05_fault_;
  bool pi05_fault_reported_{false};
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pi05_status_publisher_;
  rclcpp::TimerBase::SharedPtr pi05_status_timer_;
  void reportPi05Status_();''')
    begin = source.index('  if ((pi05_last_update_ != 0')
    end = source.index('  pi05_last_update_ = pi05_now;', begin)
    source = source[:begin] + '''  const auto reason = pi05::update_fault(
      pi05_last_update_, pi05_now, command->sequence != 0,
      command->source_stamp.nanoseconds(), command->expiry_ns);
  pi05_fault_.capture({reason, pi05_now, command->source_stamp.nanoseconds(),
      command->expiry_ns, pi05_last_update_ == 0 ? 0 : pi05_now - pi05_last_update_,
      command->envelope_sequence + 1, command->envelope_sequence});
''' + source[end:]
    source = once(source, 'if (pi05_fault_.load())', 'if (pi05_fault_.faulted())')
    begin = source.index('  if (pi05_fault_.load() ||')
    end = source.index('  pi05_run_ = envelope.run;', begin)
    source = source[:begin] + '''  if (pi05_fault_.faulted()) return;
  auto reason = pi05::Fault::none;
  const bool decoded = pi05::decode(msg.header.frame_id, envelope);
  const auto expected = pi05_run_.empty() ? 0 : pi05_sequence_ + 1;
  if (!decoded) reason = pi05::Fault::decode_failed;
  else {
    reason = pi05::deadline_fault(pi05_created, envelope.expiry, pi05_now, false);
    if (reason == pi05::Fault::none)
      reason = pi05::identity_fault(pi05_run_.empty(), envelope.run == pi05_run_, expected, envelope.sequence);
  }
  if (reason != pi05::Fault::none) {
    pi05_fault_.capture({reason, pi05_now, pi05_created, envelope.expiry, 0, expected, envelope.sequence});
    return;
  }
''' + source[end:]
    source = once(source, '  command.expiry_ns = envelope.expiry;', '  command.expiry_ns = envelope.expiry;\n  command.envelope_sequence = envelope.sequence;')
    source = once(source, '  command_status_timer_ =', '''  pi05_status_publisher_ = rclcpp::create_publisher<std_msgs::msg::String>(get_node(), "~/pi05_status", rclcpp::QoS(1));
  pi05_status_timer_ = get_node()->create_wall_timer(10ms, [this]() { reportPi05Status_(); });
  command_status_timer_ =''')
    marker = 'void JointImpedanceController::reportCommandStatus_() {'
    source = once(source, marker, '''void JointImpedanceController::reportPi05Status_() {
  std_msgs::msg::String msg;
  msg.data = pi05::status_json(pi05_fault_, get_node()->now().nanoseconds(), pi05_run_, pi05_sequence_);
  // Publish outside the realtime loop; log only the first completed snapshot.
  pi05_status_publisher_->publish(msg);
  pi05::FaultRecord first;
  if (!pi05_fault_reported_ && pi05_fault_.read(first)) {
    pi05_fault_reported_ = true;
    RCLCPP_ERROR(get_node()->get_logger(), "Pi05 controller fault: %s", msg.data.c_str());
  }
}

''' + marker)
    return header, source


def install(target):
    include = target / 'include/franka_fr3_arm_controllers'
    header = include / 'joint_impedance_controller.hpp'
    source = target / 'src/joint_impedance_controller.cpp'
    h, s = instrument(header.read_text(), source.read_text())
    cmake = target / 'CMakeLists.txt'
    c = cmake.read_text()
    if 'find_package(std_msgs REQUIRED)' not in c:
        c = once(c, 'find_package(rclcpp REQUIRED)', 'find_package(rclcpp REQUIRED)\nfind_package(std_msgs REQUIRED)')
        c = once(c, '        ${PROJECT_NAME}\n        controller_interface', '        ${PROJECT_NAME}\n        std_msgs\n        controller_interface')
        c = once(c, 'ament_export_dependencies(\n', 'ament_export_dependencies(\n        std_msgs\n')
    if 'add_executable(pi05_controller_status_check' not in c:
        c = once(c, 'ament_package()', '''add_executable(pi05_controller_status_check test/pi05_controller_status_check.cpp)
target_include_directories(pi05_controller_status_check PRIVATE include)
ament_target_dependencies(pi05_controller_status_check rclcpp rclcpp_lifecycle std_msgs)
ament_package()''')
    package = target / 'package.xml'
    p = package.read_text().replace('<exec_depend>std_msgs</exec_depend>', '<depend>std_msgs</depend>')
    # All transformations validate before writing any source.
    header.write_text(h)
    source.write_text(s)
    cmake.write_text(c)
    package.write_text(p)
    shutil.copy2(Path(__file__).with_name('controller_fault.hpp'), include / 'pi05_controller_fault.hpp')
    (target / 'test').mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).with_name('test_controller_publisher.cpp'), target / 'test/pi05_controller_status_check.cpp')


if __name__ == '__main__':
    from .prepare_overlay import OUTPUT
    install(OUTPUT / 'src/franka_fr3_arm_controllers')
    print('Updated Pi05 controller diagnostics source; rebuild the overlay before execution.')
