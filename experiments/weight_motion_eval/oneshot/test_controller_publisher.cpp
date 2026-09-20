// ROS transport test for the exact publisher API used by the controller.
// Only nodes and a synthetic fault are created; no controller or driver loads.
#include "franka_fr3_arm_controllers/pi05_controller_fault.hpp"
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <std_msgs/msg/string.hpp>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>

int main(int argc, char** argv) {
  if (!std::getenv("ROS_DOMAIN_ID") || std::string(std::getenv("ROS_DOMAIN_ID")) != "197")
    throw std::runtime_error("Use isolated ROS_DOMAIN_ID=197");
  rclcpp::init(argc, argv);
  auto controller = std::make_shared<rclcpp_lifecycle::LifecycleNode>("joint_impedance_controller", "/left");
  auto monitor = std::make_shared<rclcpp::Node>("pi05_status_test");
  // Ordinary publisher is intentional: diagnostics work during preparation,
  // active motion AND after a lifecycle fault; no activation gate hides them.
  auto publisher = rclcpp::create_publisher<std_msgs::msg::String>(controller, "~/pi05_status", rclcpp::QoS(1));
  bool received = false;
  auto subscription = monitor->create_subscription<std_msgs::msg::String>(
      "/left/joint_impedance_controller/pi05_status", 1,
      [&](const std_msgs::msg::String& msg) {
        received = msg.data.find("previous_command_expired") != std::string::npos;
      });
  pi05::FaultLatch fault;
  const auto now = controller->now().nanoseconds();
  fault.capture({pi05::Fault::previous_command_expired, now, now - 20'000'000, now, 1'000'000, 504, 503});
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(monitor);
  executor.add_node(controller->get_node_base_interface());
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (!received && std::chrono::steady_clock::now() < deadline) {
    std_msgs::msg::String msg;
    msg.data = pi05::status_json(fault, controller->now().nanoseconds(), std::string(32, 'a'), 503);
    publisher->publish(msg);
    executor.spin_some();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  if (!received) throw std::runtime_error("Controller status publisher did not reach its absolute topic");
  std::cout << "C++ lifecycle-node diagnostic publisher: passed (no hardware)" << std::endl;
  rclcpp::shutdown();
}
