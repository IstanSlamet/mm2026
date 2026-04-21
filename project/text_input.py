#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading
import sys

class TextInputNode(Node):
    def __init__(self):
        super().__init__('text_input_node')
        self.target_pub = self.create_publisher(String, '/task/target_object', 10)
        self.get_logger().info('--- Standalone Text Input Node Started ---')

    def run_cli(self):
        """Dedicated thread for terminal interaction."""
        while rclpy.ok():
            # Use sys.stdout.write to ensure the prompt appears immediately
            sys.stdout.write("\n[Command] Type object to find (or 'quit'): ")
            sys.stdout.flush()
            
            target = sys.stdin.readline().strip()
            
            if target.lower() == 'quit':
                break
            
            if target:
                msg = String()
                msg.data = target
                self.target_pub.publish(msg)
                self.get_logger().info(f'Published target: "{target}"')

def main():
    rclpy.init()
    node = TextInputNode()
    
    # Run the input loop in a daemon thread so it doesn't block rclpy.spin
    cli_thread = threading.Thread(target=node.run_cli, daemon=True)
    cli_thread.start()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()