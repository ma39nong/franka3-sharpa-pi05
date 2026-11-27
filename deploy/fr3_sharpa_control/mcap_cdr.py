"""Bulk CDR encoding for standard binary messages; no per-pixel Python loop."""
from io import BytesIO


def binary_encoder(kind):
    if kind not in ('foxglove_msgs/msg/CompressedVideo', 'sensor_msgs/msg/CompressedImage', 'sensor_msgs/msg/Image'):
        return None
    def encode(message):
        from mcap_ros2._cdr import CdrWriter
        stream = BytesIO(); writer = CdrWriter(stream)
        if kind == 'foxglove_msgs/msg/CompressedVideo':
            stamp = message['timestamp']; frame_id = message['frame_id']
        else:
            stamp = message['header']['stamp']; frame_id = message['header']['frame_id']
        writer.write_int32(stamp['sec']); writer.write_uint32(stamp['nanosec']); writer.write_string(frame_id)
        if kind == 'sensor_msgs/msg/Image':
            writer.write_uint32(message['height']); writer.write_uint32(message['width'])
            writer.write_string(message['encoding']); writer.write_uint8(message['is_bigendian'])
            writer.write_uint32(message['step'])
        else:
            writer.write_string(message['format'])
        payload = bytes(message['data'])
        writer.write_uint32(len(payload)); writer.write_bytes(payload)
        return stream.getvalue()
    return encode
