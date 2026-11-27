"""One Annex-B access unit per native RGB frame; never on the control thread."""
from fractions import Fraction
import numpy as np
import av


class VideoEncoder:
    def __init__(self, channel):
        self.channel = channel
        self.codec = av.CodecContext.create('libx264', 'w')
        self.codec.width = channel['width']; self.codec.height = channel['height']
        self.codec.pix_fmt = 'yuv420p'; self.codec.time_base = Fraction(1, 1_000_000_000)
        self.codec.framerate = Fraction(str(channel['fps']))
        self.codec.thread_count = 1
        self.codec.max_b_frames = 0
        self.codec.flags |= av.codec.context.Flags.low_delay
        self.codec.options = dict(preset='medium', **{'rc-lookahead':'0'}, crf=str(channel['crf']), g='30', keyint_min='30',
            sc_threshold='0', bf='0', **{'x264-params':'open-gop=0:repeat-headers=1:annexb=1:rc-lookahead=0:sync-lookahead=0:force-cfr=1',
            'color_range':'tv', 'colorspace':'bt709', 'color_primaries':'bt709', 'color_trc':'iec61966-2-1'})
        self.codec.open()

    def encode(self, message, stamp):
        if (message.width, message.height) != (self.codec.width, self.codec.height):
            raise ValueError('Camera native size changed during episode')
        if message.encoding not in ('rgb8','bgr8'):
            raise ValueError(f'Unsupported native camera encoding: {message.encoding}')
        pixels = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
        pixels = pixels[:, :message.width*3].reshape(message.height,message.width,3)
        frame = av.VideoFrame.from_ndarray(pixels, format=message.encoding.replace('8','24'))
        frame = frame.reformat(format='yuv420p', src_colorspace='ITU709', dst_colorspace='ITU709',
                               src_color_range='JPEG', dst_color_range='MPEG')
        frame.pts = stamp; frame.time_base = self.codec.time_base
        packets = self.codec.encode(frame)
        if len(packets) != 1:
            raise RuntimeError('Low-latency H.264 encoder did not produce exactly one access unit')
        return bytes(packets[0])

    def close(self):
        if self.codec.encode(None):
            raise RuntimeError('Unexpected buffered H.264 frames')
