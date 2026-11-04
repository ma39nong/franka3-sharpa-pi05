"""Pinned reference from LTT-BIT/tomato-openpi e64c9a4, flash_native.py.

Only the two pure preprocessing methods; dataset loading/training is excluded.
Upstream Apache-2.0 (deployment repository LICENSE). Keep numerics unchanged.
"""
class NativeDataset:

    @staticmethod
    def resize_image(rgb):
        pixels = rgb.astype(np.float32) / 127.5 - 1
        height, width = pixels.shape[:2]
        ratio = max(width / 224, height / 224)
        resized_h, resized_w = (int(height / ratio), int(width / ratio))
        tensor = torch.from_numpy(pixels).permute(2, 0, 1)[None]
        value = F.interpolate(tensor, size=(resized_h, resized_w), mode='bilinear', align_corners=False, antialias=True)[0].permute(1, 2, 0).numpy()
        top, left = ((224 - resized_h) // 2, (224 - resized_w) // 2)
        return np.pad(value, ((top, 224 - resized_h - top), (left, 224 - resized_w - left), (0, 0)), constant_values=-1)[None]

    def observation(self, index):
        history, target, state, prompt = self.window(index)
        stat = self.stats['state']
        normalized = (state - stat.q01) / (stat.q99 - stat.q01 + 1e-06) * 2 - 1
        tokens, mask = self.tokenizer.tokenize(prompt, normalized)
        images = {key: self.resize_image(self.image(index, key)) for key in CAMERAS}
        obs = model_lib.Observation(images=images, image_masks={k: np.ones(1, bool) for k in images}, state=np.asarray(normalized[None], np.float32), tokenized_prompt=np.asarray(tokens[None], np.int32), tokenized_prompt_mask=np.asarray(mask[None]))
        return (obs, history[None], target[None])
