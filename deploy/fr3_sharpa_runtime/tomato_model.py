"""Content identity for normal93 B-coarse and complete100 four-view r11 runs.

Names and paths are not evidence. These fingerprints bind the Orbax manifest,
parameter metadata and checkpoint-owned normalization, independent of location.
Full parameter bytes/shapes are validated during Orbax restoration on GPU.
"""

STATS_SHA256 = "3f82a27ba53cb6a0553886d453c8e05ccb466354d131ee9c956cd0a12637350a"
MANIFEST_SHA256 = "55d037b0002ca3a76112ae848944663c3b602eac8c9a813847f67445050bdf60"
METADATA_SHA256 = "78df4cd15c8e0c09c250299c846da7504ec5da44b1463df3757db647beea4df0"


FOUR_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "interaction_0_rgb")
R11_STATS_SHA256 = "cadfda5b116b16f66a5b739825bd6a16321be2c09b714bef481bf4c19064cf83"
R11_MANIFEST_SHA256 = "5acf0591c985300fcd2651a96eb234bb1d395b523b03aeb382f0a2f7115bd038"

R11_STEPS = {
    "411f7df60ce51c46cf4c8a9c7791e77d656ed9dde9e5e9439ad55e4fcf32ef21": 4000,
    "c8f57d67c944b5e1ed06d7d21bba0a7cdbd575314ed7c28584f71e44f76ab0bb": 8000,
    "fe97551f016bfc8531abc91952c37908f2aa939ee3e7300c9d3efae2f7231dd6": 12000,
    R11_MANIFEST_SHA256: 16000,
}


def identify(stats_sha, manifest_sha, metadata_sha):
    if stats_sha == R11_STATS_SHA256:
        if metadata_sha != METADATA_SHA256 or manifest_sha not in R11_STEPS:
            raise ValueError("Four-view r11 statistics require a verified matching 4000/8000/12000/16000 checkpoint")
        return dict(name=f"complete100 B RGB+crop r11 · {R11_STEPS[manifest_sha]}", training_date="2026-11-14",
                    subtask_variant="b_coarse", profile="full-delta", action_dim=58,
                    action_horizon=50, action_hz=30, stage_contract="tomato-coarse-four-prompts-en-v4",
                    image_keys=list(FOUR_IMAGE_KEYS), visual_tokens=1024,
                    offline_training_roi_uses_future=True, online_roi_required=True,
                    evidence="verified checkpoint-owned statistics and Orbax fingerprints")
    if (stats_sha, manifest_sha, metadata_sha) != (STATS_SHA256, MANIFEST_SHA256, METADATA_SHA256):
        return None
    return dict(
        name="normal93 B Coarse FullFT Delta · 7999",
        training_date="2026-11-09",
        subtask_variant="b_coarse",
        profile="full-delta",
        action_dim=58,
        action_horizon=50,
        action_hz=30,
        stage_contract="tomato-coarse-four-prompts-en-v4",
        evidence="verified Orbax manifest, metadata and checkpoint-owned normalization fingerprints",
    )
