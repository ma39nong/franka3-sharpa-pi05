"""Content identity for normal93, complete100 r11 and five-stage in-hand119.

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


IN_HAND_STATS_SHA256 = "8b06358841e0570ff842883ab8a49a0fd3c9be7b68b59820d93741cd8a06ccde"
IN_HAND_STEPS = {'a48a691acbc68dc0742071970f9de31c3437666dbf0c305170e4b7e083e14345': 10000, '98f287ff0f89a3f76e2e41096fdddde35362451f5770fe15564081aac1139841': 15000, 'd43cc4708b3019e99f5162a3fde6cf92c866835e42470fc1a7028f1b877accd0': 20000}

def identify(stats_sha, manifest_sha, metadata_sha):
    if stats_sha == IN_HAND_STATS_SHA256:
        if metadata_sha != METADATA_SHA256 or manifest_sha not in IN_HAND_STEPS:
            raise ValueError("119条权重必须匹配已核对的10000/15000/20000参数与配套统计")
        return dict(name=f"in-hand119 四路五阶段 · {IN_HAND_STEPS[manifest_sha]}",
                    task_name="手中调整 · 五阶段", input_name="四路 RGB（含局部图）",
                    dataset_episodes=119, dataset_version="手中调整 v3/r1", checkpoint_step=IN_HAND_STEPS[manifest_sha],
                    training_date="2026-11-18", subtask_variant="in_hand_b_coarse", profile="full-delta",
                    action_dim=58, action_horizon=50, action_hz=30,
                    task_profile="tomato-in-hand119", stage_contract="tomato-in-hand-five-phases-v3",
                    image_keys=list(FOUR_IMAGE_KEYS), visual_tokens=1024,
                    roi_profile="in_hand119_right_truss_v1", offline_training_roi_uses_future=True,
                    online_roi_required=True, evidence="verified checkpoint-owned statistics and Orbax fingerprints")
    if stats_sha == R11_STATS_SHA256:
        if metadata_sha != METADATA_SHA256 or manifest_sha not in R11_STEPS:
            raise ValueError("Four-view r11 statistics require a verified matching 4000/8000/12000/16000 checkpoint")
        return dict(name=f"complete100 B RGB+crop r11 · {R11_STEPS[manifest_sha]}", training_date="2026-11-14",
                    task_name="放回盘中调整 · 七阶段", input_name="四路 RGB（含局部图）",
                    dataset_episodes=100, dataset_version="整串采摘 r11", checkpoint_step=R11_STEPS[manifest_sha],
                    subtask_variant="b_coarse", profile="full-delta", action_dim=58,
                    action_horizon=50, action_hz=30, stage_contract="tomato-coarse-four-prompts-en-v4",
                    image_keys=list(FOUR_IMAGE_KEYS), visual_tokens=1024,
                    offline_training_roi_uses_future=True, online_roi_required=True,
                    evidence="verified checkpoint-owned statistics and Orbax fingerprints")
    if (stats_sha, manifest_sha, metadata_sha) != (STATS_SHA256, MANIFEST_SHA256, METADATA_SHA256):
        return None
    return dict(
        name="normal93 B Coarse FullFT Delta · 7999",
        task_name="放回盘中调整 · 七阶段", input_name="三路 RGB（全图）",
        dataset_episodes=93, dataset_version="整串采摘 B", checkpoint_step=7999,
        training_date="2026-11-09",
        subtask_variant="b_coarse",
        profile="full-delta",
        action_dim=58,
        action_horizon=50,
        action_hz=30,
        stage_contract="tomato-coarse-four-prompts-en-v4",
        evidence="verified Orbax manifest, metadata and checkpoint-owned normalization fingerprints",
    )
