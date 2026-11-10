"""Content identity verified against the completed 2026-11-09 B-coarse run.

Names and paths are not evidence. These fingerprints bind the Orbax manifest,
parameter metadata and checkpoint-owned normalization, independent of location.
Full parameter bytes/shapes are validated during Orbax restoration on GPU.
"""

STATS_SHA256 = "3f82a27ba53cb6a0553886d453c8e05ccb466354d131ee9c956cd0a12637350a"
MANIFEST_SHA256 = "55d037b0002ca3a76112ae848944663c3b602eac8c9a813847f67445050bdf60"
METADATA_SHA256 = "78df4cd15c8e0c09c250299c846da7504ec5da44b1463df3757db647beea4df0"


def identify(stats_sha, manifest_sha, metadata_sha):
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
