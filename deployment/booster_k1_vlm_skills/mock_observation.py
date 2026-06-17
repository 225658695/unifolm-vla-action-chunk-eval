from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from schemas import Observation


def build_mock_image(out_path: str | Path, instruction: str) -> str:
    """Create a tiny synthetic camera frame for CPU-only dry-run tests."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (640, 360), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((245, 130, 335, 230), fill=(220, 40, 40), outline=(120, 0, 0), width=3)
    draw.rectangle((410, 160, 560, 245), outline=(40, 40, 40), width=4)
    draw.text((20, 20), "Mock K1 camera observation", fill=(0, 0, 0))
    draw.text((20, 45), instruction[:88], fill=(0, 0, 0))
    draw.text((252, 235), "target", fill=(0, 0, 0))
    draw.text((430, 250), "place area", fill=(0, 0, 0))
    image.save(out_path)
    return str(out_path)


def make_observation(instruction: str, image_path: str | None, step_idx: int) -> Observation:
    state = {
        "head_yaw": 0.0,
        "head_pitch": 0.0,
        "left_arm_joint_pos": [0.0] * 7,
        "right_arm_joint_pos": [0.0] * 7,
        "gripper": "unknown",
    }
    return Observation(
        image_path=image_path,
        instruction=instruction,
        state=state,
        step_idx=step_idx,
    )

