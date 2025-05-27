from .dual_ur_bottle import DualURBottle
from .dual_franka_bottle import DualFrankaBottle
from .dual_franka_reorientation import DualFrankaReorientation

# Mappings from strings to environments
isaacgym_task_map = {
    "DualURBottle": DualURBottle,
    "DualFrankaBottle": DualFrankaBottle,
    "DualFrankaReorientation": DualFrankaReorientation,
}
