from .dual_ur_bottle import DualURBottle
from .dual_franka_bottle import DualFrankaBottle

# Mappings from strings to environments
isaacgym_task_map = {
    "DualURBottle": DualURBottle,
    "DualFrankaBottle": DualFrankaBottle,
}
