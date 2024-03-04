from miniworld.params import DomainParams


# Default simulation parameters
DEFAULT_PARAMS = DomainParams()
DEFAULT_PARAMS.set("sky_color", [0.25, 0.82, 1], [0.1, 0.1, 0.1], [1.0, 1.0, 1.0])
DEFAULT_PARAMS.set("light_pos", [0, 2.5, 0], [-40, 2.5, -40], [40, 5, 40])
DEFAULT_PARAMS.set("light_color", [0.7, 0.7, 0.7], [0.45, 0.45, 0.45], [0.8, 0.8, 0.8])
DEFAULT_PARAMS.set(
    "light_ambient", [0.45, 0.45, 0.45], [0.35, 0.35, 0.35], [0.55, 0.55, 0.55]
)
DEFAULT_PARAMS.set("obj_color_bias", [0, 0, 0], [-0.2, -0.2, -0.2], [0.2, 0.2, 0.2])
DEFAULT_PARAMS.set("forward_step", 0.15, 0.15, 0.15)
DEFAULT_PARAMS.set("forward_drift", 0, -0.05, 0.05)
DEFAULT_PARAMS.set("turn_step", 5, 5, 5)
DEFAULT_PARAMS.set("bot_radius", 0.4, 0.38, 0.42)
DEFAULT_PARAMS.set("cam_pitch", 0, -5, 5)
DEFAULT_PARAMS.set("cam_fov_y", 60, 55, 65)
DEFAULT_PARAMS.set("cam_height", 1.5, 1.45, 1.55)
DEFAULT_PARAMS.set("cam_fwd_disp", 0, -0.05, 0.10)