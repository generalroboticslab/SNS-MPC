from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_path(*parts: str) -> str:
    return str(REPO_ROOT.joinpath(*parts))


def resolve_repo_path(path: str | None) -> str | None:
    if path is None:
        return None

    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(REPO_ROOT / candidate)


GO2_ASSETS_DIR = repo_path("Mj_models", "Go2_dog", "assets")
SCENE_TORQUE_XML = repo_path("Mj_models", "Go2_dog", "scene_torque.xml")
GO2_TORQUE_XML = repo_path("Mj_models", "Go2_dog", "go2_torque.xml")

ACTUATOR_NETWORK = repo_path("actuator_logs", "actuator_network")
ACTUATOR_NETWORK_200HZ = repo_path("actuator_logs", "actuator_network_200hz")
ACTUATOR_DATA_200HZ = repo_path("actuator_logs", "data_200hz.npz")

OPEN_SANS_REGULAR_TTF = repo_path("OpenSans-Regular.ttf")
HELVETICA_FONT_PATHS = (
    repo_path("helvetica-255", "Helvetica.ttf"),
    repo_path("helvetica-255", "Helvetica-Bold.ttf"),
    repo_path("helvetica-255", "Helvetica-Oblique.ttf"),
)

EXAMPLE_CHECKPOINT_DIR = repo_path(
    "Trained_models",
    "Primary",
    "Ckpt",
    "Lipschitz_2025-07-14_15-48-44",
)
EXAMPLE_DYNAMICS_PATH = repo_path(
    "Trained_models",
    "Primary",
    "Ckpt",
    "Lipschitz_2025-07-14_15-48-44",
    "ckpt_Lipschitz_MPPI_Policy_2025-07-14_15-49-01_dynamics_params.npz",
)
EXAMPLE_OBSERVER_PATH = repo_path(
    "Trained_models",
    "Primary",
    "Ckpt",
    "Lipschitz_2025-07-14_15-48-44",
    "ckpt_Lipschitz_MPPI_Policy_2025-07-14_15-49-01_observer_params.npz",
)
EXAMPLE_STATES_PATH = repo_path(
    "Trained_models",
    "Primary",
    "Ckpt",
    "Lipschitz_2025-07-14_15-48-44",
    "ckpt_Lipschitz_MPPI_Policy_2025-07-14_15-49-01_states.dill",
)
EXAMPLE_ACTIONS_PATH = repo_path(
    "Trained_models",
    "Primary",
    "Ckpt",
    "Lipschitz_2025-07-14_15-48-44",
    "ckpt_Lipschitz_MPPI_Policy_2025-07-14_15-49-01_actions.dill",
)
EXAMPLE_OBSERVATIONS_PATH = repo_path(
    "Trained_models",
    "Primary",
    "Ckpt",
    "Lipschitz_2025-07-14_15-48-44",
    "ckpt_Lipschitz_MPPI_Policy_2025-07-14_15-49-01_observations.dill",
)
