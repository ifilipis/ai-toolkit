from toolkit.extension import Extension


class DiffusionDPOTrainerExtension(Extension):
    uid = "diffusion_dpo_trainer"
    name = "Diffusion DPO Trainer"

    @classmethod
    def get_process(cls):
        from .DiffusionDPOTrainer import DiffusionDPOTrainer

        return DiffusionDPOTrainer


AI_TOOLKIT_EXTENSIONS = [
    DiffusionDPOTrainerExtension,
]
