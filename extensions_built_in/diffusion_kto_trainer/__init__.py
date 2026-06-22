from toolkit.extension import Extension


class DiffusionKTOTrainerExtension(Extension):
    uid = "diffusion_kto_trainer"
    name = "Diffusion KTO Trainer"
    process_name = "diffusion_kto_trainer"

    @classmethod
    def get_process(cls):
        from .DiffusionKTOTrainer import DiffusionKTOTrainer

        return DiffusionKTOTrainer


AI_TOOLKIT_EXTENSIONS = [
    DiffusionKTOTrainerExtension,
]
