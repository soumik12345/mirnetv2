import ml_collections


def get_dataloader_configs() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.image_size = 256
    config.bit_depth = 8
    config.val_split = 0.2
    config.local_batch_size = 4
    config.visualize_on_wandb = False
    config.dataset_artifact_address = "ml-colabs/mirnet-v2/lol-dataset:v0"

    return config


def get_model_configs() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.channels = 80
    config.num_mrb_blocks = 2
    config.channel_factor = 1.5
    config.add_residual_connection = True

    return config


def get_training_configs() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.global_batch_size = 8
    config.initial_learning_rate = 2e-4
    config.minimum_learning_rate = 1e-6
    config.decay_rate_1 = 0.9
    config.decay_rate_2 = 0.999
    config.weight_decay = 1e-4
    config.charbonnier_epsilon = 1e-3
    config.psnr_max_val = 1.0
    config.ssim_max_val = 1.0
    config.save_best_checkpoint_only = False
    config.num_evaluation_batches = 2
    config.epochs = 100

    return config


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.seed = 42
    config.data_loader_configs = get_dataloader_configs()
    config.model_configs = get_model_configs()
    config.training_configs = get_training_configs()

    return config
