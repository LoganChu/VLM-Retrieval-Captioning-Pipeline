import torch
from dataclasses import dataclass
from typing import Literal

@dataclass
class DataConfig:
    """Dataset configuration"""
    dataset_name: str = "coco"  # or "flickr30k"
    data_dir: str = "./data"
    num_samples: int = 1000
    num_ood: int = 100
    train_size: int = 800
    val_size: int = 100
    test_size: int = 100  # in-distribution test
    # OOD will be mixed into test
    image_size: int = 224
    cache_dir: str = "./cache"
    
@dataclass
class HardwareConfig:
    """Hardware-specific settings"""
    device_type: Literal["rtx5000_ada", "a5000", "rtx3070"] = "rtx5000_ada"
    
    def get_clip_batch_size(self) -> int:
        if self.device_type == "rtx5000_ada":
            return 96  # 32GB VRAM - can handle larger batches
        elif self.device_type == "a5000":
            return 64
        else:
            return 16
    
    def get_image_size(self) -> int:
        if self.device_type == "rtx5000_ada":
            return 336  # Ada architecture supports larger images efficiently
        elif self.device_type == "a5000":
            return 224
        else:
            return 192
    
    def get_linear_probe_bs(self) -> int:
        if self.device_type == "rtx5000_ada":
            return 128  # Double the A5000
        elif self.device_type == "a5000":
            return 64
        else:
            return 16
    
    def get_captioner_bs(self) -> int:
        if self.device_type == "rtx5000_ada":
            return 48  # 50% more than A5000
        elif self.device_type == "a5000":
            return 32
        else:
            return 10
    
    def use_gradient_checkpointing(self) -> bool:
        return self.device_type == "rtx3070"  # Not needed for RTX 5000 Ada
    
    def get_num_workers(self) -> int:
        """Data loader workers - Ada benefits from more parallelism"""
        return 8 if self.device_type == "rtx5000_ada" else 4

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    clip_model: str = "openai/clip-vit-base-patch32"
    clip_embed_dim: int = 512
    
    # Linear probe
    probe_lr: float = 1e-3
    probe_epochs: int = 10
    
    # Captioner
    captioner_type: Literal["blip", "gpt2_decoder"] = "gpt2_decoder"
    decoder_layers: int = 6  # Increased for RTX 5000 Ada (was 4 for A5000, 2 for RTX 3070)
    decoder_hidden: int = 768  # Increased for RTX 5000 Ada (was 512 for A5000)
    decoder_heads: int = 12  # Increased for RTX 5000 Ada (was 8)
    max_caption_length: int = 32
    
    # LoRA
    use_lora: bool = False
    lora_r: int = 16  # Increased rank for Ada (was 8)
    lora_alpha: int = 32  # Increased for Ada (was 16)
    lora_dropout: float = 0.1
    
    # Training
    captioner_lr: float = 2e-4
    captioner_epochs: int = 5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    use_amp: bool = True
    gradient_clip: float = 1.0
    
    # RTX 5000 Ada optimizations
    use_flash_attention: bool = True  # Ada supports flash attention 2
    use_tf32: bool = True  # Enable TF32 for Ada architecture

@dataclass
class ExperimentConfig:
    """Experiment settings"""
    num_seeds: int = 3
    seeds: list = None
    output_dir: str = "./outputs"
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    profile: bool = True
    
    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 123, 456]

class Config:
    """Main configuration class"""
    def __init__(self, device_type: str = "rtx5000_ada"):
        self.data = DataConfig()
        self.hardware = HardwareConfig(device_type=device_type)
        self.model = ModelConfig()
        self.experiment = ExperimentConfig()
        
        # Adjust model config for hardware
        if device_type == "rtx5000_ada":
            # RTX 5000 Ada (32GB) - Most powerful configuration
            self.data.image_size = 336
            self.model.decoder_layers = 6
            self.model.decoder_hidden = 768
            self.model.decoder_heads = 12
            self.model.lora_r = 16
            self.model.lora_alpha = 32
            self.model.use_flash_attention = True
            self.model.use_tf32 = True
        elif device_type == "a5000":
            # A5000 (24GB) - Balanced configuration
            self.data.image_size = 224
            self.model.decoder_layers = 4
            self.model.decoder_hidden = 512
            self.model.decoder_heads = 8
            self.model.lora_r = 8
            self.model.lora_alpha = 16
        elif device_type == "rtx3070":
            # RTX 3070 (8GB) - Memory-constrained configuration
            self.data.image_size = 192
            self.model.decoder_layers = 2
            self.model.decoder_hidden = 384
            self.model.decoder_heads = 6
            self.model.lora_r = 4
            self.model.lora_alpha = 8
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        
        # Enable TF32 for RTX 5000 Ada (significant speedup on Ada architecture)
        if device_type == "rtx5000_ada" and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✓ TF32 enabled for RTX 5000 Ada - expect ~2x speedup on matmul operations")
    
    def __repr__(self):
        return (f"Config(device_type={self.hardware.device_type}, "
                f"clip_bs={self.hardware.get_clip_batch_size()}, "
                f"captioner_bs={self.hardware.get_captioner_bs()}, "
                f"image_size={self.data.image_size})")
    
    def print_hardware_info(self):
        """Print detailed hardware configuration"""
        print("\n" + "="*60)
        print("HARDWARE CONFIGURATION")
        print("="*60)
        print(f"Device Type: {self.hardware.device_type}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
        print(f"\nBatch Sizes:")
        print(f"  CLIP Encoding: {self.hardware.get_clip_batch_size()}")
        print(f"  Linear Probe: {self.hardware.get_linear_probe_bs()}")
        print(f"  Captioner: {self.hardware.get_captioner_bs()}")
        print(f"\nImage Size: {self.data.image_size}x{self.data.image_size}")
        print(f"Decoder: {self.model.decoder_layers} layers, {self.model.decoder_hidden} hidden")
        print(f"TF32 Enabled: {self.model.use_tf32}")
        print(f"Flash Attention: {self.model.use_flash_attention}")
        print("="*60)
