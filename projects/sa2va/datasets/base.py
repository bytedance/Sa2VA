"""
Base classes for Sa2VA datasets with common functionality.
"""
from functools import partial
from typing import Literal, Optional, Dict, List, Any
import torch
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from torch.utils.data import Dataset
from mmengine import print_log
from xtuner.registry import BUILDER
from .data_utils import dynamic_preprocess, template_map_fn, tokenize_conversation


class Sa2VADatasetMixin:
    """
    Mixin class containing common functionality for Sa2VA datasets.
    This includes architecture configuration, image processing, and tokenization logic.
    """
    
    # Default constants
    DEFAULT_IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    DEFAULT_IMG_START_TOKEN = '<img>'
    DEFAULT_IMG_END_TOKEN = '</img>'
    
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    def _init_architecture_config(self, arch_type: Literal['intern_vl', 'qwen', 'llava'] = 'intern_vl'):
        """Initialize architecture-specific configurations."""
        self.arch_type = arch_type
        
        # Set default tokens
        self.IMG_CONTEXT_TOKEN = self.DEFAULT_IMG_CONTEXT_TOKEN
        self.IMG_START_TOKEN = self.DEFAULT_IMG_START_TOKEN
        self.IMG_END_TOKEN = self.DEFAULT_IMG_END_TOKEN
        
        # Architecture-specific overrides
        if self.arch_type == 'qwen':
            self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
            self.IMG_START_TOKEN = '<|vision_start|>'
            self.IMG_END_TOKEN = '<|vision_end|>'
        elif self.arch_type == 'llava':
            self.IMG_CONTEXT_TOKEN = '<image>'
            self.IMG_START_TOKEN = ''
            self.IMG_END_TOKEN = ''
    
    def _init_image_processing_config(self, 
                                    min_dynamic_patch: int = 1,
                                    max_dynamic_patch: int = 12,
                                    image_size: int = 448,
                                    use_thumbnail: bool = True,
                                    downsample_ratio: float = 0.5,
                                    patch_size: int = 14):
        """Initialize image processing configurations."""
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.use_thumbnail = use_thumbnail
        
        # Architecture-specific adjustments
        if self.arch_type == 'llava':
            self.downsample_ratio = 1
            self.image_size = 336
        else:
            self.downsample_ratio = downsample_ratio
            self.image_size = image_size
            
        # Calculate patch tokens
        if self.arch_type == 'qwen':
            self.patch_token = 1
            self.min_pixels_single = 512*28*28
            self.max_pixels_single = 2048*28*28

            self.min_pixels_multi = 128*28*28
            self.max_pixels_multi = 512*28*28
        else:
            self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))
    
    def _init_tokenizer(self, tokenizer_config, special_tokens: Optional[List[str]] = None):
        """Initialize tokenizer with special tokens."""
        self.tokenizer = BUILDER.build(tokenizer_config)
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)
    
    def _init_image_processor(self, preprocessor_config=None):
        """Initialize image processor/transformer."""
        if preprocessor_config is None:
            self.transformer = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
            ])
            self.preprocessor = None
        else:
            self.transformer = None
            self.preprocessor = BUILDER.build(preprocessor_config)
    
    def _init_extra_image_processor(self, extra_image_processor_config=None):
        """Initialize extra image processor for grounding."""
        if extra_image_processor_config is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor_config)
        else:
            self.extra_image_processor = None
    
    def _setup_system_prompt(self):
        """Setup system prompt (empty by default for all architectures)."""
        self._system = ''
    
    def _process_single_image(self, image: Image.Image, single_image_mode: bool = False) -> Dict[str, Any]:
        """
        Process a single image and return pixel values and number of tokens.
        
        Args:
            image: PIL Image
            single_image_mode: Whether to use single image mode
            
        Returns:
            Dictionary containing processed image data
        """
        result = {}
        
        # Process for grounding if needed
        if hasattr(self, 'extra_image_processor') and self.extra_image_processor is not None:
            g_image = np.array(image)
            g_image = self.extra_image_processor.apply_image(g_image)
            g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
            result['g_pixel_values'] = g_pixel_values
        
        # Process images
        if self.preprocessor is not None:
            if self.arch_type == 'qwen':
                images = [image]
                merge_length = self.preprocessor.image_processor.merge_size ** 2
                _data_dict = self.preprocessor.image_processor(
                    images=images, min_pixels=self.min_pixels_single, max_pixels=self.max_pixels_single
                )
                # _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                # _data_dict['image_grid_thw'] = torch.tensor(_data_dict['image_grid_thw'], dtype=torch.int)
                num_image_tokens = int(_data_dict['image_grid_thw'][0].prod()) // merge_length
            elif self.arch_type == 'llava':
                raise NotImplementedError("LLaVA preprocessor not implemented for single image mode")
                _data_dict = self.preprocessor(images, do_resize=True, size=(self.image_size, self.image_size))
                _data_dict['pixel_values'] = np.stack(_data_dict['pixel_values'], axis=0)
                _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                num_image_tokens = _data_dict['pixel_values'].shape[0] * self.patch_token
            else:
                raise NotImplementedError(f"Preprocessor not implemented for {self.arch_type}")
            result.update(_data_dict)
        else:
            assert self.transformer is not None, "Transformer must be defined if no preprocessor"
            # Prepare images for processing
            if single_image_mode:
                images = [image]
            else:
                images = dynamic_preprocess(image, self.min_dynamic_patch,
                                        self.max_dynamic_patch,
                                        self.image_size, self.use_thumbnail)

            pixel_values = [self.transformer(img) for img in images]
            pixel_values = torch.stack(pixel_values)
            result['pixel_values'] = pixel_values
            num_image_tokens = pixel_values.shape[0] * self.patch_token
        
        result['num_image_tokens'] = num_image_tokens
        return result
    
    def _process_multiple_images(self, images: List[Image.Image]) -> Dict[str, Any]:
        """
        Process multiple images (for video datasets) and return pixel values and number of tokens.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Dictionary containing processed image data
        """
        result = {}
        pixel_values = []
        extra_pixel_values = []
        
        # Process each image
        for image in images:
            image = image.convert('RGB')
            ori_width, ori_height = image.size
            
            # Process for grounding if needed
            if hasattr(self, 'extra_image_processor') and self.extra_image_processor is not None:
                g_image = np.array(image)
                g_image = self.extra_image_processor.apply_image(g_image)
                g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                extra_pixel_values.append(g_pixel_values)

            if self.preprocessor is not None:
                # Store images for batch processing
                pixel_values.append(image)
            else:
                # Apply transforms immediately
                transformed = self.transformer(image)
                pixel_values.append(transformed)

        # Process images based on preprocessor availability
        if self.preprocessor is not None:
            if self.arch_type == 'qwen':
                merge_length = self.preprocessor.image_processor.merge_size ** 2
                _data_dict = self.preprocessor.image_processor(
                    images=images, min_pixels=self.min_pixels_multi, max_pixels=self.max_pixels_multi
                )
                num_frame_tokens = int(_data_dict['image_grid_thw'][0].prod() // merge_length)
                num_frames = _data_dict['image_grid_thw'].shape[0]
                num_total_tokens = num_frame_tokens * num_frames
                result.update(_data_dict)
                result['num_frame_tokens'] = num_frame_tokens
                result['num_frames'] = num_frames
            elif self.arch_type == 'llava':
                raise NotImplementedError("LLaVA preprocessor not implemented for multiple image mode")
            else:
                raise NotImplementedError(f"Preprocessor not implemented for {self.arch_type}")
        else:
            pixel_values = torch.stack(pixel_values, dim=0)  # (n_f, 3, h, w)
            result['pixel_values'] = pixel_values
            num_total_tokens = len(images) * self.patch_token

        if extra_pixel_values:
            result['g_pixel_values'] = extra_pixel_values

        result['num_image_tokens'] = num_total_tokens
        return result
    
    def _create_token_string(self, num_tokens: int, num_frames: int = 1) -> str:
        """
        Create token string for images or videos.
        
        Args:
            num_tokens: Total number of tokens
            num_frames: Number of frames (1 for image, >1 for video)
            
        Returns:
            Token string with proper formatting
        """
        if num_frames == 1:
            # Single image case
            return f'{self.IMG_START_TOKEN}{self.IMG_CONTEXT_TOKEN * num_tokens}{self.IMG_END_TOKEN}'
        else:
            # Video case - create frame tokens
            if self.arch_type == 'qwen' and hasattr(self, 'patch_token') and self.patch_token == 1:
                # For qwen with patch_token=1, we use single tokens that will be expanded later
                frame_token_str = f'{self.IMG_START_TOKEN}{self.IMG_CONTEXT_TOKEN}{self.IMG_END_TOKEN}'
            else:
                # For other cases, use tokens per frame
                tokens_per_frame = num_tokens // num_frames
                frame_token_str = f'{self.IMG_START_TOKEN}{self.IMG_CONTEXT_TOKEN * tokens_per_frame}{self.IMG_END_TOKEN}'
            
            # Repeat for all frames with newlines
            frame_tokens = (frame_token_str + '\n') * num_frames
            return frame_tokens.strip()
    
    def _create_image_token_string(self, num_image_tokens: int) -> str:
        """Create image token string for given number of tokens (backward compatibility)."""
        return self._create_token_string(num_image_tokens, num_frames=1)
    
    def _process_conversations_for_encoding(self, conversations: List[Dict], image_token_str: Optional[str] = None, 
                                          is_video: bool = False) -> List[Dict]:
        """
        Process conversations to prepare for tokenization.
        
        Args:
            conversations: List of conversation messages
            image_token_str: Image token string to replace <image> placeholders
            is_video: Whether this is video data (affects token placement)
            
        Returns:
            List of processed conversation turns
        """
        # Handle different input formats
        if conversations and 'input' in conversations[0] and 'output' in conversations[0]:
            # Already in the correct format (from video datasets)
            return conversations
            
        input_text = ''
        out_conversation = []
        
        # Skip leading GPT messages
        while conversations and conversations[0]['from'] == 'gpt':
            conversations = conversations[1:]
        
        conv_idx = 0
        for msg in conversations:
            if msg['from'] == 'human':
                value = msg['value']
                
                # Handle image token replacement
                if '<image>' in value:
                    if image_token_str is None:
                        value = value.replace('<image>', '')
                    else:
                        assert conv_idx == 0, f"Expected conversation index to be 0, but got {conv_idx} / {value}"
                        if is_video:
                            # For video, add tokens at the beginning
                            value = value.replace('<image>', '')
                            if conv_idx == 0:
                                value = image_token_str + value
                        else:
                            # For image, replace <image> placeholder
                            value = value.replace('<image>', image_token_str)
                        value = value.strip()
                
                input_text += value
            elif msg['from'] == 'gpt':
                out_conversation.append({
                    'input': input_text,
                    'output': msg['value'].strip()
                })
                input_text = ''
            else:
                raise NotImplementedError(f"Unknown message role: {msg['from']}")
            
            conv_idx += 1
        
        return out_conversation
    
    def get_inputid_labels(self, conversations: List[Dict]) -> Dict[str, List]:
        """
        Convert conversations to input_ids and labels for training.
        Uses video_lisa_encode_fn logic with template_map_fn support.
        
        Args:
            conversations: List of conversation messages (from/value or input/output format)
            image_token_str: Image token string to replace <image> placeholders
            
        Returns:
            Dictionary with 'input_ids' and 'labels' keys
        """
        # Prepare data dict for template_map_fn
        data_dict = {'conversation': conversations}
        result = self.template_map_fn(data_dict)
        data_dict.update(result)
        result = tokenize_conversation(data_dict, tokenizer=self.tokenizer, max_length=self.max_length)
        return result
    
    def _expand_video_tokens(self, conversations: List[Dict], num_frame_tokens: int, num_total_tokens: int) -> List[Dict]:
        """
        Expand video tokens for architectures that need post-processing (like qwen).
        
        Args:
            conversations: Processed conversations
            num_frame_tokens: Tokens per frame
            num_total_tokens: Total video tokens
            
        Returns:
            Updated conversations with expanded tokens
        """
        if conversations and self.arch_type == 'qwen' and hasattr(self, 'patch_token') and self.patch_token == 1:
            # For qwen, expand the single tokens to frame tokens
            input_str = conversations[0]['input']
            input_str = input_str.replace(self.IMG_CONTEXT_TOKEN, self.IMG_CONTEXT_TOKEN * num_frame_tokens)
            assert input_str.count(self.IMG_CONTEXT_TOKEN) == num_total_tokens, \
                f"Token count mismatch: expected {num_total_tokens}, got {input_str.count(self.IMG_CONTEXT_TOKEN)}"
            conversations[0]['input'] = input_str
        return conversations
    
    def _get_modality_length_default(self, length: int = 100) -> int:
        """Get default modality length."""
        return length

    def _read_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Centralized image reading method to avoid duplicate code.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object or None if reading fails
        """
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            print(f'Error reading image {image_path}: {e}', flush=True)
            print_log(f'Error reading image {image_path}: {e}', logger='current')
            return None
    
    def _check_image_exists(self, image_path: str) -> bool:
        """
        Check if image file exists and can be opened without actually loading it.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if image exists and can be opened, False otherwise
        """
        try:
            with Image.open(image_path) as img:
                # Just check if we can open the image, don't load into memory
                img.verify()
            return True
        except Exception:
            return False


class Sa2VABaseDataset(Dataset, Sa2VADatasetMixin):
    """
    Base dataset class for Sa2VA datasets.
    Provides common initialization and utility methods.
    """
    
    def __init__(self,
                 tokenizer,
                 prompt_template,
                 max_length: int = 2048,
                 special_tokens: Optional[List[str]] = None,
                 arch_type: Literal['intern_vl', 'qwen', 'llava'] = 'intern_vl',
                 preprocessor=None,
                 extra_image_processor=None,
                 min_dynamic_patch: int = 1,
                 max_dynamic_patch: int = 12,
                 image_size: int = 448,
                 use_thumbnail: bool = True,
                 downsample_ratio: float = 0.5,
                 patch_size: int = 14,
                 max_refetch: int = 1000,
                 repeats: float = 1.0,
                 name: str = "Sa2VABaseDataset",
                 ):
        """
        Initialize base dataset with common configurations.
        
        Args:
            tokenizer: Tokenizer configuration
            prompt_template: Template for formatting prompts
            max_length: Maximum sequence length
            special_tokens: List of special tokens to add
            arch_type: Architecture type ('intern_vl', 'qwen', 'llava')
            preprocessor: Image preprocessor configuration
            extra_image_processor: Extra image processor for grounding
            min_dynamic_patch: Minimum dynamic patches
            max_dynamic_patch: Maximum dynamic patches  
            image_size: Image size
            use_thumbnail: Whether to use thumbnail
            downsample_ratio: Downsample ratio
            patch_size: Patch size
            max_refetch: Maximum refetch attempts
            repeats: Number of times to repeat the dataset (can be fractional, e.g., 0.2)
            template_map_fn: Template mapping function for xtuner format conversion
        """
        super().__init__()
        
        # Store core configurations
        self.template = prompt_template
        self.max_length = max_length
        self._max_refetch = max_refetch
        self.repeats = repeats
        
        # Pre-compute index mapping for equal distribution when using fractional repeats
        self._index_mapping = None
        
        # Template mapping function for format conversion
        self.template_map_fn = partial(template_map_fn, template=self.template)

        # Set name, it is for logging purposes
        self.name = name

        # Initialize architecture and processing configs
        self._init_architecture_config(arch_type)
        self._init_image_processing_config(min_dynamic_patch, max_dynamic_patch, 
                                         image_size, use_thumbnail, downsample_ratio, patch_size)
        
        # Initialize processors
        self._init_tokenizer(tokenizer, special_tokens)
        self._init_image_processor(preprocessor)
        self._init_extra_image_processor(extra_image_processor)
        self._setup_system_prompt()
    
    def __len__(self):
        """Get total length considering repeats."""
        return int(self.real_len() * self.repeats)
    
    def real_len(self):
        """Get the actual length without repeats. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement real_len")
    
    def _get_index_mapping(self):
        """Create or return cached index mapping for shuffled samples with fractional repeats."""
        if self._index_mapping is None:
            real_length = self.real_len()
            total_length = int(real_length * self.repeats)
            
            # Create indices based on repeats
            if self.repeats >= 1.0:
                # For repeats >= 1, repeat indices and take the first total_length
                repeated_indice = np.tile(np.arange(real_length), int(np.ceil(self.repeats)))
                indices = np.random.permutation(repeated_indice)[:total_length]
            else:
                # For repeats < 1, randomly sample total_length indices from all available indices
                indices = np.random.choice(real_length, size=total_length, replace=False)
            
            self._index_mapping = indices
                
        return self._index_mapping
    
    def shuffle_indices(self):
        """Create a new shuffled index mapping. Call this after each epoch for different sample order."""
        # Reset the index mapping to None so it gets recreated with new shuffle
        self._index_mapping = None
        # Force creation of new index mapping
        self._get_index_mapping()
    
    def __getitem__(self, index):
        """Unified __getitem__ implementation with refetch logic."""
        # Handle repeats using index mapping for equal distribution
        index_mapping = self._get_index_mapping()
        mapped_index = index_mapping[index]
        
        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(mapped_index)
            # Broken images may cause the returned data to be None
            if data is None:
                mapped_index = self._rand_another_index()
                continue
            return data
        
        # If we reach here, all retries failed
        raise RuntimeError(f"Failed to get valid data after {self._max_refetch + 1} attempts")
    
    def _rand_another_index(self) -> int:
        """Get random index for refetching."""
        return np.random.randint(0, self.real_len())
    
    def prepare_data(self, index):
        """Prepare data for a given index. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement prepare_data")
    
    @property
    def modality_length(self):
        """Get modality length for all items."""
        return [self._get_modality_length_default() for _ in range(len(self))]
