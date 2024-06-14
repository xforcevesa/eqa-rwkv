import dataclasses
from PIL import Image
import os
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_HEAD_SIZE_A"] = "64"
os.environ["RWKV_CTXLEN"] = "256"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from src.model import VisualRWKV
from src.rwkv_tokenizer import TRIE_TOKENIZER
from transformers import CLIPImageProcessor
import torch
from pathlib import Path
from src.utils import gpt4v_crop


@dataclasses.dataclass
class EQARWKVConfig:
    """
    Configuration for EQA RWKV.
    """
    load_model: str = ""
    vocab_size: int = 65536
    ctx_len: int = 256

    n_layer: int = 24
    n_embd: int = 2048
    dim_att: int = 0
    dim_ffn: int = 0
    pre_ffn: int = 0
    head_size_a: int = 64
    head_size_divisor: int = 8
    dropout: float = 0.0

    vision_tower_name: str = "openai/clip-vit-base-patch32"
    grid_size: int = 8
    detail: str = "low"
    grad_cp: int = 0

    # arguments for evaluation
    model_path: str = None
    image_folder: str = None
    question_file: str = None
    output_file: str = None
    temperature: float = 0.2
    top_p: float = None
    max_new_tokens: int = 128
    num_chunks: int = 1
    chunk_idx: int = 0
    device: str = "cuda"
    dataset_name: str = "default"
    image_position: str = 'first'

    def __dict__(self):
        return dataclasses.asdict(self)
    
    def __init__(self, **kwargs):
        super(EQARWKVConfig, self).__init__(**kwargs)
        if self.dim_att <= 0:
            self.dim_att = self.n_embd
        if self.dim_ffn <= 0:
            self.dim_ffn = int((self.n_embd * 3.5) // 32 * 32) 


def preprocess_image(
        image_dir: str, 
        image_processor: CLIPImageProcessor, 
        detail: str = 'low'):
    '''
    Preprocesses the images in the specified directory.
    '''
    image_tensors = []
    try:
        images = os.listdir(image_dir)
        images = [os.path.join(image_dir, image) for image in images]
        images = sorted(images)
        for image in images:
            try:
                image = Image.open(image)
                if detail == 'high':
                    # Bug await for fix here
                    image = [image] + gpt4v_crop(image)
                    image_tensor = image_processor(images=image, return_tensors='pt')['pixel_values']
                else:
                    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
                    image_tensor = image_tensor.unsqueeze(0)
                    image_tensors.append(image_tensor)
            except Exception as e:
                print(f"Error in preprocessing image {image}: {e}")
                continue
    except:
        if detail == 'high':
            crop_size = image_processor.crop_size
            image_tensor = torch.zeros(7, 3, crop_size['height'], crop_size['width'])
            image_tensors.append(image_tensor)
        else:
            crop_size = image_processor.crop_size
            image_tensor = torch.zeros(1, 3, crop_size['height'], crop_size['width'])
            image_tensors.append(image_tensor)
    image_tensor = torch.cat(image_tensors, dim=1)
    return image_tensor


def fetch_model(config: EQARWKVConfig) -> tuple:
    '''
    Loads the model and tokenizer.
    '''
    model_path = Path(config.model_path)
    model_name = model_path.parent.name
    # Model
    model = VisualRWKV(config)
    msg = model.load_state_dict(torch.load(model_path), strict=False)
    print("msg of loading model: ", msg)
    model = model.bfloat16().to(config.device)
    tokenizer = TRIE_TOKENIZER("src/rwkv_vocab_v20230424.txt")
    image_processor = CLIPImageProcessor.from_pretrained(config.vision_tower_name)
    return model, model_name, tokenizer, image_processor


def test_preprocessed_images():
    image_dir = "dummy_data/images/textvqa/train_images/"
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image_tensor = preprocess_image(image_dir, image_processor)
    print(image_tensor.shape) # (1, N, 3, 224, 224)


def test_fetch_model():
    config = EQARWKVConfig(
        model_path="../../../../visualrwkv-6/VisualRWKV-v060-1B6-v1.0-20240612.pth",
        
    )
    model, model_name, tokenizer, image_processor = fetch_model(config)
    print(type(model))


if __name__ == '__main__':
    # test_preprocessed_images()
    test_fetch_model()

