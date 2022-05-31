from typing import List
import numpy as np
import torch
import clip
from minDALLE.dalle.models import Dalle
from minDALLE.dalle.utils.utils import set_seed, clip_score



class MyDALLE(Dalle):
    def __init__(self):
        # super(Dalle, self).__init__()
        super(MyDALLE, self).__init__()
        self.model = Dalle.from_pretrained('minDALL-E/1.3B')


def get_model() -> MyDALLE:
    set_seed(42)
    """Model을 가져옵니다"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyDALLE.to(device)
    # model = Dalle.from_pretrained('minDALL-E/1.3B').to(device)
    return model


def txt2img(model: MyDALLE, text: str) -> List[int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Sampling
    images = model.sampling(prompt=text,
                            top_k=256,
                            top_p=None,
                            softmax_temperature=1.0,
                            num_candidates=3,
                            device=device).cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))

    # CLIP Re-ranking
    model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
    model_clip.to(device=device)
    rank = clip_score(prompt=text,
                    images=images,
                    model_clip=model_clip,
                    preprocess_clip=preprocess_clip,
                    device=device)

    # Save images
    images = images[rank]
    # img = []
    # for image in images:
    #     img.append(Image.fromarray((image*255).astype(np.uint8)))
    return images