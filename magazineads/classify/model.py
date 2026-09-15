"""CLIP-based page classifier model (inference only).

Architecture: CLIP ViT-L/14-336 vision encoder with a linear classification
head (3 classes: not-ad, ad, mixed).

Usage:
    from magazineads.classify.model import CLIPImageClassifier

    model = CLIPImageClassifier(num_labels=3)
    model.load_state_dict(torch.load("checkpoint/model.safetensors"))
    model.eval()
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPConfig
from transformers.modeling_outputs import SequenceClassifierOutput


class CLIPImageClassifier(nn.Module):
    """CLIP vision encoder + linear classification head.

    Parameters
    ----------
    num_labels : number of output classes (default 3: not-ad, ad, mixed).
    model_name : HuggingFace model ID for the CLIP backbone.
    load_weights : if True, load pretrained CLIP weights; otherwise random init.
    """

    def __init__(
        self,
        num_labels: int = 3,
        model_name: str = "openai/clip-vit-large-patch14-336",
        load_weights: bool = True,
    ):
        super().__init__()

        if load_weights:
            clip_model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
            self.vision_encoder = clip_model.vision_model
        else:
            config = CLIPConfig.from_pretrained(model_name)
            self.vision_encoder = CLIPModel(config).vision_model

        hidden_size = self.vision_encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pixel_values, labels=None):
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        pooled_output = vision_outputs.pooler_output
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
