import logging
from pathlib import PurePath
from typing import Optional, Sequence

import yaml

import torch
from torch import Tensor, nn

log = logging.getLogger(__name__)


class InvalidModelError(RuntimeError):
    """Exception raised for any model-related error (creation, loading)"""


_WEIGHTS_URL = {
    'parseq-tiny': 'https://github.com/baudm/parseq/releases/download/v1.0.0/parseq_tiny-e7a21b54.pt',
    'parseq-patch16-224': 'https://github.com/baudm/parseq/releases/download/v1.0.0/parseq_small_patch16_224-fcf06f5a.pt',
    'parseq': 'https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt',
    'abinet': 'https://github.com/baudm/parseq/releases/download/v1.0.0/abinet-1d1e373e.pt',
    'trba': 'https://github.com/baudm/parseq/releases/download/v1.0.0/trba-cfaed284.pt',
    'vitstr': 'https://github.com/baudm/parseq/releases/download/v1.0.0/vitstr-26d0fcf4.pt',
    'crnn': 'https://github.com/baudm/parseq/releases/download/v1.0.0/crnn-679d0e31.pt',
}

# Official weights hosted on the Hugging Face Hub (https://huggingface.co/baudm)
_HF_REPO = {
    'parseq-tiny': 'baudm/parseq-tiny',
    'parseq-patch16-224': 'baudm/parseq-small-patch16-224',
    'parseq': 'baudm/parseq-small',
    'abinet': 'baudm/abinet-lv',
    'trba': 'baudm/trba',
    'vitstr': 'baudm/vitstr-small',
    'crnn': 'baudm/crnn',
}
_HF_FILENAME = 'pytorch_model.bin'

# Hyperparameters the official weights were trained with. These take precedence over configs/main.yaml
# (which may have been modified for custom training) when creating a model with pretrained weights.
PRETRAINED_CONFIG = {
    'max_label_length': 25,
}
_ROOT = PurePath(__file__).parents[2]


def _get_pretrained_charset() -> str:
    with open(_ROOT / 'configs/charset/94_full.yaml', 'r') as f:
        return yaml.load(f, yaml.Loader)['model']['charset_train']


def _get_config(experiment: str, **kwargs):
    """Emulates hydra config resolution"""
    root = _ROOT
    with open(root / 'configs/main.yaml', 'r') as f:
        config = yaml.load(f, yaml.Loader)['model']
    with open(root / 'configs/charset/94_full.yaml', 'r') as f:
        config.update(yaml.load(f, yaml.Loader)['model'])
    with open(root / f'configs/experiment/{experiment}.yaml', 'r') as f:
        exp = yaml.load(f, yaml.Loader)
    # Apply base model config
    model = exp['defaults'][0]['override /model']
    with open(root / f'configs/model/{model}.yaml', 'r') as f:
        config.update(yaml.load(f, yaml.Loader))
    # Apply experiment config
    if 'model' in exp:
        config.update(exp['model'])
    config.update(kwargs)
    # Workaround for now: manually cast the lr to the correct type.
    config['lr'] = float(config['lr'])
    return config


def _get_model_class(key):
    if 'abinet' in key:
        from .abinet.system import ABINet as ModelClass
    elif 'crnn' in key:
        from .crnn.system import CRNN as ModelClass
    elif 'parseq' in key:
        from .parseq.system import PARSeq as ModelClass
    elif 'trba' in key:
        from .trba.system import TRBA as ModelClass
    elif 'trbc' in key:
        from .trba.system import TRBC as ModelClass
    elif 'vitstr' in key:
        from .vitstr.system import ViTSTR as ModelClass
    else:
        raise InvalidModelError(f"Unable to find model class for '{key}'")
    return ModelClass


def _load_hf_weights(repo_id: str) -> dict:
    from huggingface_hub import hf_hub_download

    # The official weights are public. Avoid sending a (possibly stale) token.
    path = hf_hub_download(repo_id, _HF_FILENAME, token=False)
    return torch.load(path, map_location='cpu', weights_only=True)


def get_pretrained_weights(experiment: str) -> dict:
    """Get the official pretrained weights (state_dict) for `experiment`.

    The weights are downloaded from the Hugging Face Hub, with the GitHub release as fallback.
    `experiment` can also be a Hugging Face repo ID (e.g. 'baudm/parseq-tiny').
    """
    if '/' in experiment:
        return _load_hf_weights(experiment)
    if experiment not in _WEIGHTS_URL:
        raise InvalidModelError(f"No pretrained weights found for '{experiment}'")
    try:
        return _load_hf_weights(_HF_REPO[experiment])
    except Exception as e:  # e.g. offline, or HF Hub unavailable
        log.warning(f'Failed to download weights from the Hugging Face Hub ({e}). Falling back to GitHub release.')
    return torch.hub.load_state_dict_from_url(url=_WEIGHTS_URL[experiment], map_location='cpu', check_hash=True)


def _remap_token_weights(src: Tensor, src_itos: Sequence[str], dst: Tensor, dst_itos: Sequence[str]) -> int:
    """Copy rows of `src` to `dst` (in-place) for tokens which exist in both vocabularies. Returns number of rows."""
    src_stoi = {s: i for i, s in enumerate(src_itos)}
    n = 0
    for i, s in enumerate(dst_itos):
        j = src_stoi.get(s)
        if j is not None:
            dst[i] = src[j]
            n += 1
    return n


def adapt_parseq_state_dict(state_dict: dict, model: nn.Module, src_charset: str, dst_charset: str) -> dict:
    """Adapt PARSeq weights to a model with a different charset and/or max_label_length (e.g. for fine-tuning).

    Token embeddings and output head rows are copied for characters (and special tokens) present in both charsets.
    Positional queries are copied up to the shorter of the two lengths. Everything else is left as initialized.
    """
    from strhub.data.utils import Tokenizer

    src_itos = Tokenizer(src_charset)._itos
    dst_itos = Tokenizer(dst_charset)._itos
    target = model.state_dict()
    adapted = dict(state_dict)
    with torch.no_grad():
        # text_embed: all tokens. head: all tokens except [B] and [P] (last two)
        for key, src_tok, dst_tok in [
            ('text_embed.embedding.weight', src_itos, dst_itos),
            ('head.weight', src_itos[:-2], dst_itos[:-2]),
            ('head.bias', src_itos[:-2], dst_itos[:-2]),
        ]:
            w = target[key].clone()
            n = _remap_token_weights(state_dict[key], src_tok, w, dst_tok)
            log.info(f'{key}: copied {n}/{len(dst_tok)} token weights')
            adapted[key] = w
        pos = target['pos_queries'].clone()
        n = min(pos.shape[1], state_dict['pos_queries'].shape[1])
        pos[:, :n] = state_dict['pos_queries'][:, :n]
        adapted['pos_queries'] = pos
    return adapted


def load_pretrained_weights(model: nn.Module, experiment: str, charset_train: Optional[str] = None) -> None:
    """Initialize `model` (a BaseSystem) with the official pretrained weights of `experiment`.

    For PARSeq, the weights are adapted if the model uses a different charset or max_label_length.
    """
    state_dict = get_pretrained_weights(experiment)
    if 'parseq' in experiment:
        m = model.model
        src_charset = _get_pretrained_charset()
        if charset_train is not None and (
            charset_train != src_charset or m.pos_queries.shape != state_dict['pos_queries'].shape
        ):
            state_dict = adapt_parseq_state_dict(state_dict, m, src_charset, charset_train)
    else:
        m = model
    m.load_state_dict(state_dict)


def create_model(experiment: str, pretrained: bool = False, **kwargs):
    if pretrained:
        kwargs = {**PRETRAINED_CONFIG, **kwargs}
    try:
        config = _get_config(experiment, **kwargs)
    except FileNotFoundError:
        raise InvalidModelError(f"No configuration found for '{experiment}'") from None
    ModelClass = _get_model_class(experiment)
    model = ModelClass(**config)
    if pretrained:
        load_pretrained_weights(model, experiment, config['charset_train'])
    return model


def load_from_checkpoint(checkpoint_path: str, **kwargs):
    if checkpoint_path.startswith('pretrained='):
        model_id = checkpoint_path.split('=', maxsplit=1)[1]
        model = create_model(model_id, True, **kwargs)
    else:
        ModelClass = _get_model_class(checkpoint_path)
        model = ModelClass.load_from_checkpoint(checkpoint_path, **kwargs)
    return model


def parse_model_args(args):
    kwargs = {}
    arg_types = {t.__name__: t for t in [int, float, str]}
    arg_types['bool'] = lambda v: v.lower() == 'true'  # special handling for bool
    for arg in args:
        name, value = arg.split('=', maxsplit=1)
        name, arg_type = name.split(':', maxsplit=1)
        kwargs[name] = arg_types[arg_type](value)
    return kwargs


def init_weights(module: nn.Module, name: str = '', exclude: Sequence[str] = ()):
    """Initialize the weights using the typical initialization schemes used in SOTA models."""
    if any(map(name.startswith, exclude)):
        return
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
