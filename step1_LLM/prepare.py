"""模型准备、下载与加载，以及运行配置、路径和随机种子。"""
from importlib import import_module
import importlib.metadata as metadata
import hashlib
from copy import deepcopy
from pathlib import Path
import json
import os
import platform
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('WANDB_DISABLED', 'true')
os.environ.setdefault('PYTHONUTF8', '1')
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')


def resolve_model_config(config, model_name=None):
    """把模型预设解析为各阶段共用的配置；旧版快照仍可直接读取。"""
    if not isinstance(config, dict):
        raise ValueError('配置必须为JSON对象')
    result = deepcopy(config)
    profiles = result.pop('model_profiles', None)
    if profiles is not None:
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError('model_profiles 必须为非空对象')
        selected = model_name if model_name is not None else result.get('selected_model')
        if not isinstance(selected, str) or selected not in profiles:
            raise ValueError(f'未知模型 {selected!r}；可选：{", ".join(profiles)}')
        profile = profiles[selected]
        if not isinstance(profile, dict):
            raise ValueError(f'模型预设 {selected} 必须为对象')
        if profile.get('available', True) is not True:
            reason = profile.get('unavailable_reason', '该模型不支持当前训练环境')
            raise ValueError(f'模型 {selected} 当前不可用：{reason}')
        model = {key: value for key, value in profile.items()
                 if key not in {'description', 'available', 'unavailable_reason', 'lora'}}
        overrides = profile.get('lora', {})
        if not isinstance(overrides, dict) or not isinstance(result.get('lora', {}), dict):
            raise ValueError('lora 及模型的 lora 覆盖项必须为对象')
        result['lora'] = {**result.get('lora', {}), **overrides}
        result['model'] = model
        result['selected_model'] = selected
    else:
        model = result.get('model')
        if model_name is not None:
            identities = {result.get('selected_model')}
            if isinstance(model, dict) and isinstance(model.get('name'), str):
                leaf = model['name'].rsplit('/', 1)[-1]
                identities.update((model['name'], leaf, leaf.split('-')[0]))
            if model_name not in identities:
                raise ValueError(f'该配置已绑定模型，不能切换为 {model_name!r}；请使用含 model_profiles 的配置')
    if not isinstance(model, dict) or any(
        not isinstance(model.get(key), str) or not model[key].strip()
        for key in ('name', 'revision', 'local_path')
    ):
        raise ValueError('配置必须包含非空的 model.name、model.revision 和 model.local_path')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', model['name']):
        raise ValueError('model.name 必须为具体的 Hugging Face organization/repository')
    if model.get('trust_remote_code', False) is not False:
        raise ValueError('当前训练路径不支持 trust_remote_code=True 的模型')
    if model.get('prompt_mode', 'system') not in ('system', 'user'):
        raise ValueError('model.prompt_mode 仅支持 system 或 user')
    kwargs = model.get('chat_template_kwargs', {'enable_thinking': False})
    if not isinstance(kwargs, dict) or {'tokenize', 'add_generation_prompt', 'conversation'} & kwargs.keys():
        raise ValueError('chat_template_kwargs 必须为对象，且不能覆盖模板的输入与返回方式')
    return result


def read_config(path, model_name=None):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return resolve_model_config(json.loads(path.read_text(encoding='utf-8')), model_name)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_path(config):
    path = Path(config['model']['local_path'])
    return path if path.is_absolute() else ROOT / path


def dataset_task(config):
    """加载所选数据集的读取、提示构造和判分规则。"""
    name = config.get('dataset', 'calculator')
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name):
        raise ValueError(f'无效的数据集名：{name!r}')
    if not (ROOT / 'Datasets' / name / 'task.py').is_file():
        raise FileNotFoundError(f'数据集 {name} 尚未接入，缺少 Datasets/{name}/task.py')
    return import_module(f'Datasets.{name}.task')


def render_prompt(tokenizer, record, config=None):
    system_prompt = dataset_task(config or {}).SYSTEM_PROMPT
    model = (config or {}).get('model', {})
    mode = model.get('prompt_mode', 'system')
    if mode == 'system':
        messages = [{'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': record['prompt']}]
    elif mode == 'user':
        messages = [{'role': 'user', 'content': system_prompt + '\n\n' + record['prompt']}]
    else:
        raise ValueError(f'不支持的 prompt_mode：{mode}')
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        **model.get('chat_template_kwargs', {'enable_thinking': False}))


def validate_tokenizer(tokenizer):
    if not isinstance(tokenizer.eos_token, str) or not tokenizer.eos_token or not isinstance(tokenizer.eos_token_id, int) or tokenizer.eos_token_id < 0:
        raise ValueError('模型tokenizer缺少有效EOS token，无法构造监督答案或结束生成')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not isinstance(tokenizer.pad_token_id, int) or tokenizer.pad_token_id < 0:
        raise ValueError('模型tokenizer缺少有效PAD token')
    try:
        template = tokenizer.get_chat_template()
    except (ValueError, AttributeError) as error:
        raise ValueError('模型tokenizer缺少可用的chat template') from error
    if not isinstance(template, str) or not template.strip():
        raise ValueError('模型tokenizer缺少可用的chat template')


def _model_identity(path):
    info = Path(path) / 'model_info.json'
    if not info.is_file():
        return None
    return json.loads(info.read_text(encoding='utf-8'))


def validate_adapter_base(config, adapter):
    """加载LoRA前核验基座路径/仓库；存在训练快照时也核验revision。"""
    adapter = Path(adapter)
    if not adapter.is_absolute():
        adapter = ROOT / adapter
    settings = json.loads((adapter / 'adapter_config.json').read_text(encoding='utf-8'))
    base = settings.get('base_model_name_or_path')
    model = config['model']
    matches = base == model['name']
    if isinstance(base, str) and base:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = ROOT / base_path
        matches = matches or base_path.resolve() == model_path(config).resolve()
        if not matches and base_path.is_dir():
            identity = _model_identity(base_path)
            matches = bool(identity and identity.get('repository') == model['name']
                           and identity.get('revision') == model['revision'])
    if not matches:
        raise ValueError(f'adapter基座 {base!r} 与所选模型 {model["name"]!r} 不一致')
    revision = settings.get('revision')
    if revision and revision != model['revision']:
        raise ValueError('adapter基座revision与所选模型不一致')
    summary = adapter / 'run_summary.json'
    if summary.is_file():
        source_model = json.loads(summary.read_text(encoding='utf-8')).get('config', {}).get('model', {})
        if source_model and any(source_model.get(key) != model[key] for key in ('name', 'revision')):
            raise ValueError('adapter训练快照的基座模型或revision与所选模型不一致')
    return adapter


def load_model_and_tokenizer(config, adapter=None, trainable=False):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    config = resolve_model_config(config)
    path = model_path(config)
    if not (path / 'config.json').is_file():
        raise FileNotFoundError(f'请先用 main.py 执行模型准备；缺少模型目录 {path}')
    identity = _model_identity(path)
    if not identity or identity.get('repository') != config['model']['name'] or identity.get('revision') != config['model']['revision']:
        raise ValueError('本地模型仓库/revision身份未核验或与配置不一致；请先准备模型')
    adapter_path = validate_adapter_base(config, adapter) if adapter else None
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'left'
    validate_tokenizer(tokenizer)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False, dtype=dtype,
        attn_implementation=config['model'].get('attn_implementation', 'sdpa'))
    model.to(device)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    # Transformers会从checkpoint补齐默认值，必须统一实际的采样分布。
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 0
    model.generation_config.min_p = None
    model.generation_config.repetition_penalty = 1.0
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=trainable)
    if trainable:
        model.train()
    else:
        model.eval()
    return model, tokenizer


def _complete_local_model(path):
    """核验配置、tokenizer及safetensors分片尺寸，拒绝截断的本地权重。"""
    if not all((path / name).is_file() for name in ('config.json', 'tokenizer_config.json')):
        return False
    if not ((path / 'tokenizer.json').is_file() or
            all((path / name).is_file() for name in ('vocab.json', 'merges.txt'))):
        return False
    index = path / 'model.safetensors.index.json'
    try:
        shards = set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values()) if index.is_file() else {'model.safetensors'}
        if not shards:
            return False
        for name in shards:
            shard = path / name
            if shard.parent.resolve() != path.resolve():
                return False
            with shard.open('rb') as stream:
                header_size = int.from_bytes(stream.read(8), 'little')
                if not 0 < header_size <= min(shard.stat().st_size - 8, 100_000_000):
                    return False
                header = json.loads(stream.read(header_size))
            tensors = [value for key, value in header.items() if key != '__metadata__']
            if not tensors or max(value['data_offsets'][1] for value in tensors) != shard.stat().st_size - 8 - header_size:
                return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def resolve_model_revision(config):
    """在创建实验路径前固定commit；已固定的SHA不访问Hub。"""
    model = config['model']
    revision = model['revision']
    if re.fullmatch(r'[0-9a-fA-F]{40}', revision):
        return revision
    from huggingface_hub import HfApi
    resolved = HfApi().model_info(model['name'], revision=revision).sha
    if not isinstance(resolved, str) or not re.fullmatch(r'[0-9a-fA-F]{40}', resolved):
        raise ValueError('无法把模型revision解析为不可变commit SHA')
    model['requested_revision'] = revision
    model['revision'] = resolved
    return resolved


def prepare_model(config):
    revision = resolve_model_revision(config)
    model = config['model']
    dest = model_path(config)
    requested = model.get('requested_revision', revision)
    download = model.get('download')
    source_repo = download['repository'] if download else model['name']
    source_revision = download['revision'] if download else revision
    source = {'repository': source_repo, 'revision': source_revision}
    pending_path = dest / '.cache' / 'model-download.json'
    pending = json.loads(pending_path.read_text(encoding='utf-8')) if pending_path.is_file() else None
    identity = _model_identity(dest)
    same_identity = (isinstance(identity, dict) and identity.get('repository') == model['name']
                     and identity.get('revision') == revision)
    if download and same_identity:
        same_identity = identity.get('download') == download
    # Hub离线时可能直接返回非空local_dir，不能把旧权重重新标记为本次revision。
    resumable = identity is None and pending == source
    if (dest.exists() and any(path.name != '.cache' for path in dest.iterdir())
            and not same_identity and not resumable):
        raise ValueError(f'模型目录非空且仓库/revision身份缺失或不一致：{dest}；请指定新的 model.local_path')
    if same_identity and _complete_local_model(dest):
        print('Model ready (verified local):', dest)
        return dest
    from huggingface_hub import snapshot_download
    write_json(pending_path, source)
    snapshot_download(repo_id=source_repo, revision=source_revision, local_dir=str(dest),
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', 'LICENSE'], max_workers=4)
    if not _complete_local_model(dest):
        raise ValueError(f'模型下载不完整或不支持当前safetensors加载路径：{dest}')
    identity = {'repository': model['name'], 'revision': revision,
                'requested_revision': requested, 'source': 'https://huggingface.co/' + source_repo}
    if download:
        for filename, expected in download['sha256'].items():
            with (dest / filename).open('rb') as file:
                if hashlib.file_digest(file, 'sha256').hexdigest() != expected:
                    raise ValueError(f'权重来源校验失败：{dest / filename}')
        identity['download'] = download
    write_json(dest / 'model_info.json', identity)
    print('Model ready:', dest)
    return dest


def check_environment() -> dict:
    """检查用于训练的 CUDA 环境，并执行 BF16 正向与反向。"""
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('训练需要支持 BF16 的 CUDA PyTorch 环境')
    x = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    (x @ x).float().mean().backward()
    torch.cuda.synchronize()
    if not bool(torch.isfinite(x.grad).all()):
        raise RuntimeError('GPU BF16 反向传播产生了非有限梯度')
    device = torch.cuda.get_device_properties(0)
    return {
        'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
        'cuda_available': True, 'torch_cuda': torch.version.cuda,
        'gpu': device.name, 'memory_gib': device.total_memory / 2**30,
        'compute_capability': list(torch.cuda.get_device_capability(0)),
        'bf16': True, 'bf16_backward_ok': True,
        'packages': {name: metadata.version(name) for name in
                     ('torch', 'transformers', 'trl', 'peft', 'accelerate', 'datasets')},
    }
