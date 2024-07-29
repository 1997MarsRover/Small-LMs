## mamba-minimal

Simple, minimal implementation of Mamba in one file of PyTorch.

Featuring:
* Equivalent numerical output as official implementation for both forward and backward pass
* Simplified, readable, annotated code

Does NOT include:
* Speed. The official implementation is heavily optimized, and these optimizations are core contributions of the Mamba paper. I kept most implementations simple for readability.
* Proper parameter initialization (though this could be added without sacrificing readability)

## Demo

See [demo.ipynb](demo.ipynb) for examples of prompt completions.

```python
from model import Mamba
from transformers import AutoTokenizer

model = Mamba.from_pretrained('state-spaces/mamba-370m')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

generate(model, tokenizer, 'Mamba is the')
```

```yaml
Downloading tokenizer_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 156/156 [00:00<00:00, 18.7kB/s]
Downloading vocab.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.08M/1.08M [00:01<00:00, 890kB/s]
Downloading merges.txt: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 457k/457k [00:00<00:00, 1.17MB/s]
Downloading tokenizer.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.11M/2.11M [00:02<00:00, 988kB/s]
Downloading (…)cial_tokens_map.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 90.0/90.0 [00:00<00:00, 130kB/s]
Mamba is the only species in which the presence of H. walsinghami on the Philippines does not contradict existing morphological observations on its distribution or taxonomy.

References

Category:Aptosticta
Category:Insects described in 1858
```

150 meters... 🫢 scary!

## References

The Mamba architecture was introduced in [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) by [Albert Gu](https://twitter.com/_albertgu?lang=en) and [Tri Dao](https://twitter.com/tri_dao?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor).

The official implementation is here: https://github.com/state-spaces/mamba/tree/main
