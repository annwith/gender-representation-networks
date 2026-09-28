# gender-representation-networks

## Piloto completo de redes por ocorrência

Execute o piloto metodológico completo com:

```bash
uv run python scripts/run_network_pilot.py
```

O padrão usa `pierreguillou/gpt2-small-portuguese` (GPT-2 small treinado na Wikipédia em
português) e valida que o texto português padrão contém 100–150 tokens. Os resultados ficam em
`outputs/network_pilot/`: metadados das ocorrências, tabelas de vizinhos/métricas/Jaccard/
dominância/comunidades, seis figuras, `metadata.json` e `report.md`. A rede não direcionada usa a
união das escolhas k-NN; empates são resolvidos pela posição para tornar as repetições lexicais
reproduzíveis.

---

Infraestrutura inicial para um experimento piloto que extrai representações internas do
`Qwen/Qwen3-4B` para pares de prompts contrafactuais. Esta etapa **não** cria grafos, calcula
métricas de redes ou toma decisões metodológicas sobre como transformar ativações em redes.

## Preparação

O projeto usa Python 3.11+ (o ambiente local está fixado em 3.12) e
[`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --dev
```

O modelo é grande. A execução real exige memória suficiente e pode baixar pesos do Hugging
Face. Para modelos com acesso restrito, autentique-se antes com `huggingface-cli login`.

## Entrada e execução

O CSV deve conter `pair_id`, `prompt_a` e `prompt_b`. Colunas adicionais são preservadas como
metadados do par. Um exemplo está em `data/prompts/example_pairs.csv`.

```bash
uv run python scripts/run_pilot.py \
  --config configs/model.yaml \
  --prompts data/prompts/example_pairs.csv \
  --output-dir outputs/activations
```

Também é possível usar `uv run gender-networks-pilot` com os mesmos argumentos. Use
`--log-level DEBUG` para detalhes adicionais.

## Saída

Para cada par, o pipeline escreve:

- `<pair_id>__alignment.json`: relatório de alinhamento e todas as posições divergentes;
- `<pair_id>__a.pt` e `<pair_id>__b.pt`: `input_ids`, `attention_mask` e hidden states;
- `<pair_id>__a.json` e `<pair_id>__b.json`: metadados legíveis da ocorrência.

Os hidden states são movidos para CPU antes da gravação. No `.pt`, a chave `hidden_states`
mapeia o índice configurado para um tensor `[sequence_length, hidden_size]`.

## Decisões explícitas desta etapa

- `hidden_state_indices` indexa diretamente a tupla `outputs.hidden_states` do Transformers:
  índice `0` é a saída dos embeddings, `1` é a saída do primeiro bloco Transformer e `-1` é a
  saída do último bloco. Os padrões `[0, -1]` são apenas uma configuração piloto editável.
- Cada prompt é tokenizado separadamente, com tokens especiais habilitados, sem padding e sem
  truncamento por padrão. Esses comportamentos estão no YAML.
- O alinhamento é estritamente posição a posição. Se as sequências tiverem comprimentos
  diferentes, posições ausentes aparecem como `null`; não há alinhamento por distância de
  edição, palavras ou semântica.
- O forward pass é independente para cada ocorrência e usa toda a sequência. Não há pooling,
  escolha de token representativo, geração de texto ou agregação entre pares.
- Nomes de arquivo usam uma versão sanitizada de `pair_id`; colisões após sanitização são
  rejeitadas para evitar sobrescrita silenciosa.
- O código confia somente no código padrão do Transformers (`trust_remote_code: false`) e fixa a
  revisão configurável em `main`. Para reprodutibilidade forte, troque `main` por um commit do
  repositório do modelo antes de coletar dados definitivos.

## Desenvolvimento

```bash
uv run pytest
uv run ruff check .
```

Os testes usam doubles pequenos de tokenizer/modelo e não baixam pesos.
