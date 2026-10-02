# Da rede lexical à rede contextual

Projeto da disciplina MO438 (Redes Complexas, Unicamp). Ele usa redes complexas para descrever
como as representações de tokens de um Transformer passam do espaço lexical de entrada para os
espaços contextuais das camadas internas.

- **Vértices:** *ocorrências* de tokens (cerca de 15 mil) em parágrafos da Wikipédia em
  português, escolhidos em 8 temas.
- **Arestas:** k vizinhos mais próximos (k-NN) pelo cosseno, calculado em `float64`, com três
  tratamentos explícitos de empate, nas versões direcionada, por união e mútua.
- **Representações:** os mesmos vértices recebem o *embedding* de entrada do seu token e a saída
  bruta dos blocos 1, 18 e 36 do
  [Qwen/Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base) (e dos 36 blocos para a curva
  camada a camada). Assim as redes podem ser comparadas vértice a vértice.
- **Rede lexical de tipos:** k-NN entre os cerca de 151,6 mil tokens do vocabulário do modelo.

As perguntas de pesquisa (P1–P4) estão na proposta (`report/report.tex`). As decisões de método
e o cronograma estão no plano de execução (`report/plano-de-execucao.md`). O relatório técnico,
escrito junto com a implementação, fica em `report/relatorio/`.

> Estado: os módulos das etapas `corpus`, `sample`, `extract`, `knn`, `metrics` e `lens` existem
> (os das últimas ainda em revisão). As etapas `analyze` e `report` ainda **não estão
> implementadas** (`gender_networks.analysis` e `gender_networks.report` não existem), então
> `analyze`, `report` e `all` ainda falham com `ModuleNotFoundError` (`all`, depois de
> `metrics`). Ainda não há resultados; o relatório marca como *[pendente]* tudo o que depende
> de uma execução.

## Estrutura

```
configs/
  experiment.yaml      execução principal (uma execução = um YAML)
  mini.yaml            ensaio rápido de ponta a ponta (herda de experiment.yaml)
  model.yaml           configuração do carregamento do modelo (código herdado, config.py)
src/gender_networks/
  cli.py               CLI `gender-networks`: um subcomando por etapa
  settings.py          configuração tipada de uma execução
  artifacts.py         contrato entre etapas: caminhos, colunas, manifestos
  wiki.py, textclean.py, corpus.py   etapa corpus (download com cache e limpeza da prosa)
  tokens.py, sampling.py             etapa sample (categorias de token, amostra híbrida)
  extract.py                         etapa extract (embeddings e estados do Qwen3-4B-Base)
  knn.py                             etapa knn (candidatos, vizinhos e tratamentos de empate)
  graphs.py, metrics.py, partitions.py   etapa metrics (métricas estruturais, Leiden, NMI/ARI)
  neighborhood.py                    medidas de vizinhança (Jaccard, dominância, J^w, piso de ruído)
  vocab_lens.py                      extensão opcional "vizinhos no vocabulário" (etapa lens)
  analysis.py                        etapa analyze (tabelas de P1–P4)
  report.py                          etapa report (ainda não implementada)
  network_pilot.py, network_analysis.py, tokenization.py
                                     piloto congelado (não alterar)
  modeling.py, config.py             utilitários herdados de carregamento do modelo
tests/                 testes offline (não baixam pesos)
report/
  report.tex           proposta revisada
  plano-de-execucao.md plano de execução
  relatorio/           relatório técnico vivo (ver report/relatorio/README.md)
scripts/
  feasibility/         estudo de viabilidade: scripts, saídas (sim_*.txt, hyb_*.txt) e figuras
  run_network_pilot.py piloto congelado
```

## Preparação

O projeto usa Python 3.12 e [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --dev
```

O modelo (cerca de 7,5 GiB em bf16) é baixado do Hugging Face na primeira execução da etapa
`extract`, na revisão fixada em `configs/experiment.yaml`.

## Pipeline

Cada etapa lê os artefatos da anterior e pode ser refeita sozinha. Uma etapa cujas saídas já
existem é pulada, a menos que se passe `--force`.

```bash
uv run gender-networks corpus  --config configs/experiment.yaml  # baixa e limpa a Wikipédia
uv run gender-networks sample  --config configs/experiment.yaml  # amostra híbrida de ocorrências
uv run gender-networks extract --config configs/experiment.yaml --verify  # checagens no modelo real
uv run gender-networks extract --config configs/experiment.yaml  # representações
uv run gender-networks knn     --config configs/experiment.yaml  # candidatos e vizinhos k-NN
uv run gender-networks metrics --config configs/experiment.yaml  # métricas e comunidades
uv run gender-networks analyze --config configs/experiment.yaml  # análises de P1–P4
uv run gender-networks report  --config configs/experiment.yaml  # figuras, tabelas e números (ainda não implementada)
uv run gender-networks lens    --config configs/experiment.yaml  # extensão opcional
uv run gender-networks all     --config configs/experiment.yaml  # corpus → report, em ordem (depende de analyze e report)
```

`--config configs/experiment.yaml` é o padrão. Para o ensaio de ponta a ponta (2 temas, cerca de
1.500 vértices), use `--config configs/mini.yaml`. O ensaio usa o mesmo corpus da execução
principal e grava suas redes e seu relatório gerado separados.

### Onde ficam os artefatos

| Caminho | Conteúdo |
|---|---|
| `data/raw/wiki/` | cache das respostas cruas da API da Wikipédia (ignorado pelo git) |
| `data/corpus/` | `articles.jsonl`, `paragraphs.jsonl` (ignorados) e `manifest.csv` (versionado) |
| `outputs/experiment/<nome>/sample/` | `occurrences.csv`, `sequences.jsonl`, `vocab_types.csv`, `senses.csv` |
| `outputs/experiment/<nome>/reps/` | representações em bf16 (safetensors), memmap das 36 camadas, diagnósticos |
| `outputs/experiment/<nome>/knn/` | candidatos e conjuntos de vizinhos (`.npz`) |
| `outputs/experiment/<nome>/metrics/`, `analysis/`, `lens/` | tabelas CSV/JSON |
| `report/relatorio/figuras/`, `tabelas/` | material gerado para o relatório |

`<nome>` é o campo `name` do YAML (`main` ou `mini`). Cada pasta de etapa tem um
`_manifest.json` com a configuração, as versões das bibliotecas, os tempos e as estatísticas da
etapa.

## Relatório técnico

O relatório (`report/relatorio/relatorio.tex`, em LaTeX) é atualizado a cada etapa. Figuras
(`figuras/*.pdf`), tabelas (`tabelas/*.tex`) e os números citados no texto
(`tabelas/numeros.tex`) são gerados pela etapa `report` e **nunca são editados à mão**. O que
ainda não foi gerado aparece como um quadro "pendente", e o documento compila antes de qualquer
execução:

```bash
uv run gender-networks report --config configs/experiment.yaml   # (ainda não implementada)
cd report/relatorio && latexmk -pdf relatorio.tex
```

As duas figuras do estudo de viabilidade são geradas à parte, a partir dos números fixos do
estudo:

```bash
uv run python scripts/feasibility/plot_feasibility.py
```

Detalhes em `report/relatorio/README.md`.

## Testes e lint

```bash
uv run pytest
uv run ruff check .
```

Os testes são offline e rodam na CPU. Os que precisam do tokenizer do Qwen3 usam o cache local
do Hugging Face e são pulados se ele não estiver disponível.

## Reprodutibilidade

- **Modelo:** `Qwen/Qwen3-4B-Base` na revisão `906bfd4b4dc7f14ee4320094d8b41684abff8539`.
- **Dados:** `data/corpus/manifest.csv` guarda `pageid` e `revid` de cada artigo, então o mesmo
  texto pode ser baixado de novo mesmo que os artigos mudem. O cache cru torna a limpeza
  refazível sem rede.
- **Sementes:** todas vêm do YAML (`corpus.seed`, `sample.seed`, `networks.base_seed`, 438 na
  execução principal). A semente `r` do desempate da representação `rep` com `k` vizinhos usa
  `numpy.random.default_rng([base_seed, crc32(rep), k, r])`.
- **Versões:** `uv.lock` fixa as dependências; cada `_manifest.json` registra as versões usadas.

## Hardware

A execução foi planejada para uma RTX 4070 de notebook (8 GiB), 20 núcleos e 15 GiB de RAM. Os
pesos não cabem inteiros na GPU junto com as ativações. Por isso a extração usa
`device_map="auto"` com `max_memory` (6 GiB na GPU e 10 GiB na CPU, em `configs/experiment.yaml`):
parte dos blocos fica na CPU e é transferida durante o *forward*. Isso muda o tempo, não o
resultado. Se faltar memória, reduza `max_memory`. As similaridades do k-NN são calculadas na
CPU em `float64`, em blocos. A rede do vocabulário ocupa cerca de 3 GB de memória nesse cálculo.

## Piloto congelado

O piloto metodológico (GPT-2 small português, texto fixo de 104 tokens) fica congelado, rodando
e com testes, como registro dos achados que motivaram o método (empates de cosseno 1, quebra em
subpalavras, direcionalidade):

```bash
uv run python scripts/run_network_pilot.py
```

Os resultados vão para `outputs/network_pilot/`. O pipeline novo não depende dele, mas um teste de
regressão exige que o k-NN novo (desempate por posição, união) reproduza o do piloto.
