# Da rede lexical à rede contextual

Projeto da disciplina MO438 (Redes Complexas, Unicamp). Ele usa redes complexas para descrever
como as representações de tokens de um Transformer passam do espaço lexical de entrada para os
espaços contextuais das camadas internas.

- **Vértices:** *ocorrências* de tokens (13.606 na execução principal) em parágrafos da
  Wikipédia em português, escolhidos em 8 temas.
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

> Estado: todas as etapas estão implementadas e testadas, e a execução principal
> (`configs/experiment.yaml`) já rodou de ponta a ponta. As figuras, tabelas e números do
> relatório técnico vêm dela. A entrega parcial da disciplina está pronta (veja
> [Entrega parcial](#entrega-parcial)). Faltam a anotação manual dos sentidos
> (`sample/senses.csv`) e parte da escrita do relatório técnico (resultados, robustez e
> discussão).

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
  report.py, plots.py                etapa report (figuras, tabelas e números do relatório)
  network_pilot.py, network_analysis.py, tokenization.py
                                     piloto congelado (não alterar)
  modeling.py, config.py             utilitários herdados de carregamento do modelo
tests/                 testes offline (não baixam pesos)
data/
  corpus/manifest.csv  pageid e revid de cada artigo do corpus
  graphml/             redes da entrega parcial em GraphML (ver data/graphml/README.md)
  crawl/               rede da coleta e a página interativa (rede, artigos e amostra)
docs/redes/            página interativa das redes da entrega parcial (GitHub Pages)
docs/coleta/           cópia da página da coleta, do corpus e da amostra (GitHub Pages)
report/
  report.tex           proposta revisada
  plano-de-execucao.md plano de execução
  relatorio/           relatório técnico vivo (ver report/relatorio/README.md)
  entrega-parcial/               entrega parcial: redes não direcionadas (itens 1 a 8)
  entrega-parcial-direcionada/   entrega parcial: redes direcionadas (itens 1 a 8)
  entrega-parcial-visualizacao/  entrega parcial: visualização das redes (item 9)
scripts/
  feasibility/         estudo de viabilidade: scripts, saídas (sim_*.txt, hyb_*.txt) e figuras
  partial_delivery/    GraphML, figuras, tabelas e página interativa da entrega parcial
  crawl_network/       rede da coleta e página da coleta, do corpus e da amostra
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
uv run gender-networks corpus-rebuild --force  # refaz o corpus das revisões de manifest.csv
uv run gender-networks sample  --config configs/experiment.yaml  # amostra híbrida de ocorrências
uv run gender-networks extract --config configs/experiment.yaml --verify  # checagens no modelo real
uv run gender-networks extract --config configs/experiment.yaml  # representações
uv run gender-networks knn     --config configs/experiment.yaml  # candidatos e vizinhos k-NN
uv run gender-networks metrics --config configs/experiment.yaml  # métricas e comunidades
uv run gender-networks analyze --config configs/experiment.yaml  # análises de P1–P4
uv run gender-networks report  --config configs/experiment.yaml  # figuras, tabelas e números
uv run gender-networks lens    --config configs/experiment.yaml  # extensão opcional
uv run gender-networks all     --config configs/experiment.yaml  # corpus → report, em ordem (o corpus só é refeito com --force-corpus)
```

O corpus é compartilhado por todas as execuções e custa requisições à API: o `all --force` não o
refaz, só o `all --force-corpus`. Com `GENDER_NETWORKS_OFFLINE=1`, as etapas `corpus` e
`corpus-rebuild` não acessam a rede e param com erro (`CacheMiss`) se faltar uma resposta no
cache. Use isso para refazer a limpeza depois de mudar `textclean.py` sem baixar revisões mais
novas. O `corpus-rebuild` baixa por `revid` as revisões dos artigos mantidos em
`data/corpus/manifest.csv`, limpa-as com o mesmo código e na mesma ordem do `corpus` e reescreve
`articles.jsonl` e `paragraphs.jsonl`. Como o manifest não registra a busca por categorias, os
campos `themes_reached`, `depth` e `origin` de `articles.jsonl` ficam vazios. A primeira
reconstrução precisa de rede, porque o cache do `corpus` está indexado por título.

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
| `data/graphml/` | redes da entrega parcial em GraphML (versionadas) |
| `report/entrega-parcial*/figuras/`, `tabelas/` | material gerado para a entrega parcial |
| `docs/redes/` | página interativa das redes da entrega parcial e os seus dados |
| `data/crawl/`, `docs/coleta/` | rede da coleta e página dos artigos do corpus |

`<nome>` é o campo `name` do YAML (`main` ou `mini`). Cada pasta de etapa tem um
`_manifest.json` com a configuração, as versões das bibliotecas, os tempos e as estatísticas da
etapa.

### Conferir os artigos coletados e a amostra

A página `data/crawl/rede-de-coleta.html` mostra a coleta em três abas:

- **Rede:** as buscas por categorias.
- **Artigos:** os 2.631 candidatos da etapa `corpus`, com resumo por tema e motivo de descarte,
  busca no título ou no texto, filtros, o link para a revisão exata usada (`oldid`) e o texto
  limpo dos artigos mantidos.
- **Amostra:** como a etapa `sample` forma os 13.606 vértices das redes. A aba tem:
  - as contagens por estrato e tema;
  - os 80 parágrafos do núcleo token a token: vértice, cortado pelo limite `f_max` ou espaço, e
    se o artigo veio da raiz do tema ou de uma categoria extra;
  - as cotas das palavras-alvo e de controle e as 150 palavras multitema, cada uma com a
    concordância de todas as suas ocorrências.

A página é um arquivo único que abre direto no navegador. Uma cópia fica em `docs/coleta/` e é
publicada em <https://annwith.github.io/gender-representation-networks/coleta/>. O endereço
abre direto numa aba com `#artigos` ou `#amostra`.

Para regerar as duas, o script usa três coisas locais: o cache da API, `data/corpus/` e a
amostra em `outputs/experiment/main/sample/` (escolha outra pasta com `--sample`). Ele também
precisa do tokenizer do Qwen3 no cache do Hugging Face e não acessa a rede. Antes de gravar, o
script confere a amostra contra o corpus e o `_manifest.json`.

```bash
.venv/bin/python scripts/crawl_network/build_crawl_network.py
```

## Relatório técnico

O relatório (`report/relatorio/relatorio.tex`, em LaTeX) é atualizado a cada etapa. Figuras
(`figuras/*.pdf`), tabelas (`tabelas/*.tex`) e os números citados no texto
(`tabelas/numeros.tex`) são gerados pela etapa `report` e **nunca são editados à mão**. O que
ainda não foi gerado aparece como um quadro "pendente", e o documento compila antes de qualquer
execução:

```bash
uv run gender-networks report --config configs/experiment.yaml
cd report/relatorio && latexmk -pdf relatorio.tex
```

As duas figuras do estudo de viabilidade são geradas à parte, a partir dos números fixos do
estudo:

```bash
uv run python scripts/feasibility/plot_feasibility.py
```

Detalhes em `report/relatorio/README.md`.

## Entrega parcial

A entrega parcial da disciplina tem três documentos, cada um com o seu PDF compilado:

| Documento | Conteúdo |
|---|---|
| `report/entrega-parcial/entrega-parcial.pdf` | dados, construção das redes e itens 1 a 8 nas redes não direcionadas por união |
| `report/entrega-parcial-direcionada/entrega-parcial-direcionada.pdf` | os mesmos itens nas redes direcionadas |
| `report/entrega-parcial-visualizacao/entrega-parcial-visualizacao.pdf` | visualização das redes (item 9) |

As quatro redes (lex, L01, L18 e L36, com k = 10) estão em GraphML em `data/graphml/`, nas duas
versões, com os atributos descritos em `data/graphml/README.md`. A versão interativa da
visualização fica em `docs/redes/` e é publicada pelo GitHub Pages em
<https://annwith.github.io/gender-representation-networks/redes/>.

`export_networks.py` lê os artefatos da execução principal (`outputs/experiment/main/`) e grava
os GraphML, as figuras e a tabela de métricas de uma versão; `draw_networks.py` parte dos GraphML
e grava as figuras e tabelas da visualização e a página interativa:

```bash
.venv/bin/python scripts/partial_delivery/export_networks.py --sym union
.venv/bin/python scripts/partial_delivery/export_networks.py --sym directed
.venv/bin/python scripts/partial_delivery/draw_networks.py
cd report/entrega-parcial && latexmk -pdf entrega-parcial.tex  # idem nas outras duas pastas
```

Os textos dos três documentos são escritos à mão, a partir das tabelas geradas.

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
  texto pode ser baixado de novo mesmo que os artigos mudem
  (`uv run gender-networks corpus-rebuild --force`). O cache cru torna a limpeza refazível sem
  rede (`GENDER_NETWORKS_OFFLINE=1`).
- **Sementes:** todas vêm do YAML (`corpus.seed`, `sample.seed`, `networks.base_seed`, 438 na
  execução principal). A semente `r` do desempate da representação `rep` com `k` vizinhos usa
  `numpy.random.default_rng([base_seed, crc32(rep), k, r])`.
- **Versões:** `uv.lock` fixa as dependências; cada `_manifest.json` registra as versões usadas.

## Hardware

A execução principal usou uma única RTX A5500 (24 GiB). A extração usa `device_map="auto"` com
`max_memory` (20 GiB na GPU e 10 GiB na CPU, em `configs/experiment.yaml`), e os pesos (cerca de
7,5 GiB em bf16) cabem inteiros na GPU. Numa GPU de 8 GiB, como a RTX 4070 de notebook usada no
desenvolvimento, use 6 GiB na GPU: parte dos blocos fica na CPU e é transferida durante o
*forward*. Isso muda o tempo, não o resultado. Se faltar memória, reduza `max_memory`. As
similaridades do k-NN são calculadas na CPU em `float64`, em blocos. A rede do vocabulário ocupa
cerca de 3 GB de memória nesse cálculo.

## Piloto congelado

O piloto metodológico (GPT-2 small português, texto fixo de 104 tokens) fica congelado, rodando
e com testes, como registro dos achados que motivaram o método (empates de cosseno 1, quebra em
subpalavras, direcionalidade):

```bash
uv run python scripts/run_network_pilot.py
```

Os resultados vão para `outputs/network_pilot/`. O pipeline novo não depende dele, mas um teste de
regressão exige que o k-NN novo (desempate por posição, união) reproduza o do piloto.
