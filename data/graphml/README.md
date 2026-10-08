# Redes da entrega parcial (GraphML)

As quatro redes k-NN da entrega parcial de MO438, em duas versões. Os documentos da entrega
estão em `report/entrega-parcial/` (versão não direcionada), `report/entrega-parcial-direcionada/`
e `report/entrega-parcial-visualizacao/` (visualização, item 9).

| Arquivo | Rede |
|---|---|
| `<rede>_k10_union.graphml` | não direcionada por união: aresta {i, j} quando j está entre os 10 vizinhos de i ou i entre os de j |
| `<rede>_k10_directed.graphml` | direcionada: arco i → j quando j está entre os 10 vizinhos de i (todo vértice tem grau de saída 10) |

`<rede>` é a representação de onde vêm os vetores: `lex` (o *embedding* de entrada do token no
Qwen3-4B-Base), `L01`, `L18` e `L36` (a saída bruta dos blocos 1, 18 e 36).

- Os oito arquivos têm os mesmos 13.606 vértices, na mesma ordem: o vértice `i` é a ocorrência
  `occurrence_id = i`. Por isso as redes podem ser comparadas vértice a vértice.
- Os vizinhos de uma ocorrência são as 10 ocorrências de maior cosseno (calculado em `float64`).
  Os empates, que em `lex` aparecem entre todas as ocorrências de um mesmo token, são desfeitos
  ao acaso, com semente 0.
- As redes não são ponderadas: as arestas não têm atributos.
- Sem o sentido dos arcos, a versão direcionada tem exatamente as arestas da versão de união.

## Atributos da rede

| Atributo | Conteúdo |
|---|---|
| `name`, `representation`, `representation_description` | nome do arquivo e representação |
| `k`, `symmetrization`, `similarity`, `tie_breaking` | construção do k-NN |
| `community` | como o atributo `community` dos vértices foi calculado |
| `model`, `corpus` | modelo (com a revisão) e origem do texto |
| `edge_hash` | impressão digital do conjunto de arestas, conferida com a etapa `metrics` do pipeline |

## Atributos dos vértices

Dados da ocorrência, iguais nos oito arquivos:

| Atributo | Tipo | Conteúdo |
|---|---|---|
| `occurrence_id` | inteiro | índice da ocorrência, igual ao índice do vértice |
| `token_id` | inteiro | id do token no vocabulário do Qwen3 |
| `token_text`, `label` | texto | texto do token, com o espaço inicial quando há; `label` repete `token_text` (para o Gephi) |
| `has_leading_space` | booleano | o token começa com espaço |
| `token_category` | texto | `whole_word` (palavra de um token só), `word_start` (primeiro pedaço de uma palavra de vários tokens), `continuation` (pedaço seguinte), `number` (um dígito) ou `punctuation` |
| `is_function_word` | booleano | o token pertence a uma palavra funcional do português (lista fixa em `src/gender_networks/tokens.py`) |
| `word` | texto | palavra a que o token pertence, vazia na pontuação. Palavras são sequências de caracteres alfanuméricos: "tornou-se" tem as palavras "tornou" e "se" |
| `word_n_tokens` | inteiro | número de tokens da palavra (0 na pontuação) |
| `pos_in_word` | inteiro | posição do token na palavra, a partir de 0 (−1 na pontuação). A palavra continua no token seguinte quando `pos_in_word < word_n_tokens - 1` |
| `stratum` | texto | estrato da amostra: `core` (todos os tokens de 80 parágrafos, até 50 ocorrências por token), `target` (palavras polissêmicas), `control` (palavras sem ambiguidade forte) ou `multitheme` (palavras de conteúdo presentes em vários temas) |
| `target_word`, `sense_theme` | texto | palavra-alvo ou de controle da ocorrência e o tema de sentido da sua cota; vazios quando a ocorrência não é de uma dessas palavras |
| `theme` | texto | tema do artigo: `fisica`, `biologia`, `economia`, `politica`, `computacao`, `musica`, `esporte` (categoria Futebol) ou `geografia` |
| `pageid`, `revid`, `title` | inteiro, inteiro, texto | artigo da Wikipédia e a revisão usada |
| `paragraph_id` | texto | `<pageid>-<índice do parágrafo no artigo>` |
| `pos_in_sequence` | inteiro | posição do token na sequência dada ao modelo. A posição 0 é o separador `<\|endoftext\|>`, então o primeiro token do parágrafo está na posição 1 |
| `local_context` | texto | até 8 tokens antes e depois, com o token entre « e » |
| `community` | inteiro | comunidade Leiden (modularidade, resolução 1, melhor de 10 execuções) na rede de união; o mesmo valor nos dois arquivos de cada rede |

Medidas da análise, próprias de cada arquivo:

| Atributo | Arquivos | Conteúdo |
|---|---|---|
| `degree` | união | grau |
| `knn_in_degree` | união | quantas ocorrências escolheram esta entre os seus 10 vizinhos (o grau de entrada na versão direcionada) |
| `local_clustering` | os dois | clusterização local (na versão direcionada, a de Fagiolo, 2007). Vale NaN quando não é definida (vértice com menos de 2 vizinhos), o que não acontece nestas redes |
| `component` | união | componente conexa, numerada por tamanho (0 é a maior) |
| `in_degree`, `out_degree` | direcionada | graus de entrada e de saída (o de saída é sempre 10) |
| `weak_component`, `strong_component` | direcionada | componentes fraca e forte, numeradas por tamanho (0 é a maior) |

## Como abrir

```python
import igraph as ig
import networkx as nx

g = ig.Graph.Read_GraphML("data/graphml/L36_k10_union.graphml")  # inteiros viram float
h = nx.read_graphml("data/graphml/L36_k10_union.graphml", node_type=int)
```

O Gephi abre os arquivos diretamente.

## Como foram gerados

Com `scripts/partial_delivery/export_networks.py --sym union` (ou `--sym directed`), a partir dos
artefatos da execução principal (`outputs/experiment/main/`, que ficam fora do git). O script
reconstrói as arestas com as mesmas funções da etapa `metrics`, confere a impressão digital
(`edge_hash`), grava o GraphML e calcula as métricas da entrega a partir do arquivo relido do
disco. `scripts/partial_delivery/draw_networks.py` desenha as redes e calcula os números da
visualização a partir destes arquivos.
