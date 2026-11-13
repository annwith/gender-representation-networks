# Revisão técnica do projeto

Revisão feita em 12 e 13/11/2026 sobre o commit `5bdf56f` (branch `main`). Só docstrings e
comentários foram alterados; nenhum problema listado aqui foi corrigido, e nada foi commitado.

Convenções deste documento:

- **Gravidade:** crítica, alta, média ou baixa.
- **Confiança:** *confirmado* (reproduzido com um teste não destrutivo ou demonstrado pela leitura
  do código), *provável* (a leitura indica o problema, mas falta um dado para fechar) e *requer
  investigação* (depende de artefatos que não estão nesta máquina).
- Os números de linha se referem aos arquivos já documentados (estado atual da árvore de
  trabalho), não ao commit `5bdf56f`.
- Limitações que o próprio relatório técnico já registra (`11-limitacoes.tex`,
  `12-problemas.tex`) não são repetidas como achados.

## A. Resumo executivo

O código está bem organizado e, no que é central para a pesquisa, correto. O pipeline tem uma
etapa por módulo, um contrato único de caminhos e nomes (`artifacts.py`), convenções com um só
dono (`representations.py`, `labels.py`) e manifestos por etapa. A construção das redes, que é
a parte mais delicada (cosseno em float64, listas de candidatos completas, seletor de empates
com tolerância, sorteio com sementes reprodutíveis, derivação exata da rede lexical), foi lida
linha a linha e confere com o que `06-redes.tex` descreve; não encontrei erro de lógica nela. Os
testes cobrem bem esse núcleo, com comparações contra força bruta e contra o NetworkX.

Nenhum problema crítico foi encontrado. Há **um problema de gravidade alta**: a figura de P2
com a fração de vértices cuja maior mudança cai em cada transição usa o Jaccard bruto e, por
isso, conta como reorganização o que é só ruído do sorteio de empates da rede lexical (C-01).
No ensaio mini, descontar o piso de ruído leva a parcela de `lex→L01` de 12,2% para 2,5%. Há
**quatro problemas de gravidade média**: duas afirmações do relatório que divergem do que o
código faz ou do que as tabelas geradas mostram (cotas das palavras-alvo, C-02; limite de
iterações do Leiden, C-03) e duas fragilidades de reprodutibilidade (etapa refeita com `--force`
e interrompida, C-04; proveniência do corpus nos manifestos, C-05). Os outros 44 itens são de
gravidade baixa: casos de borda, convenções duplicadas, validação incompleta da configuração e
pontos em que texto e código usam palavras diferentes para a mesma coisa.

A documentação existente era boa no nível de módulo (todo módulo tinha uma docstring que
explica o porquê das decisões) e irregular no nível de função: 370 das 684 funções do escopo
tinham docstring, quase sempre de uma linha e sem parâmetros, retorno ou exceções. Agora todas
as 684 funções e 54 classes do escopo têm docstring no estilo Google, com parâmetros conferidos
automaticamente contra as assinaturas.

## B. Cobertura da documentação

**Escopo editado** (decidido com a autora): `src/gender_networks/**` e
`scripts/{crawl_network,dev,partial_delivery}/*.py`. Foram escritas ou reescritas 732
docstrings em 43 arquivos. Seis definições (uma propriedade e cinco classes pequenas, entre
elas as exceções) já tinham docstring adequada e ficaram como estavam; os dois `__init__.py`
só têm docstring de módulo e não mudaram. As
docstrings estão em inglês (regra de idioma do `AGENTS.md`), com as seções `Args`, `Returns` ou
`Yields`, `Raises`, `Side effects` e `Notes` quando se aplicam; classes têm `Attributes`. O
texto que já existia foi mantido e ampliado. Também entraram comentários em quatro constantes
(`POS_BUCKETS`, `MANIFEST`, as listas de tabelas e `TRUE_TEXT` em `artifacts.py`; `PIPELINE` em
`cli.py`).

| Grupo | Arquivos | Funções | Com docstring antes | Depois | Classes | Com docstring antes | Depois |
|---|---:|---:|---:|---:|---:|---:|---:|
| `src/gender_networks` | 40 | 601 | 335 | 601 | 54 | 49 | 54 |
| `scripts/crawl_network` | 1 | 17 | 10 | 17 | 0 | 0 | 0 |
| `scripts/dev` | 1 | 16 | 2 | 16 | 0 | 0 | 0 |
| `scripts/partial_delivery` | 3 | 50 | 23 | 50 | 0 | 0 | 0 |
| **Total editado** | **45** | **684** | **370** | **684** | **54** | **49** | **54** |
| `scripts/feasibility` (só lido) | 4 | 28 | 9 | 9 | 2 | 1 | 1 |
| `tests` (só lido) | 20 | 376 | 23 | 23 | 8 | 3 | 3 |

"Com docstring antes" conta só a presença; as 370 que existiam também foram completadas com
parâmetros, retorno e exceções.

Por arquivo (funções / com docstring antes / depois; classes / antes / depois):

| Arquivo | Funções | Antes | Depois | Classes | Antes | Depois |
|---|---:|---:|---:|---:|---:|---:|
| `src/gender_networks/analysis.py` | 28 | 15 | 28 | 0 | 0 | 0 |
| `src/gender_networks/artifacts.py` | 68 | 20 | 68 | 1 | 1 | 1 |
| `src/gender_networks/cli.py` | 5 | 1 | 5 | 0 | 0 | 0 |
| `src/gender_networks/corpus.py` | 10 | 6 | 10 | 3 | 1 | 3 |
| `src/gender_networks/corpus_rebuild.py` | 3 | 3 | 3 | 0 | 0 | 0 |
| `src/gender_networks/extract.py` | 43 | 29 | 43 | 3 | 3 | 3 |
| `src/gender_networks/graphs.py` | 25 | 25 | 25 | 0 | 0 | 0 |
| `src/gender_networks/knn.py` | 45 | 28 | 45 | 4 | 3 | 4 |
| `src/gender_networks/labels.py` | 6 | 5 | 6 | 0 | 0 | 0 |
| `src/gender_networks/metrics.py` | 35 | 20 | 35 | 7 | 7 | 7 |
| `src/gender_networks/neighborhood.py` | 17 | 16 | 17 | 0 | 0 | 0 |
| `src/gender_networks/partitions.py` | 14 | 12 | 14 | 1 | 1 | 1 |
| `src/gender_networks/plots.py` | 46 | 33 | 46 | 4 | 4 | 4 |
| `src/gender_networks/representations.py` | 18 | 11 | 18 | 1 | 1 | 1 |
| `src/gender_networks/sampling.py` | 44 | 31 | 44 | 3 | 3 | 3 |
| `src/gender_networks/settings.py` | 7 | 4 | 7 | 14 | 14 | 14 |
| `src/gender_networks/textclean.py` | 16 | 11 | 16 | 0 | 0 | 0 |
| `src/gender_networks/tokens.py` | 14 | 12 | 14 | 2 | 1 | 2 |
| `src/gender_networks/vocab_lens.py` | 32 | 23 | 32 | 2 | 2 | 2 |
| `src/gender_networks/wiki.py` | 14 | 9 | 14 | 5 | 5 | 5 |
| `src/gender_networks/report/__init__.py` | 4 | 3 | 4 | 1 | 0 | 1 |
| `src/gender_networks/report/context.py` | 25 | 4 | 25 | 2 | 2 | 2 |
| `src/gender_networks/report/latex.py` | 15 | 6 | 15 | 1 | 1 | 1 |
| `src/gender_networks/report/numbers.py` | 4 | 1 | 4 | 0 | 0 | 0 |
| `src/gender_networks/report/sections/` (14 módulos) | 63 | 7 | 63 | 0 | 0 | 0 |
| `scripts/crawl_network/build_crawl_network.py` | 17 | 10 | 17 | 0 | 0 | 0 |
| `scripts/dev/compare_runs.py` | 16 | 2 | 16 | 0 | 0 | 0 |
| `scripts/partial_delivery/draw_networks.py` | 17 | 9 | 17 | 0 | 0 | 0 |
| `scripts/partial_delivery/draw_presentation.py` | 6 | 5 | 6 | 0 | 0 | 0 |
| `scripts/partial_delivery/export_networks.py` | 27 | 9 | 27 | 0 | 0 | 0 |

**Lido e auditado, sem edição:** `tests/` (o pedido proíbe alterar testes), `scripts/feasibility`
(registro congelado do estudo de viabilidade; um teste lê o texto-fonte de `simulate_sample.py`),
`scripts/cluster/*.sh`, `outputs/logs/run_*.sh`, `configs/*.yaml`, `README.md`, `AGENTS.md`,
`report/plano-de-execucao.md` e as seções de `report/relatorio/secoes/`.

**Limitações da cobertura:**

- A auditoria é por leitura, com reprodução nos dados que existem nesta máquina (ensaio mini e
  uma variante local da amostra). Os artefatos da execução principal não estão aqui; o que
  depende deles está marcado.
- `plots.py` e `report/sections/` foram lidos por inteiro, mas conferidos com menos
  profundidade que os módulos numéricos: são formatação de tabelas e figuras.
- `scripts/feasibility` foi lido de forma mais leve; os seus números são tratados pelo próprio
  relatório como aproximados.
- `report/entrega-final/secoes/` só foi consultado por busca de termos, não lido por inteiro.
- Os textos em português que o código emite (logs, rótulos, notas de tabela) não foram
  revisados quanto à redação.
- Cada docstring foi escrita logo depois da leitura integral do módulo, e os nomes dos
  parâmetros foram conferidos por programa contra as assinaturas. O conteúdo descritivo (o que
  a função devolve, quando levanta exceção) não passou por uma segunda leitura completa do
  diff; só foram reconferidas as afirmações que dependiam de outro módulo (cerca de trinta).

## C. Problemas identificados

### Gravidade alta

#### C-01. A "fração de vértices com maior mudança" de P2 não desconta o ruído dos empates

1. **Local:** [`transitions_table` (analysis.py:292)](src/gender_networks/analysis.py#L292) (cálculo de `largest`, linha 326, e de
   `share_largest_change`, linha 353); consumida em [`figure_p2_transicoes` (p2.py:86)](src/gender_networks/report/sections/p2.py#L86) e
   descrita em `report/relatorio/secoes/08-resultados.tex` ("Fração de vértices cuja maior
   mudança acontece em cada transição").
2. **Gravidade:** alta. **Confiança:** confirmado (no ensaio mini).
3. **Descrição:** para cada vértice, a transição de "maior mudança" é a de menor Jaccard bruto
   entre as transições consecutivas (`np.argmin` sobre `J_lex_L01`, `J_L01_L18`, `J_L18_L36`). O
   Jaccard de `lex→L01` é a média sobre as 20 sementes do sorteio de empates da rede lexical e,
   por isso, já é baixo só por causa do sorteio. O piso de ruído (`NF_lex`) é descontado em
   `excess_change_mean`, mas não nessa fração.
4. **Por que é um problema:** o próprio método (`07-metodos.tex`, "Piso de ruído") diz que uma
   mudança lexical → contextual só é interpretada como reorganização se ficar acima do piso. Um
   tipo com `f = 50` e `k = 10` tem Jaccard esperado de cerca de 0,11 entre dois sorteios do
   mesmo conjunto lexical; qualquer rede contextual, mesmo uma que mantivesse só vizinhos do
   mesmo tipo, teria `J(lex, L01)` perto desse valor, e o vértice é contado como "maior mudança
   em `lex→L01`".
5. **Quando se manifesta:** sempre que o tipo do vértice tem empates na fronteira da rede lexical
   (`f_t − 1 > k`, ou seja, as faixas 10–49 e 50), que são justamente os alvos, controles e
   multitema.
6. **Consequências:** a figura de P2 superestima a parcela de vértices cuja maior reorganização
   ocorre na entrada do contexto (`lex→L01`) e subestima as outras transições, sobretudo nas
   faixas de frequência alta. A tabela `p2_transicoes` (mudança acima do piso) não é afetada.
7. **Evidência:** script somente leitura sobre `outputs/experiment/mini` (reproduzido em
   F.3). Faixa 50: `NF_lex` = 0,118 e `J(lex, L01)` = 0,119 (mudança nula acima do piso), e
   41% dos vértices são contados em `lex→L01`; descontando o piso, 0%. Faixa 10–49: 12% → 1%.
   Todos os vértices: `lex→L01` 12,2% → 2,5%; `L01→L18` 56,6% → 64,0%; `L18→L36` 31,2% → 33,5%.
   Na execução principal a magnitude não pôde ser medida (os artefatos não estão nesta máquina).
8. **Sugestão:** calcular a transição de maior mudança sobre a mudança acima do piso
   (`M − M^ruído`, isto é, `NF_lex − J` em `lex→L01` e `1 − J` nas demais), ou reportar as duas
   versões lado a lado; registrar a escolha em `07-metodos.tex`.

### Gravidade média

#### C-02. Texto do relatório diz que nenhuma cota ficou incompleta; a tabela gerada mostra o contrário

1. **Local:** `report/relatorio/secoes/04-dados.tex` (parágrafo "2. Palavras-alvo") contra
   `report/relatorio/tabelas/amostra_cotas.tex` (gerada); regra em
   [`_Selector.select_quota_stratum` (sampling.py:766)](src/gender_networks/sampling.py#L766).
2. **Gravidade:** média. **Confiança:** confirmado.
3. **Descrição:** o texto diz que as contagens por estrato "ficam abaixo da soma das cotas, sem
   que falte ocorrência". A tabela da execução principal mostra cotas não preenchidas em 14 pares
   (palavra, tema).
4. **Por que é um problema:** a P4 compara alvos e controles supondo "a mesma alocação"; o leitor
   é levado a crer que as cotas foram atingidas. Além disso, a regra de permanência
   (`min_per_sense` em pelo menos 2 temas) mantém palavras de três temas com o terceiro tema
   abaixo do mínimo.
5. **Quando se manifesta:** na execução principal de `configs/experiment.yaml`.
6. **Consequências:** desbalanceamento entre temas de sentido maior do que o texto indica;
   *banco*/Geografia e *órgão*/Política ficam com 7 ocorrências (abaixo do mínimo de 10) e
   continuam como temas de sentido dessas palavras.
7. **Evidência:** `amostra_cotas.tex`: banco/Geografia 7 de 16; órgão/Política 7 de 16;
   família/Política 11 de 25; capital/Política 17 de 25; campo/Geografia 14 de 16;
   órgão/Música 15 de 17; nota/Economia 22 de 25; estado/Política 23 de 25; ano/Música 14 de 16;
   século/Política 14 de 16; grande/Biologia 21 de 25; primeiro/Computação 22 de 25;
   importante/Economia 22 de 25; água/Geografia 24 de 25. *planta* foi descartada (Economia 5).
8. **Sugestão:** corrigir a frase de `04-dados.tex` (as contagens ficam abaixo da soma das cotas
   por dois motivos: ocorrências que já estavam no núcleo e cotas não preenchidas) e citar a
   tabela; avaliar exigir `min_per_sense` em todos os temas de cota, ou tirar da análise por tema
   os temas abaixo do mínimo.

#### C-03. O limite de 20 iterações do Leiden não aparece em nenhum relatório

1. **Local:** [`leiden` (partitions.py:329)](src/gender_networks/partitions.py#L329), campo `leiden_iterations` em
   [`AnalysisSettings` (settings.py:255)](src/gender_networks/settings.py#L255) e `configs/experiment.yaml`
   (`analysis.leiden_iterations: 20`).
2. **Gravidade:** média. **Confiança:** confirmado.
3. **Descrição:** cada execução do Leiden para depois de 20 iterações; uma iteração extra só
   verifica se a partição ainda mudaria, e o resultado vai para `converged_fraction` em
   `metrics/communities/index.csv`. `07-metodos.tex` diz apenas que o Leiden "roda com 10
   sementes; fica a partição de maior modularidade".
4. **Por que é um problema:** uma execução interrompida no limite não é um ótimo local da
   modularidade. Sem o limite e sem a fração convergida, o leitor não consegue avaliar as
   comunidades de P3 e P4 nem reproduzi-las com outra implementação.
5. **Quando se manifesta:** em grafos em que alguma execução não converge em 20 iterações (o
   comentário de `settings.py` diz que execuções sem limite não terminavam na rede do
   vocabulário).
6. **Consequências:** possível subotimização das partições (modularidade e NMI um pouco menores);
   o efeito é desconhecido porque a fração convergida não é reportada.
7. **Evidência:** busca por "itera" e "converg" em `report/relatorio/secoes/*.tex` e
   `report/entrega-final/secoes/*.tex` não retorna nada.
8. **Sugestão:** descrever o limite em `07-metodos.tex` e reportar `converged_fraction` por rede
   (a etapa `metrics` já grava o valor).

#### C-04. Uma etapa refeita com `--force` e interrompida pode ser dada como atualizada

1. **Local:** [`run` (sampling.py:1613)](src/gender_networks/sampling.py#L1613), [`run` (knn.py:1309)](src/gender_networks/knn.py#L1309),
   [`run` (metrics.py:1062)](src/gender_networks/metrics.py#L1062) e [`run` (vocab_lens.py:891)](src/gender_networks/vocab_lens.py#L891); comparar
   com [`run` (extract.py:1653)](src/gender_networks/extract.py#L1653) e [`run` (analysis.py:1246)](src/gender_networks/analysis.py#L1246), que
   removem o manifesto antes de trabalhar.
2. **Gravidade:** média. **Confiança:** confirmado (leitura).
3. **Descrição:** `extract` e `analyze` apagam o `_manifest.json` antigo antes de regravar as
   saídas; as outras etapas não. Se um `--force` morre no meio, ficam o manifesto antigo e uma
   mistura de arquivos novos e antigos.
4. **Por que é um problema:** `artifacts.stage_is_fresh` só compara a configuração e as entradas;
   com as duas iguais, a próxima execução normal pula a etapa. O `--force` é justamente o que o
   `AGENTS.md` manda usar depois de mudar o código.
5. **Quando se manifesta:** `--force` sem mudança de configuração nem de entradas (por exemplo,
   depois de uma correção no código), interrompido depois de regravar parte das saídas.
6. **Consequências:** artefatos de duas versões do código convivem em silêncio e passam para as
   etapas seguintes; o manifesto registra o commit antigo.
7. **Evidência:** `grep "unlink(missing_ok"` só encontra a remoção do manifesto em `extract.py`
   e `analysis.py`; o comentário de `extract.run` descreve exatamente esse risco.
8. **Sugestão:** remover o manifesto no início do trabalho em todas as etapas (ou dentro de
   `artifacts.up_to_date`, quando ela devolve falso).

#### C-05. O manifesto de `sample` registra uma seção `corpus` que pode não ter produzido o corpus

1. **Local:** [`run` (corpus.py:411)](src/gender_networks/corpus.py#L411) (teste de existência),
   [`run` (sampling.py:1613)](src/gender_networks/sampling.py#L1613) e [`settings_digest` (artifacts.py:1029)](src/gender_networks/artifacts.py#L1029).
2. **Gravidade:** média. **Confiança:** confirmado (leitura).
3. **Descrição:** a etapa `corpus` é pulada sempre que `paragraphs.jsonl` existe, sem olhar a
   configuração. As etapas seguintes incluem a seção `corpus` no seu digest e gravam a
   configuração atual inteira no manifesto, sem compará-la com `data/corpus/_manifest.json`.
4. **Por que é um problema:** mudar `corpus:` no YAML (por exemplo, `cut_sections` ou os filtros
   de parágrafo) não refaz o corpus, mas faz `sample` e as etapas seguintes rodarem de novo
   sobre o corpus antigo e registrarem os parâmetros novos.
5. **Quando se manifesta:** qualquer edição da seção `corpus` sem `--force` na etapa `corpus`
   (ou sem `all --force-corpus`).
6. **Consequências:** proveniência errada nos manifestos; quem reproduzir a partir deles obtém
   outro corpus.
7. **Evidência:** `corpus.run` começa com `if paths.paragraphs.exists() and not force: return`;
   `STAGE_SETTINGS["sample"] = ("corpus", "sample")`.
8. **Sugestão:** fazer `sample` comparar o digest da seção `corpus` com o registrado em
   `data/corpus/_manifest.json` e parar com erro quando diferirem; ou usar `up_to_date` também
   na etapa `corpus`.

### Gravidade baixa

Cada item traz local, problema, condição, consequência, evidência e sugestão.

**Configuração e artefatos**

- **C-06.** [`_read_yaml` (settings.py:495)](src/gender_networks/settings.py#L495). A herança `base:` mescla só um nível:
  uma subseção aninhada dada no filho (`sample.core`, `sample.multitheme`, `sample.limits`,
  `corpus.paragraph`, `networks.vocab`) substitui a do pai inteira, e as chaves omitidas voltam
  ao padrão do dataclass, não ao valor do pai. *Condição:* o pai define um valor diferente do
  padrão numa dessas subseções. *Consequência:* a configuração efetiva difere da esperada, sem
  aviso; hoje é inofensivo porque `experiment.yaml` usa os padrões nessas subseções.
  *Evidência:* `merged[key] = {**base[key], **value}`. *Sugestão:* mescla recursiva, ou erro
  quando o filho dá uma subseção parcial. Confirmado.
- **C-07.** [`_read_yaml` (settings.py:495)](src/gender_networks/settings.py#L495). Não detecta ciclo em `base:`
  (`RecursionError`). Confirmado.
- **C-08.** [`_build` (settings.py:373)](src/gender_networks/settings.py#L373) e [`_validate` (settings.py:409)](src/gender_networks/settings.py#L409).
  Nenhum tipo é conferido e vários campos não são validados (`eps`, `bands` só em `sampling`,
  `type_candidates` e `networks.vocab.k_values` só em `knn.check_settings`; `k_values` vazio
  dá o erro genérico de `max()`). O PyYAML lê `1e-6` (sem ponto) como texto; o arquivo usa
  `1.0e-6`, que é lido como número. *Consequência:* erro tardio, longe da configuração.
  *Sugestão:* validar tipos e faixas em `load_settings`. Confirmado.
- **C-09.** [`_validate` (settings.py:409)](src/gender_networks/settings.py#L409) exige `candidates > max(k)`;
  [`check_settings` (knn.py:1280)](src/gender_networks/knn.py#L1280) exige `>=`. Duas regras para a mesma condição.
  Confirmado.
- **C-10.** [`_validate` (settings.py:409)](src/gender_networks/settings.py#L409). As cotas de alvos e controles são
  validadas contra `corpus.themes`, não contra `sample.themes`; `sample.themes: []` vale como
  "todos". *Consequência:* um tema de cota fora da amostra só gera aviso na etapa `sample`.
  Confirmado.
- **C-11.** [representations.py:36](src/gender_networks/representations.py#L36) (`FINAL_NORM_BLOCK = 36`) e
  [`_known_representation` (settings.py:476)](src/gender_networks/settings.py#L476). A saída normalizada só tem nome
  para 36 blocos (`L36n`), e [`output_paths` (extract.py:112)](src/gender_networks/extract.py#L112) grava sempre
  `L36n.safetensors`. *Condição:* outro modelo. *Sugestão:* derivar o bloco de `len(layers)`.
  Confirmado.
- **C-12.** [`library_versions` (artifacts.py:954)](src/gender_networks/artifacts.py#L954) importa `torch` e
  `transformers` em toda etapa (via `write_manifest`), inclusive `corpus`; o comentário de
  `cli.STAGES` diz que a etapa `corpus` não importa o torch. Só custa tempo. Confirmado.
- **C-13.** [`stage_is_fresh` (artifacts.py:1051)](src/gender_networks/artifacts.py#L1051). O teste de atualização não
  considera a versão do código, e a assinatura das entradas é tamanho e data de modificação sob
  o caminho como escrito (muda com a grafia de `--root`). *Consequência:* depois de mudar o
  código é preciso `--force` (o `AGENTS.md` avisa); a outra direção é segura (refaz).
  *Sugestão:* gravar e comparar o commit, ou um hash dos módulos da etapa. Confirmado.
- **C-14.** [`main` (cli.py:144)](src/gender_networks/cli.py#L144) e `PIPELINE`. O comando `all` não roda `lens`; o
  `AGENTS.md` descreve o pipeline com `lens` antes de `report`. *Consequência:* `all` gera o
  relatório com os itens da lente pendentes, ou com saídas antigas da lente. Confirmado.
- **C-15.** [`run` (metrics.py:1062)](src/gender_networks/metrics.py#L1062). Só `graph_metrics.csv` entra no teste de
  atualização; apagar `hubs.csv` ou uma partição não refaz a etapa.
  [`_save_partition` (metrics.py:913)](src/gender_networks/metrics.py#L913) não grava de forma atômica (o
  `knn._save_npz` grava). Confirmado.
- **C-16.** [`read_manifest` (corpus_rebuild.py:47)](src/gender_networks/corpus_rebuild.py#L47) aceita só o texto `True` em
  `kept`; `true` ou `1` viram "não mantido" em silêncio (`artifacts.parse_bool` aceita mais
  grafias). Só importa se o CSV for editado à mão. Confirmado.
- **C-17.** [`run` (corpus_rebuild.py:137)](src/gender_networks/corpus_rebuild.py#L137). `relative_to(paths.root)` levanta
  `ValueError` quando `corpus_dir` é absoluto e fora da raiz, depois de gravar `articles.jsonl`
  e `paragraphs.jsonl` e antes do manifesto. Confirmado.

**Corpus e limpeza**

- **C-18.** [textclean.py:11](src/gender_networks/textclean.py#L11) (`_HEADING`). O `\s*` do padrão aceita
  quebra de linha: uma linha só com `==` seguida de uma linha terminada em `==` casa como um
  título de duas linhas. *Evidência:* reproduzido com o padrão isolado (`'==\nfoo =='`). Raro em
  wikitext real. *Sugestão:* `[ \t]*` no lugar de `\s*`. Confirmado.
- **C-19.** [`template_names` (textclean.py:51)](src/gender_networks/textclean.py#L51) é uma expressão regular sobre o
  texto cru: predefinições dentro de comentários HTML ou de `nowiki` contam. *Consequência:* uma
  infobox comentada marca o artigo como biografia. *Sugestão:* usar a árvore do
  `mwparserfromhell`. Confirmado.
- **C-20.** [`wikitext_to_text` (textclean.py:265)](src/gender_networks/textclean.py#L265) e
  [`split_paragraphs` (textclean.py:290)](src/gender_networks/textclean.py#L290). Um parágrafo é uma linha do texto limpo;
  um parágrafo quebrado em várias linhas no wikitext vira vários candidatos, quase todos
  reprovados pelo filtro de 40 palavras e fim de frase. O relatório não menciona. Confirmado.
- **C-21.** [`_ends_with_abbreviation` (textclean.py:327)](src/gender_networks/textclean.py#L327). Uma maiúscula isolada
  ("no grupo A. Depois") ou uma abreviação em fim real de frase ("etc. Depois") impede o corte
  de sentença. *Consequência:* o rótulo de sentença da P3 fica um pouco mais grosso; não medido.
  Confirmado.
- **C-22.** [`build_corpus` (corpus.py:183)](src/gender_networks/corpus.py#L183). Os títulos do último lote que vêm
  depois de a cota fechar são baixados e não ganham registro em `articles.jsonl`; não entram em
  "artigos considerados". Confirmado.
- **C-23.** [`_retry_after` (wiki.py:579)](src/gender_networks/wiki.py#L579) só lê `Retry-After` e `retry-after`;
  outra capitalização cai no recuo exponencial (≥ 30 s), que é seguro. Confirmado.

**Amostra**

- **C-24.** [`_Selector.select_quota_stratum` (sampling.py:766)](src/gender_networks/sampling.py#L766). `need =
  min(cota − núcleo_no_tema, f_max − total)`, e `total` conta toda ocorrência do tipo, inclusive
  as do núcleo em temas sem cota. Como as cotas somam `f_max` (25 + 25, 17 + 17 + 16), cada
  ocorrência do alvo no núcleo fora dos temas de cota tira uma vaga do último tema da lista.
  *Consequência:* o déficit depende da ordem dos temas no YAML. *Evidência:* em
  `outputs/experiment/main-local/sample` (variante local com `root_only`, mesmo código): estado
  8 + 17 em Física, 24 em Política e 1 em Biologia (núcleo) = 50; massa 25 + 24 + 1; primeiro
  25 + 24 + 1; importante 25 + 22 + 3; campo 17 + 17 + 13 + 3; ano 17 + 17 + 14 + 2. *Sugestão:*
  reservar as vagas dos temas de cota antes de aplicar `f_max`, ou somar as cotas abaixo de
  `f_max`. Confirmado (na variante local; a execução principal mostra o mesmo padrão em
  estado, água e primeiro).
- **C-25.** [`build_sample` (sampling.py:1372)](src/gender_networks/sampling.py#L1372) (`word_of_id`). Um alvo descartado
  continua com `target_word` e `sense_theme` nas suas ocorrências do núcleo, e
  [`type_groups` (analysis.py:801)](src/gender_networks/analysis.py#L801) o classifica como "target". *Condição:* o alvo
  descartado ter `f ≥ 10` só com o núcleo. *Evidência:* em `main-local`, *planta* tem 2 linhas,
  ambas com `target_word = planta`; na principal tem 1 e é excluída pelo corte `f ≥ 10`.
  *Sugestão:* montar `word_of_id` só com as palavras mantidas. Confirmado (latente).
- **C-26.** [`assemble_sample` (sampling.py:1102)](src/gender_networks/sampling.py#L1102). `sense_theme` recebe o tema do
  parágrafo em qualquer tema, também fora dos temas de cota da palavra. Confirmado.
- **C-27.** [`_Selector.theme_counts` (sampling.py:842)](src/gender_networks/sampling.py#L842). A contagem "sorteável"
  não desconta as ocorrências do núcleo no mesmo artigo (o limite por artigo do sorteio conta).
  Só afeta a elegibilidade do multitema. Confirmado.
- **C-28.** [`_find_words` (tokens.py:263)](src/gender_networks/tokens.py#L263) usa `\w+`: marcas combinantes (texto em
  NFD) partem a palavra. A Wikipédia é quase toda NFC. Confirmado.

**Extração**

- **C-29.** [`predict_next` (extract.py:545)](src/gender_networks/extract.py#L545). A previsão usa a matriz de entrada
  como matriz de saída e nada confere `tie_word_embeddings` (o `verify` só registra o valor).
  *Condição:* modelo sem amarração. Os logits são calculados em float32 a partir de linhas
  bf16, então em quase empates o argmax pode diferir do que o modelo daria em bf16.
  *Sugestão:* tornar `tie_word_embeddings` uma checagem bloqueante. Confirmado.
- **C-30.** `report/relatorio/secoes/05-modelo.tex` diz que o modelo é carregado com atenção
  `sdpa`; [`load_model` (extract.py:163)](src/gender_networks/extract.py#L163) não passa `attn_implementation` (vale o
  padrão da biblioteca; `verify.json` registra o valor usado). Confirmado.
- **C-31.** Duas checagens diferentes do token de prefixo:
  [`resolve_prefix_id` (sampling.py:1559)](src/gender_networks/sampling.py#L1559) (ida e volta) e
  [`run` (extract.py:1653)](src/gender_networks/extract.py#L1653) (`isinstance int` e diferente de `unk_token_id`).
  Confirmado.
- **C-32.** [`verify` (extract.py:1011)](src/gender_networks/extract.py#L1011). `model.verify_sequences: 0` levanta
  `ValueError` e bloqueia a extração. [`representation_summary` (extract.py:1228)](src/gender_networks/extract.py#L1228)
  só conta linhas não finitas; um NaN contamina quantis e o JSON sai com `NaN`. Confirmado.

**k-NN, vizinhança, grafos e comunidades**

- **C-33.** [`noise_floor` (neighborhood.py:393)](src/gender_networks/neighborhood.py#L393) devolve o Jaccard médio entre
  sementes, enquanto o relatório chama de "piso de ruído" a mudança `1 − J`. O uso em
  `analysis.transitions_table` é coerente (`piso − J`), mas o nome induz a erro. Confirmado.
- **C-34.** [`dominance` (neighborhood.py:227)](src/gender_networks/neighborhood.py#L227) divide pelo número de entradas da
  linha; [`jaccard` (neighborhood.py:158)](src/gender_networks/neighborhood.py#L158) trata linhas como conjuntos. Só
  divergem com ids repetidos numa linha, que o seletor não produz. Confirmado.
- **C-35.** [`key_jaccard` (graphs.py:542)](src/gender_networks/graphs.py#L542): `ZeroDivisionError` com dois conjuntos
  vazios. [`same_label_share` (graphs.py:560)](src/gender_networks/graphs.py#L560): NaN com aviso do numpy em grafo
  sem arestas. [`directed_clustering` (graphs.py:592)](src/gender_networks/graphs.py#L592): laço Python de custo
  proporcional à soma dos quadrados dos graus (só usado nos scripts da entrega parcial).
  Confirmado.
- **C-36.** [metrics.py:71](src/gender_networks/metrics.py#L71): `TIES` inclui `pos_eps`, que
  `plan_graphs` nunca planeja. Confirmado.

**Análises**

- **C-37.** [`core_metrics` (analysis.py:658)](src/gender_networks/analysis.py#L658) grava `core_metrics.csv` com as
  métricas do subgrafo induzido no núcleo. `09-robustez.tex` e `06-redes.tex` dizem que a rede
  só do núcleo não é o subgrafo induzido ("Planejado, ainda sem etapa"). O relatório não lê a
  tabela. *Consequência:* um artefato que contradiz o texto, se alguém o usar. *Sugestão:*
  remover a tabela ou renomeá-la para deixar claro o que mede. Confirmado.
- **C-38.** [`layer_sweep` (analysis.py:383)](src/gender_networks/analysis.py#L383) usa a regra de posição e compara o
  bloco 1 com os conjuntos lexicais por posição; as medidas principais usam o sorteio. Nos
  blocos contextuais os empates são raros, mas o primeiro ponto da curva `J(ℓ−1, ℓ)` é contra a
  rede lexical por posição, a que o relatório descreve como fonte de hubs artificiais. O rótulo
  do eixo da figura avisa ("em ℓ = 1, comparação com a rede lex por posição"); o texto de
  `07-metodos.tex` não diz qual regra a curva usa. Confirmado.
- **C-39.** [`vertex_measures` (analysis.py:192)](src/gender_networks/analysis.py#L192) reatribui o nome `pairs` (o
  parâmetro que está sendo percorrido) dentro do próprio laço. Funciona porque o `for` guarda o
  iterador da lista original; é frágil. Confirmado.
- **C-40.** [`robustness_neighbors` (analysis.py:542)](src/gender_networks/analysis.py#L542). `J_vs_lex_mean` com a regra
  `rand` usa só a semente 0 dos dois lados e, portanto, inclui o ruído do sorteio (as medidas
  principais fazem a média sobre as sementes). Confirmado.
- **C-41.** [`run` (vocab_lens.py:891)](src/gender_networks/vocab_lens.py#L891). As similaridades são float32 no
  dispositivo escolhido (GPU quando existe); postos em quase empates podem diferir entre CPU e
  GPU. O manifesto registra o dispositivo. Confirmado.

**Relatório gerado**

- **C-42.** [`table_metricas_globais` (metrics.py:84)](src/gender_networks/report/sections/metrics.py#L84). A nota da
  tabela diz "média ± desvio entre as 20 sementes na rede lexical", mas só as sementes 0 e 1
  têm comunidades (`metrics.plan_communities`); as linhas "Modularidade" e "Comunidades" da
  coluna `lex` são média e desvio de 2 valores. *Evidência:* `metricas_globais.tex` traz
  "186,0 ± 1,4" comunidades, o desvio amostral de dois inteiros que diferem de 2.
  *Sugestão:* dizer na nota quantas sementes entram em cada linha. Confirmado.
- **C-43.** [`table_knn_empates` (knn.py:23)](src/gender_networks/report/sections/knn.py#L23). A coluna
  "Recalculadas" usa `fallback_rows`, contado na tolerância de completude
  `max(eps, eps_sensitivity)` = 10⁻⁴; a nota da tabela só cita ε = 10⁻⁶. O manifesto também
  tem `fallback_rows_eps`. *Sugestão:* usar `fallback_rows_eps` ou explicar na nota.
  Confirmado.
- **C-44.** [`run` (report/__init__.py:249)](src/gender_networks/report/__init__.py#L249). A etapa `report` lê o que houver em
  disco e não confere se as etapas anteriores estão atualizadas entre si (por exemplo,
  `analysis/` mais antiga que `metrics/`). A tabela `versoes` e a macro `\revisaoGit` avisam
  quando as etapas vêm de revisões diferentes, mas não quando vêm de entradas diferentes.
  *Sugestão:* comparar a assinatura de entradas gravada em cada manifesto com o manifesto da
  etapa anterior e avisar. Confirmado.
- **C-45.** [`transition_share_figure` (plots.py:1312)](src/gender_networks/plots.py#L1312) calcula `rows` e o apaga
  com `del rows` sem usar (código morto). Confirmado.

**Scripts**

- **C-46.** [`compare_pdf` (compare_runs.py:158)](scripts/dev/compare_runs.py#L158) compara PDFs rasterizados a 50 dpi
  (`RASTER_DPI`): uma mudança pequena numa figura (um dígito num rótulo de 7 pt) pode alterar
  poucos pixels ou nenhum. [`without_timings` (compare_runs.py:266)](scripts/dev/compare_runs.py#L266) ignora toda chave
  JSON terminada em `_s`, não só as de tempo. [`compare_npy` (compare_runs.py:116)](scripts/dev/compare_runs.py#L116)
  carrega os dois arquivos inteiros (com `--include-reps`, `all_layers.npy` tem 2,8 GB). Essa é
  a ferramenta que o `AGENTS.md` manda usar para conferir que uma mudança não alterou
  resultados, e ela não tem teste. Confirmado.
- **C-47.** [`coarse_blocks` (draw_networks.py:319)](scripts/partial_delivery/draw_networks.py#L319). A docstring do módulo
  diz que os blocos usam o Leiden "as in the `metrics` stage", mas aqui as execuções não têm
  limite de iterações e usam a semente 0; a etapa `metrics` usa limite de 20 e a semente
  `base_seed`. Confirmado.
- **C-48.** [`check_against_pipeline` (build_crawl_network.py:206)](scripts/crawl_network/build_crawl_network.py#L206) recebe
  três parâmetros que não usa (`client`, `settings`, `walk_list`).
  [`sample_payload` (build_crawl_network.py:561)](scripts/crawl_network/build_crawl_network.py#L561) repete a regra
  `articles_per_extra_category * 4` de `corpus.walks` em vez de lê-la dos `Walk`. O script não
  tem teste. Confirmado.
- **C-49.** Três convenções para ler booleanos de CSV: `artifacts.parse_bool`,
  `row["kept"] == "True"` em `corpus_rebuild.read_manifest` (C-16) e
  `.str.lower().eq("true")` em [`vertex_table` (export_networks.py:142)](scripts/partial_delivery/export_networks.py#L142).
  Confirmado.

## D. Inconsistências científicas

Divergências entre código, metodologia descrita e resultados, em ordem de impacto.

1. **Fração de maior mudança por transição (P2)** não segue a regra do piso de ruído que o
   método enuncia. Ver C-01. É o único ponto encontrado em que um resultado reportado muda de
   forma relevante com a correção (no ensaio mini, `lex→L01` cai de 12,2% para 2,5%).
2. **Cotas das palavras-alvo.** O texto afirma que nenhuma ocorrência faltou; a tabela gerada
   mostra 14 cotas incompletas e dois temas de sentido com 7 ocorrências. Ver C-02 e C-24.
3. **Leiden com limite de iterações.** O método descrito não menciona o limite de 20 iterações
   nem a fração de execuções convergidas. Ver C-03.
4. **Curva camada a camada (P2).** A curva usa a regra de posição e parte da rede lexical por
   posição; o texto não diz isso. Ver C-38.
5. **Rede só do núcleo.** O texto diz que ainda não existe; o código grava uma tabela
   (`core_metrics.csv`) do subgrafo induzido, que o texto diz não servir. Ver C-37.
6. **Multitema.** O texto diz "pelo menos 5 ocorrências em pelo menos 3 temas fora do núcleo"; o
   código conta só posições de palavra inteira e depois dos limites por parágrafo e por artigo
   (contagem "sorteável"). O manifesto de `sample` registra as duas contagens
   (`multitheme_eligible` e `multitheme_eligible_raw_counts`). Diferença de redação, não de
   resultado.
7. **Piso de ruído.** O texto define o piso como mudança `M = 1 − J`; o código guarda o Jaccard
   (`NF_lex`) e subtrai na hora de reportar. Os valores reportados são coerentes. Ver C-33.
8. **Atenção `sdpa`.** O texto afirma; o código deixa o padrão da biblioteca. Ver C-30.
9. **Tabela de métricas globais.** A nota diz 20 sementes na rede lexical; modularidade e número
   de comunidades vêm de 2. Ver C-42.
10. **Linhas recalculadas (tabela de empates).** A contagem é feita com tolerância 10⁻⁴; a nota
    cita 10⁻⁶. Ver C-43.

**Validade.** Os pontos 1 e 2 mudam a leitura de resultados (P2 e P4). O ponto 3 afeta a
descrição do método de P3 e P4, não necessariamente os números. Os demais são de redação ou de
artefatos que o relatório não usa.

**Reprodutibilidade.** As sementes vêm todas do YAML e os geradores não dependem da ordem de
execução (conferido em `knn.rep_rng`, `metrics.build_plan`, `partitions.leiden`,
`sampling.build_sample`). Os riscos encontrados estão fora do cálculo: o teste de atualização
das etapas ignora a versão do código (C-13), o `--force` interrompido (C-04), a proveniência do
corpus (C-05) e a lente em float32 dependente do dispositivo (C-41). A extração em bf16
depende do hardware, o que o relatório já registra nas limitações.

O que foi conferido e está de acordo com o relatório: cosseno em float64 a partir dos vetores
bf16; listas de candidatos completas para `k_max` com recálculo da linha; definição de `F`, `B` e
`r` medida a partir de `s_k`; sorteio uniforme sem reposição com o gerador
`default_rng([base_seed, crc32(nome), k, r])`; regra de posição; variante (b); variante (c) com o
próprio tipo excluído; derivação exata da rede lexical; união e mútua; métricas estruturais
(densidade, grau médio, clusterização local só em grau ≥ 2, reciprocidade, assimetria); 20
sementes só na rede lexical; NMI, ARI, pureza e linha de base por permutação; saídas brutas por
hooks e as cinco verificações do `--verify`; cópia por grupo de prefixo; limite `f_max` por
sorteio depois de juntar o núcleo.

## E. Melhorias recomendadas

Nada disto foi implementado.

**Resultados e texto (antes da entrega)**

1. Refazer a fração de maior mudança de P2 sobre a mudança acima do piso de ruído, ou mostrar as
   duas versões (C-01). É a única recomendação que altera um resultado.
2. Corrigir a frase sobre as cotas em `04-dados.tex` e decidir o que fazer com os temas de
   sentido abaixo do mínimo (C-02).
3. Descrever o limite de iterações do Leiden e reportar a fração convergida (C-03).
4. Ajustar as notas das tabelas `metricas_globais` e `knn_empates` (C-42, C-43) e dizer no texto
   que a curva camada a camada usa a regra de posição (C-38).

**Testes**

5. Um teste de `transitions_table` com tipos de frequência alta em que a rede contextual é
   igual à lexical a menos do sorteio: a fração de maior mudança em `lex→L01` deve ser zero. O
   teste atual só confere que as frações somam 1.
6. Estender às etapas `sample`, `knn`, `metrics` e `lens` o teste que já existe para `extract`
   (`failed_forced_run_is_redone_by_the_next_run`).
7. Testes para `scripts/dev/compare_runs.py`: um caso por tipo de arquivo, com uma diferença
   plantada que tem de ser detectada, inclusive um PDF com um dígito trocado.
8. Um teste que compare as cotas do manifesto de `sample` com o que o texto do relatório afirma,
   no espírito dos testes de `test_relatorio.py`.
9. Testes de borda hoje ausentes: herança `base:` com subseção aninhada parcial, `k_values`
   vazio, `key_jaccard` com conjuntos vazios, `verify_sequences: 0`.

**Organização e manutenção**

10. Uma só função para ler booleanos de CSV (`parse_bool`) e uma só checagem do token de
    prefixo (C-49, C-31).
11. Remover o manifesto no início de toda etapa, de preferência dentro de `up_to_date` (C-04), e
    gravar no manifesto um hash dos módulos da etapa para que mudanças de código a tornem
    desatualizada (C-13).
12. Validar tipos e faixas em `load_settings` e fazer a mescla de `base:` recursiva, ou recusar
    subseções parciais (C-06, C-08).
13. Fazer `sample` conferir que o corpus em disco foi produzido com a seção `corpus` atual
    (C-05).
14. Derivar o bloco da norma final do número de blocos do modelo, em vez da constante 36 (C-11).
15. Remover ou renomear `core_metrics.csv`, `pos_eps` e a variável morta de
    `transition_share_figure` (C-37, C-36, C-45).
16. Incluir `lens` no comando `all`, ou tirar a etapa da descrição do pipeline no `AGENTS.md`
    (C-14).

**Desempenho**

17. `graphs.directed_clustering` é um laço Python sobre pares de vizinhos; uma versão com matriz
    esparsa (`S @ S` restrito às arestas) resolveria redes maiores. Hoje só é usado nas redes de
    13,6 mil vértices da entrega parcial.
18. `artifacts.library_versions` importa `torch` e `transformers` em toda etapa; ler as versões
    com `importlib.metadata.version` evita o custo (C-12).
19. `analysis.community_membership` relê `communities/index.csv` a cada chamada; o custo é
    pequeno, mas um cache simples o elimina.

## F. Garantia de preservação

### F.1 Arquivos modificados

43 arquivos Python do escopo, só em docstrings e comentários, e este `REVISAO_TECNICA.md`
(novo). `git status --short` mostra 43 linhas ` M` (todas em `src/gender_networks/` ou em
`scripts/{crawl_network,dev,partial_delivery}/`) e uma `??` (este arquivo). `git diff --stat`
restrito a `tests`, `configs`, `data`, `docs`, `report`, `outputs`, `pyproject.toml`, `uv.lock`,
`README.md` e `AGENTS.md` não devolve nada. Total: 7.547 linhas inseridas e 313 removidas (as
removidas são docstrings antigas, substituídas por versões ampliadas).

Arquivos alterados: em `src/gender_networks/`, `analysis.py`, `artifacts.py`, `cli.py`,
`corpus.py`, `corpus_rebuild.py`, `extract.py`, `graphs.py`, `knn.py`, `labels.py`, `metrics.py`,
`neighborhood.py`, `partitions.py`, `plots.py`, `representations.py`, `sampling.py`,
`settings.py`, `textclean.py`, `tokens.py`, `vocab_lens.py`, `wiki.py`; em `report/`,
`__init__.py`, `context.py`, `latex.py`, `numbers.py` e os 14 módulos de `sections/` com
funções; e os cinco scripts `scripts/dev/compare_runs.py`,
`scripts/crawl_network/build_crawl_network.py` e
`scripts/partial_delivery/{export_networks,draw_networks,draw_presentation}.py`.

### F.2 Verificações de que só documentação mudou

Todas comparam a árvore de trabalho com `git show HEAD:<arquivo>`, nos 45 arquivos do escopo,
sem gravar nada no repositório. Os scripts de verificação ficaram num diretório temporário fora
do repositório.

| # | Verificação | Resultado |
|---|---|---|
| 1 | **AST:** `ast.dump` das duas versões depois de remover as docstrings de módulo, classe e função | idênticas nos 45 arquivos |
| 2 | **Tokens** (independente do módulo `ast`): sequência de tokens do `tokenize` sem comentários e sem as linhas lógicas formadas só por strings; `INDENT` e `DEDENT` mantidos | 127.855 tokens de código comparados, 0 arquivos diferentes |
| 3 | **Bytecode** no modo em que o projeto roda (`optimize=0`): instruções de cada objeto de código, pondo de lado só a constante da docstring e o par `LOAD_CONST; STORE_NAME __doc__` | 991 objetos de código comparados, 0 arquivos diferentes |
| 4 | **Comentários de ferramenta** (`# noqa`, `# fmt:`, `# type:`): mesmo conjunto por arquivo | iguais |
| 5 | **Textos de que o comportamento depende:** `"1990s"` na docstring de `tokens.py` (exigido por `tests/test_tokens.py`) e a primeira linha da docstring dos cinco scripts (texto de ajuda do argparse) | preservados |
| 6 | **Cobertura:** toda função e classe do escopo com docstring; nomes da seção `Args` iguais, na ordem, aos parâmetros da assinatura | 0 problemas |
| 7 | `uv run ruff check .` | "All checks passed!" antes e depois |
| 8 | `uv run pytest` | 303 testes passam antes e depois |

Sobre a verificação 8: a linha de base foi medida duas vezes. Antes de qualquer edição, na
árvore de trabalho ainda limpa (303 passaram). No fim, de novo, numa cópia limpa do commit
extraída com `git archive HEAD` para um diretório temporário e executada com o mesmo ambiente
virtual e `PYTHONPATH` apontando para a cópia (conferido que o pacote era importado de lá): 303
passaram em 23,0 s. Na árvore de trabalho documentada: 303 passaram em 20,3 s.

A verificação 1 pode ser refeita com:

```python
import ast, subprocess, sys

def strip(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:]
    return ast.dump(tree)

for path in sys.argv[1:]:
    old = subprocess.run(["git", "show", f"5bdf56f:{path}"], capture_output=True, text=True).stdout
    same = strip(ast.parse(old)) == strip(ast.parse(open(path, encoding="utf-8").read()))
    print("igual   " if same else "DIFERENTE", path)
```

**Um alarme falso, registrado por transparência.** A primeira versão da verificação 3 compilava
com `optimize=2` (que descarta docstrings) e acusou diferença em 41 arquivos. A causa foi
isolada com um exemplo mínimo: sob `optimize=2`, o CPython 3.12 deixa de tratar a string como
docstring, compila-a como uma instrução de constante e mantém um `NOP` no lugar dela; uma
função que ganhou docstring passa a ter esse `NOP` a mais. No modo normal (`optimize=0`), que é
o que o projeto usa, as instruções são idênticas e só muda a constante da docstring. A
verificação foi refeita nesse modo (linha 3 da tabela). Consequência prática: quem rodar o
projeto com `python -OO` terá um `NOP` a mais em cada função documentada, sem efeito no
resultado.

**Um segundo alarme falso, da própria ferramenta.** Na rodada final, a verificação 3 passou a
acusar um arquivo (`wiki.py` ou `textclean.py`, conforme a execução), sem que nenhum código
tivesse mudado. Repetindo com `PYTHONHASHSEED` fixo, o resultado dependia da semente de hash.
A causa era a comparação pelo `repr()` de uma constante `frozenset` (os 11 códigos de
`PERMANENT_ERRORS`): as duas compilações produzem conjuntos iguais, mas a ordem de iteração de
um conjunto, e portanto o seu `repr()`, pode diferir. Com os elementos ordenados antes da
comparação, a verificação dá 0 arquivos diferentes em 12 sementes de hash (0 a 11). As
verificações 1 e 2 não usam `repr()` de conjuntos e não foram afetadas.

**Outro incidente.** O diretório temporário da sessão foi apagado entre os dois dias de
trabalho. As ferramentas de verificação foram reescritas e todas as verificações acima foram
executadas de novo sobre o estado final; os registros da primeira linha de base (saídas do
`pytest` e do `ruff` do dia 12) se perderam, e por isso a linha de base foi medida outra vez na
cópia limpa do commit.

### F.3 Reproduções usadas na auditoria (somente leitura)

- **C-01:** leitura de `outputs/experiment/mini/analysis/vertex_measures.csv` e de
  `outputs/experiment/mini/sample/occurrences.csv`; cálculo, por faixa de frequência, do
  `argmin` das três colunas `J_*` consecutivas com e sem o desconto `1 − (NF_lex − J_lex_L01)`.
- **C-24 e C-25:** leitura de `outputs/experiment/main-local/sample/_manifest.json` e
  `occurrences.csv` (variante local com `root_only`), contando as ocorrências de cada alvo e
  controle por tema.
- **C-02, C-42:** leitura das tabelas geradas `report/relatorio/tabelas/amostra_cotas.tex` e
  `metricas_globais.tex`.
- **C-18:** o padrão `_HEADING` aplicado a um texto de teste.
- **C-03:** `grep -i "itera\|converg"` nas seções dos dois relatórios.

### F.4 O que não foi feito

- **O ensaio mini não foi refeito.** O roteiro do `AGENTS.md` (`--force` nas etapas e
  `compare_runs.py`) sobrescreveria `outputs/experiment/mini`, o que o pedido desta revisão
  proíbe. A garantia de que os resultados não mudam vem das verificações 1 a 3, que mostram que
  o código executável é o mesmo, e não de uma comparação de saídas.
- Nenhuma etapa do pipeline foi executada, nem os scripts que gravam em `docs/`, `data/` ou
  `report/`. A etapa `report` não foi rodada (ela recompilaria o relatório).
- Nada foi commitado. Enquanto a documentação não for commitada, `git describe --dirty` devolve
  um sufixo `-dirty`, que iria para o `_manifest.json` de qualquer etapa executada e para a
  macro `\revisaoGit` do relatório.
