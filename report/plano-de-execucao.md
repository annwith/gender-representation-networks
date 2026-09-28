# Plano de execução: da rede lexical à rede contextual com Qwen3-4B-Base

## Contexto

A proposta revisada ([report.tex](report.tex)) define as perguntas P1–P4, três tipos de rede,
cerca de 1,5×10⁴ ocorrências da Wikipédia em português, k-NN por cosseno, três tratamentos de
empate, versões direcionada, por união e mútua, e comunidades comparadas por NMI/ARI. O
repositório tem hoje três pilotos independentes. Nenhum implementa corpus, amostragem, empates
com sementes, rede direcionada, rótulos de contexto ou escala de 15 mil vértices.

O estudo de viabilidade (arquivos em `scripts/feasibility/`) usou 958 artigos em 8 temas, 7.876
parágrafos, 1,28 milhão de tokens Qwen e cerca de 23 mil tipos. Ele mostrou três coisas: a meta
de vértices é fácil; parágrafos inteiros não deixam palavras repetidas suficientes para a P4; um
desenho híbrido resolve.

Decisões tomadas:
- modelo **Qwen3-4B-Base**;
- piloto de redes **congelado**;
- código contrafactual de gênero e notebook **removidos** (continuam no histórico do git);
- rede lexical de tipos sobre o **vocabulário inteiro** do modelo;
- extensão **"vizinhos no vocabulário"** como opcional;
- relatório técnico **detalhado, escrito junto com a implementação** (seção "Relatório
  técnico").

## Opinião sobre o piloto

O piloto cumpriu o papel. Ele mostrou os empates de cosseno 1, a quebra em subpalavras ("praca"
→ "pra" + "ca") e a questão da direcionalidade, que viraram as respostas à avaliação (Seção 11
da proposta).

Mas ele não escala:
- é monolítico e usa um texto fixo de 104 tokens;
- só desempata por posição, o que a 15 mil vértices cria hubs artificiais: todas as
  ocorrências de um tipo com f = 50 escolhem os mesmos vizinhos de menor índice;
- só constrói a rede por união;
- no NetworkX, as distâncias levam de 20 a 25 minutos por rede.

Ele fica congelado, rodando e com testes. O pipeline novo vive em módulos próprios. Um teste de
regressão exige que o k-NN novo, com desempate por posição e união, reproduza exatamente o
`build_union_knn_graph` do piloto.

## Decisões técnicas

1. **Modelo e carregamento.** `Qwen/Qwen3-4B-Base`, com revisão fixada por SHA, carregado com
   `AutoModel` (sem a cabeça de LM). Usar `dtype=torch.bfloat16` (`torch_dtype` está obsoleto no
   transformers 5.16), `device_map="auto"` e `max_memory` de cerca de {GPU: 6 GiB, CPU: 10 GiB}.
   Os pesos ocupam cerca de 7,5 GiB e a RTX 4070 tem 8 GiB.
2. **Representações.**
   - Lexical: linhas de `embed_tokens.weight`, pelo mesmo princípio de `extract_input_embeddings`.
   - Contextuais: saída **bruta** dos blocos 1, 18 e 36, capturada por hooks em `layers[0]`,
     `layers[17]` e `layers[35]`.
   - No transformers 5.16, `hidden_states[36]` vem **depois** da RMSNorm final. Essa versão
     (hook em `norm`) fica como variante de robustez.
   - Hooks nas 36 camadas gravam os vértices num memmap em disco (cerca de 2,8 GB em bf16),
     para a curva camada a camada da P2.
   - Embeddings amarrados (`tie_word_embeddings: true`, confirmado no config do Qwen3-4B;
     conferir no Base): e_t também é a linha de saída do modelo. Mencionar isso na
     interpretação.
3. **Sequências.**
   - Um parágrafo por forward pass (lote 1), prefixado com `<|endoftext|>` (id 151643), que
     nunca vira vértice.
   - Cada parágrafo é processado só até o último vértice escolhido nele, o que é válido porque o
     modelo é causal.
   - Vértices com o mesmo prefixo de tokens têm estados matematicamente iguais (por exemplo, "O"
     no começo de vários parágrafos). Agrupar pelo hash do prefixo e copiar os vetores do
     primeiro membro, registrando a maior diferença como diagnóstico de ruído numérico.
4. **Tokens.**
   - Categorias pelos offsets de caracteres, não pelo decode: palavra inteira, início de
     fragmentada, continuação, pontuação e **número**.
   - Tokens só de espaço não viram vértices (" 1990" → "Ġ", "1", "9", "9", "0").
   - Palavras-alvo contam só no id minúsculo com espaço antes (`Ġbanco`). "(banco)" vira "ban" +
     "co", e "Estado" é outro tipo.
5. **Corpus.**
   - 8 temas definidos por categorias, com busca até 2 níveis e cache das respostas cruas da API.
   - Registrar todos os temas que cada página alcança. Descartar **páginas multitema** e
     **biografias** (infoboxes `Info/Biografia`, `Info/Futebolista`…), que misturam vocabulário
     entre temas.
   - Categorias extras, ligadas aos 8 temas, para os sentidos fracos. Contagens atuais no corpus:
     nota/Economia (7), carga/Economia (6), órgão/Biologia (6), banco/Geografia (4); matriz e
     planta têm o segundo sentido fraco.
6. **Amostra híbrida** (15 mil vértices, `f_max = 50`, sementes fixas).
   - **Núcleo:** por tema, 5 artigos × 2 parágrafos (cerca de 80 parágrafos e 11 mil vértices),
     para que artigo e parágrafo sejam rótulos distintos na P3. O limite por tipo é aplicado por
     **sorteio uniforme** depois de juntar tudo, e não por ordem de chegada.
   - **Alvos** (12: banco, campo, órgão, estado, carga, rede, nota, capital, família, massa,
     matriz, planta): cotas por **tema de sentido** (2–3 temas, com ≥ 10–15 ocorrências cada).
     No máximo 1 ocorrência por (tipo, parágrafo) e 3 por (tipo, artigo). Um alvo que não chegar
     a ≥ 10 ocorrências em ≥ 2 temas sai da lista.
   - **Controles** (ano, século, grande, primeiro, importante, água): **mesma alocação** dos
     alvos (mesmo número de temas e mesmas cotas), para não enviesar a comparação da P4.
   - **Multitema:** 150 palavras de conteúdo (≥ 5 ocorrências em ≥ 3 temas fora do núcleo) × 20
     ocorrências.
7. **Onde o k-NN é aplicado.** Os tipos da amostra (os token_ids distintos entre as 15 mil
   ocorrências, cerca de 3,3 mil) saem só da amostragem, antes de qualquer k-NN. Depois, o k-NN
   roda sobre três conjuntos de vértices:
   - **15 mil ocorrências:** as redes principais (lexical por ocorrência e camadas 1, 18 e 36),
     onde P1–P4 são respondidas. Elas ficam restritas à amostra porque só tokens que aparecem
     num texto têm vetor contextual.
   - **Vocabulário inteiro (cerca de 151,6 mil tokens):** a rede lexical de tipos que entra nos
     resultados, com as métricas da disciplina.
     - Ficam de fora os 26 tokens especiais e as 267 linhas da matriz que o tokenizer nunca usa.
     - Atributos de cada tipo: frequência no corpus e na amostra (zero para a maioria), classe
       de escrita (latina, chinês/japonês/coreano, código/misto, pontuação, outras escritas,
       pedaços de bytes) e se aparece no corpus.
     - Composição medida: 50% escrita latina, 20% chinês/japonês/coreano, 13% código/misto.
     - Tokens pouco treinados podem ter embeddings quase iguais e formar aglomerados ou hubs;
       isso é diagnosticado e reportado.
   - **Cerca de 3,3 mil tipos da amostra:** uma tabela de vizinhos auxiliar, não reportada como
     rede. Ela serve para (i) derivar de forma exata a rede lexical por ocorrência (decisão 8) e
     (ii) dar o T_i lexical da variante (c).
   - A tabela auxiliar não sai de um recorte da rede do vocabulário. O recorte perde vizinhos
     que estão fora da amostra e não tem arestas que o k-NN calculado só na amostra tem.
8. **k-NN e empates.**
   - Similaridades em **float64** na CPU, em blocos. Em float32 o erro fica em torno de
     10⁻⁶–10⁻⁵ com d = 2560, o que estraga ε = 10⁻⁶.
   - Candidatos top-128 por representação, com garantia de conter todo j com s ≥ s₍₂₀₎ − ε; se
     não contiver, a linha inteira é recalculada.
   - A mesma rotina serve para a rede do vocabulário (151,6 mil vetores, cerca de 3 GB em
     float64). Estimativa: de alguns minutos a uns 20 minutos na CPU.
   - Um seletor único, com F = {j: s > s_k + ε} e B = {j: |s − s_k| ≤ ε}, gera as variantes: (a)
     sorteio em B com 20 sementes, mais o desempate por posição como referência; (b) F ∪ B.
   - A rede lexical por ocorrência é derivada **de forma exata** do ranking entre os tipos da
     amostra e das contagens. Há um único valor de similaridade por par de tipos, então os
     empates são exatos.
   - Variante (c): máximo por tipo nas colunas (`np.maximum.reduceat`), excluindo o próprio tipo.
     Ela entra só nas medidas de vizinhança (Jaccard sobre conjuntos de tipos).
   - k = 10 como principal, 5 e 20 como robustez; ε = 10⁻⁴ como teste de sensibilidade.
9. **Medidas.**
   - J_i e D_i. Na variante (b), D divide por |N_i|. Reportar também D relativo ao teto lexical
     min(f_t − 1, k)/k.
   - Piso de ruído: Jaccard entre sementes lexicais, por faixa de frequência.
   - Complemento: Jaccard por composição de tipos, J^w, que não depende da semente na etapa
     lexical.
   - Faixas de frequência pela amostra (em parte definida pelo próprio desenho) **e** pelo
     corpus.
10. **Métricas e comunidades com igraph.**
    - Distância média, diâmetro e histograma de distâncias exatos (em C; segundos a um minuto
      por rede de 15 mil vértices, a medir), clusterização e componentes.
    - Na rede do vocabulário, distâncias exatas só na configuração principal (estimativa: de
      dezenas de minutos a cerca de 1 hora). Nas variantes de robustez dela, estimativa por
      busca em largura a partir de fontes sorteadas e limite inferior do diâmetro por varredura
      dupla.
    - Leiden com `objective_function="modularity"` (o padrão do igraph é CPM), 10 sementes,
      ficando com a de maior modularidade e reportando a estabilidade. Louvain como checagem.
    - NMI/ARI via `igraph.compare_communities`, com linha de base por permutação; pureza em
      numpy.
    - O NetworkX fica como referência nos testes.

## Etapas de implementação

### 0. Preparação e limpeza
- Os scripts e resultados do estudo de viabilidade ficam em `scripts/feasibility/`.
- **Remover:** `pipeline.py`, `prompts.py`, `storage.py`, `scripts/run_pilot.py`, `data/prompts/`,
  `outputs/activations/`, `tests/test_prompts.py`, `tests/test_storage.py` e
  `pilot_rede_contextual.ipynb`.
- Em `src/gender_networks/tokenization.py`, tirar `compare_positionally` e afins. Fica o
  `tokenize_prompt`, que o piloto usa; trocar o teste de comparação por um teste de
  `tokenize_prompt`.
- **Não mexer no piloto:** `src/gender_networks/network_pilot.py`,
  `src/gender_networks/network_analysis.py`, `scripts/run_network_pilot.py` e os testes deles.
- **Ajustes pequenos:**
  - `src/gender_networks/modeling.py`: `dtype=` no lugar de `torch_dtype=`;
  - `src/gender_networks/config.py`: `max_memory` e as configurações do experimento, mantendo o
    padrão de dataclasses congeladas;
  - `configs/model.yaml` atualizado para o Base, e `tests/test_config.py` junto.
- **pyproject, README e .gitignore:**
  - `pyproject.toml`: nova descrição, entrada `gender-networks` para a nova CLI em
    `src/gender_networks/cli.py`, dependências `igraph`, `mwparserfromhell` e `pandas`;
  - README reescrito;
  - `.gitignore` para `data/raw/`, `data/corpus/*.jsonl`, `outputs/experiment/`,
    `scripts/feasibility/*.jsonl` e os arquivos auxiliares do LaTeX em `report/` (`*.aux`,
    `*.log`, `*.fls`, `*.fdb_latexmk`, `*.synctex.gz`, `*.out`).
- Criar o esqueleto do relatório técnico em `report/relatorio/` (seção "Relatório técnico") e
  já escrever o que existe hoje: motivação e perguntas (da proposta), estudo de viabilidade e
  as decisões e os problemas da fase de planejamento.
- Sugestão: versionar `report/`, que hoje está fora do git.

### 1. `corpus` — `wiki.py`, `textclean.py`, `configs/corpus.yaml`
Evolução do `fetch_wiki.py`:
- busca por categorias com cache cru;
- wikitext em lotes de 50 por `revid`, com pausa de 2,5 s, respeito ao `Retry-After` e retomada;
- `User-Agent` com um contato preenchido no config (política da Wikimedia);
- limpeza com mwparserfromhell, cortando Referências, Ligações externas e seções parecidas;
- parágrafos com ≥ 40 palavras, fim de frase e ≤ 8% de dígitos, sem duplicatas;
- sentenças por regex, com uma lista de abreviações.

Saídas: `data/corpus/articles.jsonl` e `paragraphs.jsonl`. Versionar só
`data/corpus/manifest.csv` (pageid, revid, título, tema, categoria de origem).

### 2. `sample` — `tokens.py`, `sampling.py`, `configs/sample.yaml`
Tokenização com offsets e rotulagem (decisão 4), e amostra híbrida (decisão 6), com registro das
faltas por (alvo, tema).

Saída principal: `data/sample/occurrences.csv`, uma linha por vértice, com:
- identificação: id (a ordem segue sequência e posição, e define o desempate por posição) e
  estrato;
- origem: tema, pageid, revid, título, paragraph_id, sentence_id;
- posição: posição na sequência, faixa de posição, grupo de prefixo;
- token: token_id, texto, offsets, token anterior e seguinte;
- palavra: a palavra, a posição do token nela e a categoria;
- frequência: na amostra e no corpus, com as faixas correspondentes;
- estratos: palavra-alvo e tema de sentido;
- contexto local.

Também saem `sequences.jsonl` e `sample_manifest.json`. A anotação manual de sentido, com cerca de
20 ocorrências por alvo, vai em `data/sample/senses.csv`. Um arquivo `vocab_types.csv` guarda,
para cada token do vocabulário, a classe de escrita e as frequências no corpus e na amostra.

### 3. `extract` — `extract.py`
Aplica as decisões 1–3. Antes da extração completa, `--verify` roda em 8 sequências e checa:
- `len(hidden_states) == 37`;
- `hidden_states[0] == embed_tokens(ids)` (o RoPE não soma nada ao fluxo residual);
- o hook de `layers[k−1]` é igual a `hidden_states[k]` para k ∈ {1, 18, 35};
- `norm(hook_36) == hidden_states[36]`;
- duas execuções dão resultado idêntico.

Diagnósticos por representação:
- quantis de norma;
- cosseno médio de pares aleatórios (anisotropia);
- dimensões dominantes (ativações massivas);
- normas por faixa de posição.

Saídas em `outputs/experiment/reps/`: safetensors em bf16, o memmap das 36 camadas e a matriz de
embeddings inteira (151.669 linhas, cerca de 780 MB em bf16), usada pela rede do vocabulário e
pela extensão.

### 4. `knn` — `knn.py`
Aplica a decisão 8. As listas de candidatos ficam em formato CSR (índices int32, similaridades
float64). Os conjuntos de vizinhos ficam em `.npz`, um por (representação, k, variante, semente):
estimativa de 100 a 300 MB no total, mais o k-NN da rede do vocabulário.

### 5. `metrics` e `analyze` — `graph_metrics.py`, `neighborhood.py`, `partitions.py`, `analysis.py`
Métricas exigidas pela disciplina em todas as redes da configuração principal: |V|, |E|, grau
médio, distribuição de graus (entrada e saída), densidade, clusterização global e local,
distância média, diâmetro, componentes e distribuição de tamanhos. Somar reciprocidade e hubs.

A grade de robustez roda em paralelo: estimativa de cerca de 200 grafos distintos e 15–20
minutos com 12 processos, fora as distâncias exatas da rede do vocabulário. As análises estão
na seção seguinte.

### 6. `report` — `plots.py`
Gera as figuras (PDF), as tabelas (.tex) e o arquivo de números (`numeros.tex`) do relatório
técnico a partir dos artefatos das etapas já executadas. Pode rodar a qualquer momento, então o
relatório sempre mostra os resultados da última execução. Detalhes na seção "Relatório
técnico". No relatório, ajustar a frase da Seção 7.3 da proposta: na etapa lexical, T_i coincide
com o k-NN entre os tipos da amostra, e não mais com a rede lexical de tipos (que agora é a do
vocabulário).

**CLI:** `uv run gender-networks <corpus|sample|extract|knn|metrics|analyze|report>`. Cada etapa
lê os artefatos da anterior e grava um `_manifest.json` com config, versões das bibliotecas e
tempos. Um `configs/mini.yaml` (2 temas, cerca de 1.500 vértices) serve para ensaiar tudo de
ponta a ponta rapidamente.

### 7. Extensão opcional (semana 5): vizinhos no vocabulário — `vocab_lens.py`
Para cada ocorrência e representação (lexical, blocos 1, 18 e 36, e bloco 36 normalizado):
- calcular os k tokens do vocabulário inteiro mais próximos por cosseno (float32 na GPU; não há
  tratamento de empates, porque só entram posições no ranking);
- medir, camada a camada, a posição do **próprio token** e do **próximo token** nesse ranking;
- medir o Jaccard dos conjuntos de vizinhos no vocabulário entre camadas;
- reportar por categoria de token e por estrato.

A pergunta que a curva responde: em que camada a ocorrência deixa de parecer com o próprio token
e passa a parecer com o próximo. Cuidado de interpretação: nas camadas intermediárias, a
comparação mistura espaços que não foram treinados para se alinhar. Só o bloco 36 normalizado é
exatamente o que a matriz de saída lê (embeddings amarrados). Custo estimado: minutos.

## Análises por pergunta
- **P1:**
  - distribuições de J e D por faixa × categoria × estrato, controladas por faixa de posição;
  - CCDF do grau de entrada, hubs e reciprocidade;
  - métricas globais por camada e comunidades.
- **P2:**
  - J acima do piso de ruído e ΔD em cada transição (lexical→1, 1→18, 18→36, e as acumuladas);
  - fração de vértices cuja maior mudança cai em cada transição;
  - curvas J(ℓ, ℓ+1) e D(ℓ) nas 36 camadas.
- **P3** (só no núcleo):
  - NMI (bruta e menos a linha de base), ARI e pureza das comunidades contra token, tema,
    artigo, parágrafo, sentença, próximo token e faixa de posição (esta como controle de
    artefato), camada a camada;
  - tema também na rede inteira.
- **P4** (tipos com f ≥ 10, com alvos, controles, multitema e palavras funcionais separados):
  - número efetivo de comunidades ocupadas (exp H);
  - NMI entre comunidade e tema dentro do tipo;
  - autossimilaridade (cosseno médio entre as próprias ocorrências);
  - diferença entre o cosseno dentro do mesmo tema e entre temas;
  - separação pelo sentido anotado;
  - redes ego de 2–3 alvos.
  - Hipótese: os alvos se separam mais que os controles.
- **Rede lexical de tipos (vocabulário):**
  - distribuição de graus, hubs e clusterização;
  - comunidades comparadas com a classe de escrita (NMI/ARI);
  - onde ficam os tokens que aparecem no corpus (fração por comunidade);
  - vizinhança das palavras-alvo no vocabulário inteiro.
- **Robustez:** k ∈ {5, 20}, versão mútua, variantes (b) e (c), desempate por posição, bloco 36
  normalizado, cosseno centrado, ε = 10⁻⁴, resolução do Leiden e rede só do núcleo (cerca de 11
  mil vértices, acima de 10⁴).

## Relatório técnico (escrito junto com a implementação)

Um relatório detalhado, atualizado a cada etapa, que registra o que foi feito, por que foi feito
assim, o que deu errado e o que se encontrou. Ele serve de base para o relatório final da
disciplina, que pode ser uma versão condensada dele.

**Formato e local.** LaTeX, como a proposta, em `report/relatorio/`:
- `relatorio.tex` como documento principal e `secoes/` com um arquivo por seção;
- `figuras/` e `tabelas/`, preenchidas pela etapa `report` (arquivos gerados, que não se editam
  à mão);
- `tabelas/numeros.tex` com macros para os números citados no texto (por exemplo, `\nVertices`
  e `\nTiposVocab`), para o texto nunca divergir da última execução;
- compilação com `latexmk -pdf relatorio.tex`.

**Seções.**
1. Introdução e motivação.
2. Perguntas de pesquisa e hipóteses (P1–P4).
3. Terminologia: palavra, token, tipo, ocorrência e categorias de token.
4. Dados: corpus (temas, categorias, limpeza, filtros e estatísticas), estudo de viabilidade e
   amostra híbrida (desenho, cotas, composição e faltas por palavra-alvo).
5. Modelo e extração: Qwen3-4B-Base, camadas escolhidas, hooks e a questão da RMSNorm final,
   prefixo, grupos de prefixo e diagnósticos de norma e anisotropia.
6. Construção das redes: rede do vocabulário, redes por ocorrência, k-NN em float64,
   tratamentos de empate e versões direcionada, por união e mútua.
7. Métricas e métodos de análise: métricas estruturais, Jaccard, dominância, J^w, piso de
   ruído, comunidades e concordância entre partições.
8. Resultados: rede do vocabulário, P1, P2, P3 e P4, e a extensão "vizinhos no vocabulário",
   se for feita.
9. Robustez.
10. Discussão.
11. Limitações.
12. Problemas encontrados e soluções.
13. Apêndices: registro de decisões e reprodutibilidade (versões, SHA do modelo, sementes e
    comandos de cada etapa).

**Registro de decisões.** Cada decisão entra numa tabela com contexto, alternativas
consideradas, escolha, motivo e consequência. O registro começa com as decisões do
planejamento:
- Qwen3-4B-Base em vez do pós-treinado, do Tucano-2b4 e do GPT-2 português;
- desenho híbrido em vez de parágrafos inteiros;
- rede de tipos sobre o vocabulário inteiro;
- similaridades em float64;
- saída bruta do bloco 36;
- igraph e Leiden;
- piloto congelado e remoção do código legado.

**Problemas encontrados.** Cada problema entra com sintoma, causa, solução e impacto nos
resultados. Já há entradas do planejamento:
- limitação de taxa da API da Wikipédia (HTTP 429), resolvida com lotes, pausas e retomada;
- páginas de usuário e categorias de manutenção entrando na busca por categorias;
- fragmentação do português no tokenizer do Qwen3 (só 28% dos vértices são palavras inteiras),
  que limitou as palavras-alvo às que viram um token só;
- `hidden_states[36]` vindo depois da RMSNorm final no transformers 5.16, e `torch_dtype`
  obsoleto;
- na simulação, limite por tipo aplicado por ordem de chegada e artigo quase igual a parágrafo
  no núcleo, ambos corrigidos no desenho final.

**Figuras previstas**, cada uma gerada pela etapa que a produz:
- estudo de viabilidade: comparação dos tokenizers; amostragem natural contra híbrida;
- corpus: parágrafos por tema, distribuição de tamanhos, páginas descartadas por motivo;
- amostra: composição por estrato, tema, faixa de frequência e categoria; cotas das
  palavras-alvo;
- extração: normas e cosseno médio por camada; dimensões dominantes;
- redes: distribuições de grau de entrada e saída, redes ego de palavras-alvo e desenhos de
  subgrafos com layout fixo;
- análises: curvas de J e D por camada, NMI por camada e rótulo, separação intra-tipo da P4;
- rede do vocabulário: distribuição de graus e comunidades por classe de escrita.

Todas as figuras seguem um estilo único, definido em `plots.py`, em PDF vetorial e legíveis em
preto e branco.

**Regra de trabalho.** Uma etapa só está pronta quando:
- o código e os testes passam;
- a seção correspondente do relatório foi escrita ou atualizada, explicando o que foi feito e
  por quê;
- as decisões, os problemas e as limitações da etapa estão registrados;
- as figuras e tabelas da etapa foram geradas;
- o relatório compila sem erros.

## Cronograma (Semana 1 = 28/09–04/10; alinhado à Seção 9 da proposta)
| Semana | Implementação | Relatório |
|---|---|---|
| 1 | Preparação e limpeza; etapa `corpus` com categorias extras; `tokens.py` e `textclean.py` com testes; download do modelo | Esqueleto; seções 1–4 (introdução, perguntas, terminologia, dados e estudo de viabilidade); primeiros registros de decisões e problemas |
| 2 | `sampling.py` e congelamento da amostra; `extract.py` (`--verify`, config mini, extração completa, diagnósticos); início da anotação de sentidos | Amostra híbrida (Seção 4) e modelo e extração (Seção 5), com os diagnósticos |
| 3 | `knn.py` com teste de regressão e comparação com força bruta; variantes e sementes; k-NN do vocabulário; J, D, piso de ruído e J^w | Construção das redes e métodos (Seções 6–7); primeiros resultados de P1 e P2 |
| 4 | Métricas no igraph (incluindo as distâncias da rede do vocabulário), Leiden, NMI/ARI; configuração principal; P3 e P4 | Resultados da rede do vocabulário, de P3 e de P4 |
| 5 | Grade de robustez, curva camada a camada; extensão opcional "vizinhos no vocabulário"; nova execução limpa a partir de `sample` | Robustez, extensão e discussão |
| 6 | Consolidação | Revisão completa (resumo, limitações, conclusões e reprodutibilidade); relatório final da disciplina condensado a partir dele |

## Riscos
- **GPU sem memória ou offload lento no WSL2:** reduzir `max_memory` ou usar lotes por orçamento
  de tokens. Último recurso: Qwen3-1.7B, que cabe inteiro na GPU.
- **Limitação de taxa da Wikipédia:** cache cru, respeito ao `Retry-After`, retomada e fixação
  por `revid`.
- **Poucas ocorrências por sentido:** categorias extras; o alvo sai se não atingir o mínimo;
  anotação manual.
- **Ativações massivas e hubness:** diagnósticos, cosseno centrado e k-NN mútuo. A hubness vira
  um achado da P1.
- **Ruído dos empates escondendo a mudança:** piso de ruído por faixa, J^w, variante (c) e D
  normalizado.
- **Viés do NMI e desenho híbrido:** linha de base por permutação, ARI, só o núcleo na P3 e
  resultados por estrato.
- **Rede do vocabulário grande (151,6 mil vértices):** distâncias exatas só na configuração
  principal; estimativas por amostragem nas variantes; memória controlada com cálculo em blocos.
- **Relatório ficar para trás do código:** a regra de trabalho inclui o relatório em cada
  etapa, e figuras, tabelas e números são gerados automaticamente.
- **Variantes demais:** uma configuração principal fixa (k = 10, união, empate (a)) e a
  robustez reunida numa tabela automática.

## Verificação
- `uv run pytest` e `uv run ruff check .`, com testes que não baixam pesos:
  - `test_knn.py`:
    - regressão contra o piloto;
    - derivação lexical exata igual à força bruta;
    - invariantes do seletor (F ⊂ N ⊂ F ∪ B);
    - uniformidade e reprodutibilidade do sorteio;
    - bordas de ε;
    - recálculo quando o empate passa de K;
    - (c) exclui o próprio tipo;
    - união e mútua iguais ao NetworkX.
  - `test_neighborhood.py`: J vetorizado igual ao J por conjuntos; D lexical = min(f − 1, k)/k na
    variante (a); J^w invariante à semente na etapa lexical.
  - `test_graph_metrics.py` e `test_partitions.py`: igraph igual ao NetworkX em grafos pequenos;
    NMI, ARI e pureza em partições conhecidas; Leiden determinístico dada a semente.
  - `test_tokens.py`: casos "praça", " 1990", "tornou-se" e "(banco)"; classe de escrita de
    tokens conhecidos.
  - `test_sampling.py`: limite aplicado por sorteio, cotas, 2 parágrafos por artigo,
    reprodutibilidade pela semente.
  - `test_extract.py`, com um Qwen3 minúsculo e aleatório: hooks iguais aos `hidden_states`,
    `norm(hook) == hidden_states[L]`, cópia por grupo de prefixo.
- O piloto continua rodando: `uv run python scripts/run_network_pilot.py`.
- O relatório compila sem erros com `latexmk -pdf` ao fim de cada etapa, sem referências
  quebradas a figuras ou tabelas.
- Os números de resultados citados no texto vêm de `numeros.tex`; nenhum é digitado à mão.
- `extract --verify` no modelo real e ensaio completo com `configs/mini.yaml` antes da execução
  principal.
- Extensão: na etapa lexical, o próprio token é o primeiro do ranking; no bloco 36 normalizado,
  o maior produto interno com a matriz de embeddings coincide com o argmax dos logits em
  praticamente todas as ocorrências.
- Asserções ao fim de cada etapa:
  - o mesmo |V| em todas as redes por ocorrência (cerca de 15 mil);
  - rede de tipos com cerca de 151,6 mil vértices e nenhum token especial;
  - grau de saída igual a k;
  - nenhum laço;
  - J(lex_s, lex_s) = 1.
