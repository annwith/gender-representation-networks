# Anotação de sentidos das palavras-alvo (P4)

`senses.csv` traz o sentido de todas as 508 ocorrências das 11 palavras-alvo da execução
principal (as de `sample/occurrences.csv` com `target_word` preenchido, inclusive as que estão no
núcleo). A etapa
`analyze` lê este arquivo e mede, para cada palavra, se as ocorrências do mesmo sentido ficam mais
próximas entre si do que as de sentidos diferentes e se caem nas mesmas comunidades da rede
(`analysis/p4_senses.csv`, colunas "Sentido" da tabela da P4).

**Quem anotou.** A anotação foi feita pelo modelo de linguagem Claude Opus 5.5
(`claude-opus-5-5`, Anthropic), a pedido do autor, em duas rodadas: em 11/11/2026, as 20
ocorrências por palavra do modelo `sample/senses.csv` que a etapa `sample` gera (220 linhas); em
12/11/2026, as 288 restantes, com o mesmo protocolo e o mesmo inventário de rótulos, que ganhou os
oito sentidos que só apareceram nessa rodada. Ela não foi revisada por uma
pessoa. A coluna `annotator` registra quem anotou cada linha. Uma revisão humana deve trocar o
valor dessa coluna nas linhas que mudar.

## Colunas

| Coluna | Conteúdo |
|---|---|
| `paragraph_id`, `char_start` | a ocorrência: o parágrafo do corpus e a posição do token nele. A etapa `analyze` liga cada linha à ocorrência da amostra por esse par, que vale para qualquer execução que sorteie o mesmo parágrafo |
| `target_word`, `title`, `local_context` | só para leitura: a palavra, o artigo e o trecho em volta da ocorrência |
| `sense` | o sentido, decidido com o parágrafo inteiro (rótulos abaixo) |
| `left_determinable` | `true` quando o sentido já se decide com o que o modelo vê: o parágrafo do início até a palavra, inclusive. O Qwen3 é causal, então a representação de uma ocorrência não tem acesso ao que vem depois |
| `annotator` | quem anotou a linha |

## Como foi feito

1. **Sentido.** Para cada ocorrência foram lidos cerca de 110 caracteres antes e depois da
   palavra, tirados do parágrafo. O tema do artigo (`sense_theme`) e o título ficaram fora da
   vista, para que o rótulo dissesse o sentido e não o tema: "inversões do «campo» magnético" num
   artigo de Geografia é `campo_fisico`. Os rótulos de cada palavra foram definidos durante a
   leitura e depois uniformizados.
2. **Contexto à esquerda.** Numa segunda passada, cada ocorrência foi mostrada só com o
   parágrafo até a palavra (os últimos 260 caracteres). A pergunta era se um leitor escolheria o
   mesmo sentido só com esse trecho. Essa passada não foi cega, porque o sentido já era
   conhecido. 442 das 508 ocorrências ficaram com `true` (196 das 220 da primeira rodada e 246 das
   288 da segunda).

Uma linha em branco em `sense` fica fora da análise. Qualquer texto preenchido vira um sentido.

## Sentidos

| Palavra | Rótulo | Sentido |
|---|---|---|
| banco | `instituicao_financeira` | instituição financeira, banco central |
| | `banco_de_dados` | banco de dados |
| | `reserva_biologica` | banco de sementes |
| | `deposito_sedimentar` | banco de areia |
| campo | `campo_fisico` | campo magnético, elétrico, quântico |
| | `area_de_estudo` | área do conhecimento |
| | `campo_de_jogo` | local da partida; "em campo" |
| | `trabalho_de_campo` | no local, fora do laboratório ("experimentos de campo") |
| | `regiao_geologica` | área com um fenômeno geológico ("campo hidrotermal", "campo de petróleo") |
| | `campo_de_dados` | parte de um registro ou endereço ("campo identificador de host") |
| | `area_delimitada` | terreno destinado a outra atividade ("campo de batalha", "de detenção") |
| órgão | `orgao_biologico` | parte de um organismo |
| | `orgao_institucional` | corpo de uma instituição (comitê, senado, sindicato) |
| | `instrumento_musical` | o instrumento |
| estado | `condicao` | situação ou estado físico ("estado estacionário", "estado de emergência") |
| | `entidade_politica` | o Estado, governo, país ("estado de direito") |
| | `unidade_federativa` | estado de uma federação |
| | `verbo_estar` | particípio de *estar* ("havia estado") |
| carga | `carga_eletrica` | carga elétrica |
| | `carga_transportada` | mercadoria, carga útil, carga de um forno |
| | `esforco_aplicado` | carga mecânica ou elétrica aplicada a um sistema |
| | `onus` | fardo ("carga tributária", "carga de trabalho", "carga emocional") |
| rede | `rede_de_computadores` | rede de computadores ou de telecomunicação de dados |
| | `sistema_interligado` | sistema físico de canais ou estações (fluvial, telegráfica, geodésica) |
| | `rede_de_relacoes` | rede de pessoas, organizações ou trocas |
| | `empresa_de_midia` | emissora ou plataforma ("rede social X", rede da HBO) |
| | `malha` | o objeto de malha (rede de plâncton) |
| nota | `nota_musical` | nota musical |
| | `classificacao` | nota de crédito, *rating* |
| | `anotacao` | registro escrito ou observação ("tomar nota", nota explicativa, "digno de nota") |
| | `verbo_notar` | forma de *notar* ("nota-se", "Syme nota que") |
| | `cedula` | papel-moeda |
| | `titulo_de_credito` | nota promissória |
| | `documento_fiscal` | nota fiscal |
| capital | `recurso_economico` | capital financeiro ou físico, fator de produção |
| | `capital_imaterial` | capital humano, social, cultural |
| | `cidade_sede` | cidade sede de governo |
| | `pena_capital` | adjetivo: punível com a morte |
| família | `grupo_taxonomico` | família na classificação biológica |
| | `grupo_familiar` | parentes |
| | `grupo_organizado` | clã de uma organização criminosa ("família Profaci") |
| | `classe_de_objetos` | conjunto de itens aparentados (família de moléculas) |
| massa | `massa_fisica` | a grandeza física |
| | `porcao_de_materia` | porção de matéria ("massa de água", "massa de terra") |
| | `em_massa` | em grande escala ("comunicação em massa", "transporte de massa") |
| matriz | `matriz_matematica` | matriz em álgebra linear |
| | `substancia_envolvente` | meio em que algo está imerso (matriz extracelular, óssea) |
| | `tecido_de_origem` | camada germinativa |
| | `modelo_estrutural` | quadro, modelo ("matriz institucional") |
| | `entorno_paisagem` | paisagem dominante em volta (ecologia da paisagem) |
| | `arranjo_de_dispositivos` | conjunto de discos (*array*) |
| | `sede_empresarial` | sede de uma empresa, em oposição às filiais |
