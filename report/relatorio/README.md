# Relatório técnico (documento vivo)

Relatório detalhado do experimento "Da rede lexical à rede contextual", escrito junto com a
implementação. Cada etapa do pipeline só está pronta quando a seção correspondente foi
atualizada e o documento compila sem erros.

## Estrutura

| Caminho | Conteúdo | Quem escreve |
|---|---|---|
| `relatorio.tex` | documento principal: pacotes, macros, capa, bibliografia | à mão |
| `secoes/NN-*.tex` | uma seção por arquivo (01 introdução ... 13 apêndices) | à mão |
| `figuras/*.pdf` | figuras em PDF vetorial | **gerado** |
| `tabelas/*.tex` | tabelas (só o ambiente `tabular`, com booktabs) | **gerado** |
| `tabelas/numeros.tex` | macros com os números citados no texto (`\nVertices`, `\nTiposVocab`...) | **gerado** |

## Material gerado: nunca editar à mão

- `figuras/` e `tabelas/` são escritas pela etapa `report` do pipeline
  (`uv run gender-networks report --config configs/experiment.yaml`) a partir dos artefatos da
  última execução. As duas figuras do estudo de viabilidade (`viabilidade_*.pdf`) são a exceção:
  vêm de `scripts/feasibility/plot_feasibility.py`, com números fixos do estudo.
- `tabelas/numeros.tex` define com `\newcommand` as macros dos números citados no texto. Em
  `relatorio.tex`, cada macro tem um `\providecommand` que imprime *[pendente]* enquanto o
  arquivo não existe ou não a define. Para citar um número novo: acrescentar o
  `\providecommand` em `relatorio.tex` e fazer a etapa `report` gravar a macro. Números de
  resultados nunca são digitados no texto.
- Os números em `numeros.tex` e nas tabelas são **texto simples já formatado em pt-BR**
  (`15.104`, `0,83`), sem `\num`: o relatório não carrega o `siunitx`, e envolver um número
  formatado em pt-BR com `\num` num ambiente que o carregue trocaria `15.104` por `15,104`.
- Figuras e tabelas entram pelos macros `\figuraopcional[quem gera]{arquivo}{legenda}{rótulo}` e
  `\tabelaopcional[quem gera]{arquivo}{legenda}{rótulo}` (arquivo sem extensão). Se o arquivo
  ainda não existe, aparece um quadro "pendente" com o caminho esperado e a etapa que o produz;
  a legenda e o rótulo ficam no texto, então as referências nunca quebram.
- As figuras são desenhadas na largura final (no máximo a largura do texto): o macro só reduz
  uma figura mais larga que a coluna, para as fontes não mudarem de tamanho.

O ensaio com `configs/mini.yaml` grava o material gerado em `outputs/experiment/mini/relatorio/`,
e não aqui.

## Compilação

```bash
cd report/relatorio
latexmk -pdf relatorio.tex          # ou: latexmk -pdf -interaction=nonstopmode relatorio.tex
latexmk -c                          # remove os arquivos auxiliares (já ignorados pelo git)
```

O documento compila antes de qualquer execução do pipeline. Requer uma distribuição TeX com
`babel` (português), `lmodern`, `microtype`, `amsmath`, `amssymb`, `graphicx`, `booktabs`,
`tabularx`, `longtable`, `array`, `xcolor`, `enumitem`, `geometry`, `titlesec`, `caption`,
`tikz`, `seqsplit` e `hyperref`. No TeX Live, isso corresponde a `texlive-latex-recommended`
mais `texlive-latex-extra` (onde está o `seqsplit`) e `texlive-lang-portuguese`.
