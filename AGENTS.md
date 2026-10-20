# Guia de desenvolvimento

Para quem vai mudar o código, pessoa ou agente de IA. O que o projeto faz e como rodar cada etapa
está no [README](README.md); aqui ficam as regras e o roteiro de verificação.

## O pipeline

`corpus → sample → extract → knn → metrics → analyze → lens → report`, uma etapa por módulo
(`uv run gender-networks <etapa> --config <yaml>`). Cada `run(settings, paths, force)`:

1. pula a etapa quando ela está atualizada (`artifacts.up_to_date`: saídas presentes, entradas e
   seções da configuração que ela usa sem mudança);
2. confere as entradas (`artifacts.require_inputs`);
3. lê só artefatos de etapas anteriores, pelos caminhos de `RunPaths`;
4. grava as saídas e, por último, o `_manifest.json` (`artifacts.write_manifest`).

## Onde mexer

| Para… | Mexa em |
|---|---|
| uma camada ou transformação nova (padronizar, tirar componentes principais…) | `representations.py` (o nome e o que ele significa) e a configuração |
| um arquivo novo de alguma etapa | as listas e os métodos de `RunPaths` em `artifacts.py`; nunca monte o nome com f-string no módulo que lê |
| um rótulo ou texto em português | `labels.py` |
| uma estatística reaproveitável | `analysis.py`, gravada numa tabela de `ANALYSIS_TABLES` |
| uma figura ou tabela do relatório | uma função em `report/sections/<seção>.py`, um `Item` em `report/__init__.py` e o `\figuraopcional`/`\tabelaopcional` em `report/relatorio/secoes/` (os testes conferem a correspondência) |
| uma métrica de grafo | `graphs.py` (scripts e etapas a importam; não copie) |

## Regras

- **Resultados não mudam sem querer.** Uma refatoração tem de deixar todas as saídas iguais.
  Confira com o ensaio mini antes de commitar (roteiro abaixo). Quando a mudança altera
  resultados de propósito, diga quais arquivos mudam e por quê, no commit.
- **Material do relatório é gerado.** Figuras (`figuras/*.pdf`), tabelas (`tabelas/*.tex`) e
  números (`tabelas/numeros.tex`) saem da etapa `report`; não edite à mão.
- **Idioma.** Código, nomes de arquivo e chaves dos dados em inglês; documentos LaTeX, as suas
  pastas e o texto que o leitor vê, em português.
- **Caminhos publicados.** Não mova `report/entrega-parcial*/`, `docs/` nem `data/graphml/`: as
  entregas enviadas e o GitHub Pages apontam para eles.
- **Testes.** `uv run pytest` e `uv run ruff check .` antes de cada commit. Os testes rodam na
  CPU, sem baixar pesos.

## Conferir que uma mudança não alterou resultados

O ensaio mini (`configs/mini.yaml`, cerca de 1.500 vértices) roda nesta máquina.

```bash
# 1. Antes da mudança: guarde as saídas atuais (outputs/ é ignorado pelo git).
cp -a outputs/experiment/mini outputs/experiment/mini-golden

# 2. Depois da mudança: refaça as etapas afetadas (a lens antes do report, que lê as saídas dela).
for s in knn metrics analyze lens report; do
  uv run gender-networks $s --config configs/mini.yaml --force
done

# 3. Compare (ignora manifests e tempos; --include-reps compara também as representações).
uv run python scripts/dev/compare_runs.py outputs/experiment/mini-golden outputs/experiment/mini
```

Sem mudança no `knn.py`, comece pelo `metrics`: o `knn` leva de 12 a 25 minutos, quase tudo na
rede do vocabulário. O `extract` precisa da GPU; numa de 8 GiB, use uma configuração que herde
o mini e reserve 6 GiB, como na extração original:

```yaml
base: <caminho absoluto>/configs/mini.yaml
model:
  max_memory: {0: 6GiB, cpu: 10GiB}
```

Os artefatos da execução principal (`outputs/experiment/main/`) ficam na máquina com a RTX
A5500, não nesta.
