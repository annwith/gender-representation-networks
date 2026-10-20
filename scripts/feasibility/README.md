# Estudo de viabilidade (congelado)

Registro do estudo que decidiu o modelo, o tokenizer e o desenho híbrido da amostra antes do
pipeline existir (Seção 4 do relatório técnico). Os scripts ficam como foram usados e não são
mantidos junto com o pacote: o ruff os ignora (`pyproject.toml`), e eles repetem de propósito
versões antigas de código que hoje vive em `src/gender_networks/` (busca por categorias,
limpeza de parágrafos e categorias de token).

| Arquivo | O que faz |
|---|---|
| `fetch_wiki.py` | baixa a amostra temática da Wikipédia em `wiki_articles.jsonl` (7 MB, fora do git) |
| `simulate_sample.py` | simula a amostragem natural de ocorrências para cada tokenizer; saídas em `sim_<modelo>.txt` |
| `sim_hybrid.py` | simula o desenho híbrido (núcleo de parágrafos inteiros e estratos esparsos); saídas em `hyb_<modelo>.txt`. Importa `simulate_sample.py`, então roda de dentro desta pasta |
| `plot_feasibility.py` | gera `report/relatorio/figuras/viabilidade_*.pdf` a partir dos números copiados das saídas |

Dois testes dependem destes arquivos: `tests/test_tokens.py` lê a lista `STOP` de
`simulate_sample.py`, e `tests/test_relatorio.py` roda `plot_feasibility.py`.

Para regerar as figuras:

```bash
uv run python scripts/feasibility/plot_feasibility.py
```
