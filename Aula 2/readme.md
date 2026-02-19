# Aula 2: Básico de Python & Bibliotecas de Data Science

Este notebook é uma introdução prática aos conceitos fundamentais de Python e às principais bibliotecas utilizadas em Data Science.

## 📚 Objetivos

- Revisar conceitos fundamentais do Python usados em Data Science
- Manipular dados com **NumPy** e **Pandas**
- Construir gráficos com **Matplotlib**
- Realizar uma análise exploratória de dados (EDA) simples
- Entender variáveis, padrões, outliers, valores ausentes e relações entre colunas

## 📋 Conteúdo

### 1. Revisão de Python
- **Listas**: criação, acesso, operações básicas (append, extend, insert, remove, pop, sort)
- **Dicionários**: pares chave-valor, iteração, contagem
- **Estruturas de repetição**: loops `for` e `while`
- **Compreensão de listas**: forma compacta de criar e transformar listas

### 2. Bibliotecas de Data Science

#### 2.1 NumPy
- Criação e operações com arrays (ndarrays)
- Operações vetorizadas (sem loops explícitos)
- Reshape e manipulação de matrizes
- Funções agregadas (sum, mean, etc.)

#### 2.2 Pandas
- Séries e DataFrames (tabelas de dados)
- Seleção de colunas e filtragem
- Estatísticas descritivas
- Agrupamento por categoria (groupby)
- Tratamento de valores ausentes (NaN)

#### 2.3 Matplotlib
- Histogramas (distribuição de variáveis)
- Gráficos de barras (comparação entre categorias)
- Scatter plots (relação entre duas variáveis)
- Boxplots (análise de outliers)

### 3. Análise Exploratória de Dados (EDA)

Checklist mínimo de EDA:
1. Entender o contexto do dataset
2. Inspecionar estrutura (head, shape, dtypes)
3. Verificar valores ausentes
4. Gerar resumo numérico (describe)
5. Resumo categórico (value_counts)
6. Criar visualizações

### 4. Atividade Final

Aplicar EDA em um dataset real usando o **Iris dataset** (já carregado) ou um CSV local.

## 🛠️ Pré-requisitos

- Python 3.7+
- Bibliotecas instaladas:
  ```bash
  pip install numpy pandas matplotlib scikit-learn
  ```

## 🚀 Como Usar

1. Abra o notebook `BasicoPython&Bibliotecas.ipynb` em Jupyter Notebook ou Jupyter Lab
2. Execute as células sequencialmente (pode usar Shift + Enter)
3. Complete os exercícios propostos (marcados com **Exercício**)
4. Siga o template de atividade final para praticar EDA

## 📊 Principais Bibliotecas

| Biblioteca | Versão Mínima | Uso |
|-----------|---------------|-----|
| NumPy | 1.19+ | Cálculos numéricos e arrays |
| Pandas | 1.1+ | Manipulação de dados tabulares |
| Matplotlib | 3.0+ | Visualização de dados |
| scikit-learn | 0.23+ | Dataset Iris (exemplo) |

## ✏️ Exercícios Inclusos

1. **Listas**: Converter temperaturas de Celsius para Fahrenheit
2. **Dicionários**: Contar frequência de palavras
3. **EDA Completa**: Análise do Iris dataset com pelo menos 3 gráficos e 3 insights

## 💡 Dicas de Uso

- Use `display()` para melhor visualização de DataFrames
- Sempre verifique valores ausentes antes de trabalhar com dados
- Pratique list comprehension - é muito usada em Python
- Matplotlib é ótimo para EDA inicial; explore também Seaborn e Plotly depois

## 📝 Saída Esperada

Ao completar este notebook, você será capaz de:
- ✅ Manipular dados com Python (listas, dicionários)
- ✅ Trabalhar com NumPy arrays para cálculos numéricos
- ✅ Usar Pandas para carregar, explorar e manipular dados
- ✅ Criar visualizações com Matplotlib
- ✅ Realizar uma EDA completa em qualquer dataset

## 📖 Referências Rápidas

- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

---
