# Curso de Inteligência Artificial 🤖

Bem-vindo ao curso de IA! Este repositório contém materiais didáticos e exemplos práticos sobre Inteligência Artificial, Machine Learning e Deep Learning.

## 📚 Aula 1 - Fundamentos de IA + Ambiente + Git

### Objetivos da Aula
- Entender o que é IA, ML e DL
- Diferenciar aprendizado supervisionado e não supervisionado
- Conhecer aplicações práticas de IA
- Entender e configurar ambientes virtuais (venv)
- Aprender Git e Github para versionamento de código

---

## 🧠 Conceitos Principais

### O que é Inteligência Artificial?
IA é o desenvolvimento de sistemas capazes de:
- Aprender com dados
- Identificar padrões
- Tomar decisões automaticamente

**IA = Matemática + Dados + Computação**

### Aprendizado Supervisionado vs Não Supervisionado

| Característica | Supervisionado | Não Supervisionado |
|----------------|----------------|---------------------|
| Tem rótulo (y)? | ✅ Sim | ❌ Não |
| Existe resposta correta? | ✅ Sim | ❌ Não |
| Objetivo | Prever | Descobrir padrões |
| Exemplo | Classificação | Agrupamento |

### Machine Learning vs Deep Learning
- **Machine Learning (ML)**: Algoritmos que aprendem padrões, com features selecionadas manualmente
- **Deep Learning (DL)**: Redes neurais profundas que aprendem features automaticamente

---

## 🛠️ Configuração do Ambiente

### Criando o Ambiente Virtual (VSCode)

```bash
# 1. Criar o ambiente virtual
python -m venv ia_env

# 2. Ativar o ambiente
# Windows:
ia_env\Scripts\activate
# Mac/Linux:
source ia_env/bin/activate

# 3. Instalar dependências
pip install -r requirements.txt
```

### Google Colab
No Colab, não é necessário criar ambiente virtual. Apenas execute:
```python
!pip install -r requirements.txt
```

---

## 📦 Bibliotecas Utilizadas

- `scikit-learn` - Machine Learning tradicional
- `tensorflow` - Deep Learning
- `numpy` - Operações numéricas
- `pandas` - Manipulação de dados
- `matplotlib` - Visualização de dados

Para instalar todas as dependências:
```bash
pip install -r requirements.txt
```

---

## 📊 Exemplos Práticos

O notebook contém exemplos usando o dataset **Iris**:

1. **Classificação Supervisionada** - Decision Tree
2. **Clusterização Não Supervisionada** - K-Means
3. **Comparação ML vs DL** - Decision Tree vs Neural Network

---

## 🔄 Git e GitHub - Boas Práticas

### Tipos de Commit

| Tipo | Quando usar |
|------|-------------|
| `feat` | Nova funcionalidade |
| `fix` | Correção de erro |
| `refactor` | Reorganização de código |
| `docs` | Documentação |
| `style` | Formatação |
| `perf` | Melhoria de performance |
| `test` | Testes |
| `chore` | Manutenção |

### Fluxo de Trabalho Git

```bash
# Inicializar repositório
git init

# Adicionar arquivos
git add .

# Commit com mensagem descritiva
git commit -m "(feat): Adiciona classificação com Decision Tree"

# Enviar para GitHub
git push
```

---

## 📝 Estrutura do Projeto

```
Aula 1/
├── Aula_1_Fundamentos_de_IA_+_Ambiente_+_Git.ipynb
├── README.md
└── requirements.txt
```

---

## 🎯 Atividade Proposta

Criar um projeto completo de Machine Learning com:
- ✅ Ambiente configurado (venv ou Colab)
- ✅ Código bem comentado e identado
- ✅ README.md explicativo
- ✅ requirements.txt
- ✅ Commits seguindo boas práticas
- ✅ Implementação adicional (nova feature/melhoria)

**⚠️ Importante**: Não commitar a pasta do ambiente virtual!

---

## 👨‍💻 Autor

Álvaro Sampaio

