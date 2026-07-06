# API com Troca de Modelo

Esta API, feita com FastAPI, permite fazer upload de modelos `.joblib`, trocar entre eles e usar o modelo carregado para fazer previsões.

## Como rodar

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Depois acesse `http://127.0.0.1:8000` no navegador.

## Como adaptar para o seu próprio modelo

Você não precisa reescrever a API do zero. Basta seguir os passos abaixo:

### 1. Treine e salve o seu modelo com `joblib`

No seu notebook/script de treino, ao final, salve o modelo assim:

```python
import joblib

joblib.dump(meu_modelo, "meu_modelo.joblib")
```

Isso gera um arquivo `.joblib` que pode ser enviado para a API.

### 2. Coloque o arquivo na pasta `models`

Crie a pasta `models` (se ela ainda não existir) dentro de `Aula8` e coloque seu `.joblib` lá, ou faça o upload direto pela interface web em `http://127.0.0.1:8000` (botão de upload).

### 3. Ajuste a entrada de dados (`DadosEntrada`)

O modelo padrão espera uma lista de números (`list[float]`), em `main.py`:

```python
class DadosEntrada(BaseModel):
    valores: list[float]
```

Se o seu modelo espera outro formato de entrada (por exemplo, várias colunas nomeadas, texto, etc.), altere essa classe. Exemplo, para um modelo que espera `idade`, `salario` e `tempo_empresa`:

```python
class DadosEntrada(BaseModel):
    idade: float
    salario: float
    tempo_empresa: float
```

E ajuste a função `prever` para montar a entrada corretamente:

```python
@app.post("/prever")
def prever(dados: DadosEntrada):
    entrada = [[dados.idade, dados.salario, dados.tempo_empresa]]
    resultado = modelo_atual["objeto"].predict(entrada)
    ...
```

### 4. Ajuste a saída, se precisar

Se o seu modelo faz classificação e você quer devolver o nome da classe (e não só o número), trate o resultado antes de retornar:

```python
previsao = resultado.tolist()
# exemplo: mapear 0/1 para rótulos
rotulos = {0: "não", 1: "sim"}
previsao = [rotulos[p] for p in previsao]
```

### 5. Teste a previsão

Com o servidor rodando, envie uma requisição para `/prever`. Exemplo com `curl`:

```bash
curl -X POST http://127.0.0.1:8000/prever \
  -H "Content-Type: application/json" \
  -d "{\"valores\": [1.0, 2.0, 3.0]}"
```

Ou ajuste o corpo do JSON conforme os campos que você definiu em `DadosEntrada`.

## Resumo do que normalmente muda por aluno

| O que muda | Onde alterar |
|---|---|
| Modelo treinado (`.joblib`) | Pasta `models/` (via upload ou copiando o arquivo) |
| Formato dos dados de entrada | Classe `DadosEntrada` em `main.py` |
| Como a entrada é montada para o `.predict()` | Função `prever()` em `main.py` |
| Formato da resposta (ex.: rótulos de classe) | Função `prever()` em `main.py` |

O restante da API (upload de modelo, troca de modelo, listagem de modelos) já funciona sem precisar de alterações.
