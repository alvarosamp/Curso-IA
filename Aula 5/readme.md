# Módulo 5: Visão Computacional

**Carga Horária:** 12 horas  
**Nível:** Graduação/Pós-graduação  
**Pré-requisitos:** Módulos anteriores (Python, ML, Deep Learning básico). Conhecimentos de redes neurais e TensorFlow/PyTorch são desejáveis.

---

## 📋 Objetivos

Ao final deste módulo, você será capaz de:

- ✅ Compreender os fundamentos da Visão Computacional e suas aplicações
- ✅ Dominar técnicas básicas de processamento de imagens com OpenCV
- ✅ Entender o funcionamento das Redes Neurais Convolucionais (CNNs)
- ✅ Aplicar Transfer Learning para classificação e detecção
- ✅ Utilizar o Roboflow para gerenciamento de datasets
- ✅ Treinar um modelo de detecção de objetos com YOLOv8 em um dataset customizado

---

## 📚 Sumário

### 1. Fundamentos de Visão Computacional
- 1.1 O que é Visão Computacional?
- 1.2 Aplicações
- 1.3 Desafios

### 2. Processamento de Imagens com OpenCV
- 2.1 Introdução ao OpenCV
- 2.2 Leitura e exibição de imagens
- 2.3 Redimensionamento
- 2.4 Normalização
- 2.5 Conversão de cores
- 2.6 Operações básicas (borramento, threshold)

### 3. Redes Neurais Convolucionais (CNNs)
- 3.1 Por que CNNs para imagens?
- 3.2 Camadas convolucionais
- 3.3 Pooling
- 3.4 Arquiteturas clássicas (LeNet, AlexNet, VGG, ResNet)

### 4. Transfer Learning
- 4.1 Conceito e motivação
- 4.2 Estratégias: fine-tuning vs feature extraction
- 4.3 Exemplo com TensorFlow/Keras

### 5. Avaliação de Modelos de Visão
- 5.1 Métricas para classificação
- 5.2 Métricas para detecção: IoU, Precision, Recall, mAP
- 5.3 Curva Precision-Recall

### 6. Introdução ao Roboflow
- 6.1 O que é Roboflow?
- 6.2 Roboflow Universe: datasets públicos
- 6.3 Exportando datasets para diferentes formatos

### 7. Detecção de Objetos com YOLO
- 7.1 O que é YOLO? (You Only Look Once)
- 7.2 Evolução: YOLOv8 e YOLOv11
- 7.3 Instalação da biblioteca Ultralytics
- 7.4 Treinamento com um dataset customizado
- 7.5 Inferência e avaliação

### 8. Atividade Prática
- 8.1 Objetivo
- 8.2 Passo a passo
- 8.3 Entrega

### 9. Conclusão e Próximos Passos

### 10. Referências

---

## 1️⃣ Fundamentos de Visão Computacional

### 1.1 O que é Visão Computacional?

**Visão Computacional** é um campo da inteligência artificial que capacita máquinas a interpretar e entender o mundo visual a partir de imagens ou vídeos. O objetivo é replicar a capacidade humana de reconhecer objetos, cenas e extrair informações úteis.

**Relação com outros campos:**

- **Processamento de imagens:** Transformações de imagem para melhorar qualidade ou extrair características
- **Machine Learning:** Algoritmos que aprendem a partir de dados visuais
- **Deep Learning:** Redes neurais profundas, especialmente CNNs, que revolucionaram a área

### 1.2 Aplicações

| Área | Exemplos |
|------|----------|
| **Saúde** | Diagnóstico por imagem (raios-X, tomografias), detecção de tumores |
| **Automotiva** | Carros autônomos, detecção de pedestres, leitura de placas |
| **Segurança** | Reconhecimento facial, vigilância por vídeo |
| **Varejo** | Checkout automático, análise de comportamento do cliente |
| **Agricultura** | Monitoramento de plantações, detecção de pragas |
| **Indústria** | Inspeção de qualidade, robótica |

### 1.3 Desafios

- **Variabilidade intraclasse:** Um mesmo objeto pode ter aparências muito diferentes (ex: cadeiras de vários modelos)
- **Condições de iluminação:** Sombras, reflexos
- **Oclusões:** Objetos parcialmente escondidos
- **Escala e rotação:** Objetos podem aparecer em diferentes tamanhos e orientações
- **Background complexo:** Fundo confuso pode atrapalhar a detecção

---

## 2️⃣ Processamento de Imagens com OpenCV

### 2.1 Introdução ao OpenCV

**OpenCV (Open Source Computer Vision Library)** é uma biblioteca opensource amplamente utilizada para processamento de imagens e visão computacional. Oferece mais de 2.500 algoritmos otimizados.

**Principais funcionalidades:**
- Leitura e escrita de imagens e vídeos
- Transformações geométricas
- Filtros e operações morfológicas
- Detecção de features e objetos
- Integração com frameworks de Deep Learning

### 2.2 Leitura e Exibição de Imagens

```python
import cv2
import matplotlib.pyplot as plt

# Carregar imagem
img = cv2.imread('imagem.jpg')

# OpenCV lê em BGR, converter para RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Exibir
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
```

### 2.3 Operações Básicas

- **Redimensionamento:** Alterar dimensões da imagem
- **Normalização:** Escalar valores de pixel para [0, 1]
- **Conversão de cores:** BGR ↔ RGB ↔ Grayscale
- **Borramento:** Reduzir ruído (Gaussian blur, Median blur)
- **Threshold:** Binarização de imagens

---

## 3️⃣ Redes Neurais Convolucionais (CNNs)

### 3.1 Por que CNNs para Imagens?

As CNNs são especialmente projetadas para dados com estrutura espacial, como imagens. Diferentemente das redes totalmente conectadas (MLPs), as CNNs:

- **Preservam a estrutura espacial** dos dados
- **Compartilham parâmetros** (filtros convolucionais)
- **Capturam hierarquia de features:** bordas → texturas → partes → objetos

### 3.2 Camadas Convolucionais

Aplicam filtros (kernels) que deslizam pela imagem, detectando padrões locais:

- **Filtros de baixo nível:** Detectam bordas, cantos
- **Filtros de alto nível:** Detectam formas complexas, objetos

### 3.3 Pooling

Reduz a dimensionalidade espacial, mantendo as features mais importantes:

- **Max Pooling:** Seleciona o valor máximo em uma região
- **Average Pooling:** Calcula a média

### 3.4 Arquiteturas Clássicas

| Arquitetura | Ano | Características |
|-------------|-----|-----------------|
| **LeNet** | 1998 | Primeira CNN de sucesso (dígitos manuscritos) |
| **AlexNet** | 2012 | Vencedora do ImageNet, popularizou Deep Learning |
| **VGG** | 2014 | Arquitetura profunda com blocos repetidos |
| **ResNet** | 2015 | Conexões residuais, permite redes muito profundas |

---

## 4️⃣ Transfer Learning

### 4.1 Conceito e Motivação

**Transfer Learning** consiste em aproveitar conhecimento de um modelo pré-treinado em uma tarefa grande (ex: ImageNet) e adaptá-lo para uma nova tarefa com menos dados.

**Vantagens:**
- ✅ Economiza tempo e recursos computacionais
- ✅ Funciona bem com datasets pequenos
- ✅ Melhora a performance em tarefas relacionadas

### 4.2 Estratégias

1. **Feature Extraction:** Congela a base do modelo, treina apenas as últimas camadas
2. **Fine-Tuning:** Descongela algumas camadas e as ajusta para a nova tarefa

### 4.3 Exemplo com TensorFlow/Keras

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

# Carregar ResNet50 pré-treinada (sem o topo)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar a base
base_model.trainable = False

# Adicionar novas camadas
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 10 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

## 5️⃣ Avaliação de Modelos de Visão

### 5.1 Métricas para Classificação

- **Accuracy:** Porcentagem de acertos
- **Precision:** TP / (TP + FP)
- **Recall:** TP / (TP + FN)
- **F1-Score:** Harmônica entre Precision e Recall

### 5.2 Métricas para Detecção

- **IoU (Intersection over Union):** Mede a sobreposição entre bbox predita e real
- **mAP (mean Average Precision):** Média da AP de todas as classes
- **Precision-Recall Curve:** Avalia o trade-off entre precisão e cobertura

### 5.3 IoU (Intersection over Union)

```
IoU = Área da Interseção / Área da União
```

- IoU > 0.5 geralmente é considerado detecção correta

---

## 6️⃣ Introdução ao Roboflow

### 6.1 O que é Roboflow?

**Roboflow** é uma plataforma para gerenciamento de datasets de visão computacional. Oferece:

- Anotação de imagens
- Preprocessamento (resize, augmentation)
- Exportação para diversos formatos (YOLO, COCO, TensorFlow)
- Integração com frameworks populares

### 6.2 Roboflow Universe

**Roboflow Universe** é um repositório de datasets públicos para visão computacional. Você pode:

- Explorar milhares de datasets anotados
- Fazer download direto para treinamento
- Contribuir com seus próprios datasets

🔗 **Link:** [https://universe.roboflow.com/](https://universe.roboflow.com/)

### 6.3 Exportando Datasets

1. Escolha o dataset no Roboflow Universe
2. Selecione o formato de exportação (YOLO, COCO, etc.)
3. Gere um link de download ou use a API
4. Integre diretamente no código de treinamento

---

## 7️⃣ Detecção de Objetos com YOLO

### 7.1 O que é YOLO?

**YOLO (You Only Look Once)** é uma família de modelos de detecção de objetos em tempo real. Diferente de métodos tradicionais (R-CNN, Fast R-CNN), o YOLO:

- Prediz bounding boxes e classes em **uma única passagem**
- É extremamente rápido (ideal para aplicações em tempo real)
- Balanceia velocidade e precisão

### 7.2 Evolução: YOLOv8 e YOLOv11

| Versão | Ano | Características |
|--------|-----|-----------------|
| **YOLOv5** | 2020 | Primeira versão em PyTorch, muito popular |
| **YOLOv8** | 2023 | Ultralytics, melhora na arquitetura e performance |
| **YOLOv11** | 2024 | Última versão, ainda mais preciso e rápido |

### 7.3 Instalação da Biblioteca Ultralytics

```bash
pip install ultralytics
```

### 7.4 Treinamento com Dataset Customizado

```python
from ultralytics import YOLO

# Carregar modelo pré-treinado
model = YOLO('yolov8n.pt')  # nano (mais leve)

# Treinar
results = model.train(
    data='dataset.yaml',  # arquivo de configuração do dataset
    epochs=50,
    imgsz=640,
    batch=16,
    name='custom_yolov8'
)
```

### 7.5 Inferência e Avaliação

```python
# Validação
metrics = model.val()

# Inferência em novas imagens
results = model.predict('test_image.jpg', conf=0.5)

# Visualizar resultados
for r in results:
    r.show()
```

---

## 8️⃣ Atividade Prática

### 8.1 Objetivo

Treinar um modelo **YOLOv8** para detecção de objetos em um dataset customizado do **Roboflow Universe**.

### 8.2 Passo a Passo

1. **Escolher um dataset no Roboflow Universe**
   - Acesse [universe.roboflow.com](https://universe.roboflow.com/)
   - Escolha um dataset de interesse (ex: detecção de placas, veículos, animais)

2. **Exportar o dataset em formato YOLO**
   - Selecione a versão do dataset
   - Clique em "Download" → "YOLOv8"
   - Copie o código de download fornecido

3. **Configurar o ambiente no Google Colab**
   ```python
   !pip install ultralytics roboflow
   ```

4. **Baixar o dataset**
   ```python
   from roboflow import Roboflow
   rf = Roboflow(api_key="YOUR_API_KEY")
   project = rf.workspace("workspace-name").project("project-name")
   dataset = project.version(1).download("yolov8")
   ```

5. **Treinar o modelo**
   ```python
   from ultralytics import YOLO
   
   model = YOLO('yolov8n.pt')
   results = model.train(
       data=f'{dataset.location}/data.yaml',
       epochs=50,
       imgsz=640,
       batch=16
   )
   ```

6. **Avaliar e fazer predições**
   ```python
   # Validação
   metrics = model.val()
   
   # Inferência
   results = model.predict('test_image.jpg')
   ```

### 8.3 Entrega

**Formato:**

1. **Notebook no Google Colab** (`.ipynb`) com:
   - Células markdown explicando cada etapa
   - Código executado com resultados visíveis
   - Visualizações (curvas de treinamento, matriz de confusão)

2. **Análise dos resultados:**
   - mAP50, mAP50-95
   - Precision e Recall por classe
   - Exemplos de predições (corretas e incorretas)

3. **Conclusão:**
   - Desafios encontrados
   - Possíveis melhorias (data augmentation, hyperparameter tuning)
   - Aplicações práticas do modelo treinado

**Critérios de avaliação:**
- ✅ Escolha adequada do dataset
- ✅ Configuração correta do treinamento
- ✅ Qualidade das visualizações
- ✅ Análise crítica dos resultados
- ✅ Clareza na documentação

---

## 9️⃣ Conclusão e Próximos Passos

### O que você aprendeu?

✅ Fundamentos de Visão Computacional  
✅ Processamento de imagens com OpenCV  
✅ CNNs e suas arquiteturas  
✅ Transfer Learning para aproveitar modelos pré-treinados  
✅ Métricas de avaliação para detecção  
✅ Uso do Roboflow para gestão de datasets  
✅ Detecção de objetos com YOLOv8  

### Próximos Passos

**1. Tópicos Avançados em Visão Computacional:**
- Segmentação semântica (U-Net, Mask R-CNN)
- Detecção de keypoints e pose estimation
- Reconhecimento de ações em vídeo
- GANs para geração de imagens

**2. Otimização de Modelos:**
- Quantização e pruning
- Exportação para ONNX, TensorRT
- Deploy em edge devices (Raspberry Pi, Jetson Nano)

**3. Projetos Práticos:**
- Sistema de vigilância inteligente
- Contador de pessoas/veículos
- Assistente de estacionamento
- Inspeção de qualidade em manufatura

**4. Certificações e Competições:**
- [Kaggle Computer Vision Competitions](https://www.kaggle.com/competitions?search=computer+vision)
- [Roboflow Competitions](https://blog.roboflow.com/computer-vision-competitions/)

---

## 🔟 Referências

### Livros
- **"Deep Learning for Computer Vision"** - Rajalingappaa Shanmugamani
- **"Computer Vision: Algorithms and Applications"** - Richard Szeliski (disponível gratuitamente online)
- **"Modern Computer Vision with PyTorch"** - V Kishore Ayyadevara, Yeshwanth Reddy

### Cursos Online
- **Stanford CS231n:** Convolutional Neural Networks for Visual Recognition
- **Fast.ai:** Practical Deep Learning for Coders
- **Coursera:** Deep Learning Specialization (Andrew Ng)

### Documentação e Tutoriais
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [Roboflow Blog](https://blog.roboflow.com/)
- [OpenCV Tutorials](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
- [PyImageSearch](https://pyimagesearch.com/)

### Papers Importantes
- **AlexNet:** Krizhevsky et al. (2012) - ImageNet Classification with Deep CNNs
- **VGG:** Simonyan & Zisserman (2014) - Very Deep CNNs
- **ResNet:** He et al. (2015) - Deep Residual Learning
- **YOLO:** Redmon et al. (2016) - You Only Look Once
- **YOLOv8:** Ultralytics (2023) - YOLOv8 Technical Report

### Datasets Populares
- **ImageNet:** Classificação de imagens (1.4M imagens, 1000 classes)
- **COCO:** Detecção, segmentação (200k imagens, 80 classes)
- **Open Images:** Google (9M imagens, 600 classes)
- **Roboflow Universe:** Milhares de datasets customizados

### Comunidades
- [Reddit r/computervision](https://www.reddit.com/r/computervision/)
- [Ultralytics Community](https://community.ultralytics.com/)
- [Roboflow Forum](https://discuss.roboflow.com/)

---

## 📝 Observações Finais

Este módulo foi projetado para ser executado no **Google Colab** com GPU ativada (Runtime → Change runtime type → GPU). Para datasets maiores, considere usar Colab Pro ou ambientes locais com GPU.

**Estrutura do projeto recomendada:**
```
Aula 5/
├── readme.md                  # Este arquivo
├── notebook.ipynb             # Notebook principal com teoria e exemplos
├── atividade_pratica.ipynb    # Notebook para a atividade com YOLO
├── datasets/                  # Datasets baixados do Roboflow
└── results/                   # Resultados dos treinamentos
```

**Recursos adicionais:**
- Todos os códigos estão disponíveis e testados no Colab
- Links para datasets e modelos pré-treinados incluídos
- Suporte para dúvidas via issues no repositório do curso

---

**Bons estudos! 🚀**

*Última atualização: Março 2026*