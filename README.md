# Diferenciação entre SJS/TEN e CADRs com Deep Learning

## Visão Geral
Este repositório apresenta um pipeline de aprendizado profundo para **diferenciar Síndrome de Stevens–Johnson (SJS) / Necrólise Epidérmica Tóxica (TEN)** de um grupo mais amplo de **reações cutâneas adversas a medicamentos (CADRs)**.

O grupo CADR deste estudo inclui:
- Pênfigo vulgar
- Eritema multiforme
- Penfigoide bolhoso

O principal objetivo é investigar a viabilidade do uso de redes neurais convolucionais para apoiar o **diagnóstico diferencial precoce de reações dermatológicas graves induzidas por fármacos**, com foco na distinção entre **SJS/TEN** e doenças visualmente semelhantes.

---

## Preparação do Dataset
A construção do dataset envolveu um processo cuidadoso de curadoria em múltiplas etapas.

### Fontes dos Dados
As imagens foram coletadas de dois grandes bancos dermatológicos:
- **Derm1M**
- **Mendeley Data Skin Disease Classification Dataset**

### 1) Filtragem por Confiabilidade das Fontes
A primeira etapa consistiu em filtrar o **Derm1M** para manter apenas imagens provenientes das **fontes clínicas mais confiáveis**, melhorando a qualidade do dataset e reduzindo ruído nos rótulos.

### 2) Remoção de Duplicatas com FHash
Em seguida, foi aplicada uma etapa de detecção de duplicatas usando **FHash (perceptual hashing)** para identificar:
- imagens duplicadas **entre Derm1M e Mendeley**
- imagens duplicadas **dentro do próprio dataset**

### 3) Refinamento Focado na Lesão com LabelMe
A ferramenta **LabelMe** foi utilizada para:
- focalizar as imagens nas regiões das lesões
- recortar áreas clinicamente relevantes
- extrair **múltiplas visualizações centradas na lesão de um mesmo paciente**, quando apropriado

### 4) Controle de Qualidade
Por fim, imagens de baixa qualidade foram descartadas.

### Tamanho Final do Dataset
Após a curadoria, o dataset final ficou com:
- **480 imagens de SJS/TEN**
- **861 imagens de CADR**

---

## Protocolo Experimental
### Validação Cruzada
A estratégia de treinamento utiliza **validação cruzada 6-fold**.

Cada fold utiliza:
- **5 folds para treino**
- **1 fold para validação**

---

## Data Augmentation
Foram utilizadas estratégias de aumento de dados específicas por classe para melhorar o balanceamento e a generalização.

### SJS/TEN
Transformações aplicadas:
- **3 rotações**
- **1 flip horizontal**

### CADR
Transformações aplicadas:
- **1 rotação**
- **1 flip horizontal**

Essa estratégia foi planejada para que **cada fold de treino tenha aproximadamente 2000 imagens por classe**, promovendo um treinamento mais balanceado.

---

## Arquitetura do Modelo
O modelo base utilizado neste repositório é:
- **DenseNet-201**

### Próximos Modelos
As próximas etapas experimentais incluirão arquiteturas mais recentes:
- **ResNet (variantes recentes)**
- **EfficientNet**

Esses modelos serão comparados com a DenseNet-201 para avaliar ganhos de desempenho em sensibilidade e generalização.

---

## Resultados Atuais
O repositório atualmente inclui os **resultados experimentais com DenseNet-201**.

### Desempenho da DenseNet-201 (Validação Cruzada 6-Fold)
- **Acurácia:** 87,1 ± 2,4%
- **Sensibilidade (SJS):** 77,8 ± 4,1%
- **Especificidade:** 91,9 ± 2,7%
- **Precisão:** 85,4 ± 3,7%
- **F1-score:** 81,3 ± 2,9%

Esses resultados indicam forte capacidade discriminativa, especialmente na redução de falsos positivos entre casos de CADR.

---

## Estrutura do Projeto
```text
project/
├── dataset/
│   ├── SJS/
│   │   ├── pasta1
│   │   ├── pasta2
│   │   ├── ...
│   │   └── pasta6
│   └── CADR/
│       ├── pasta1
│       ├── pasta2
│       ├── ...
│       └── pasta6
```

---


## Relevância Científica
Este projeto faz parte de uma **Iniciação Científica (IC)** voltada à aplicação de inteligência artificial em análise de imagens dermatológicas.

A motivação clínica é apoiar o diagnóstico diferencial de casos **potencialmente fatais de SJS/TEN**, reduzindo confusão com outros CADRs menos graves.

---
