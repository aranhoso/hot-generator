# Modelo de Geração de HotWheels

![Última Atualização](https://img.shields.io/badge/atualizado-2024--02--17-blue)
![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-latest-red)

Modelo de rede neural (Multi-Layer Perceptron) treinado para gerar descrições de carros HotWheels com base em um dataset de mais de 55 mil miniaturas.

## Visão Geral

O modelo utiliza uma arquitetura MLP (Multi-Layer Perceptron) com camada de embedding para processar e gerar descrições de carros HotWheels, incluindo:
- Nome do modelo
- Ano de lançamento
- Cor
- Série

## Dataset

- **Fonte**: Dados extraídos via web scraping do site collecthw.com
- **Tamanho**: 55.579 registros
- **Campos**: Model Name, Release Year, Color, Series
- **Formato**: Dados processados e tokenizados em formato XML

## Arquitetura do Modelo

### Parâmetros
- Tamanho do contexto: 32 tokens
- Dimensão do embedding: 32
- Dimensão oculta: 128
- Taxa de dropout: 0.2
- Tamanho do vocabulário: 104 tokens

### Estrutura MLP
1. Camada de Embedding (104 → 32)
2. Primeira Camada Linear (1024 → 128)
3. Layer Normalization
4. Ativação GELU
5. Dropout (0.2)
6. Segunda Camada Linear (128 → 104)

### Métricas Alcançadas
- Acurácia de treino: ~90%
- Acurácia de validação: ~85%
- Loss de treino: ~0.3
- Loss de validação: ~0.4

## Uso do Modelo

```
import torch
from model import HotWheelsLanguageModel

# Carregar o modelo
checkpoint = torch.load("model_complete.pt")
model = HotWheelsLanguageModel(
    context_length=32,
    vocab_size=104,
    embedding_dim=32,
    hidden_dim=128
)
model.load_state_dict(checkpoint['model_state_dict'])

# Configuração do modelo
model.eval()
```

## Requisitos

<li>Python 3.12+</li>
<li>PyTorch</li>
<li>NumPy</li>

# Instale as dependências 

pip install -r requirements.txt

## Autor
Criado por [@aranhoso](https://github.com/aranhoso)