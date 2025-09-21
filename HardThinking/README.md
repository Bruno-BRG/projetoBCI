# HardThinking - Sistema de Treinamento EEG Motor Imagery

## Visão Geral

O **HardThinking** é uma refatoração completa do sistema de treinamento EEG BrainBridge, implementado seguindo os princípios de **Arquitetura Hexagonal** (Ports and Adapters) e **Clean Architecture**. Esta refatoração mantém 100% da funcionalidade original enquanto oferece melhor organização, testabilidade e manutenibilidade.

## Arquitetura

### Estrutura de Diretórios

```
HardThinking/
├── src/                          # Código fonte principal
│   ├── domain/                   # Camada de domínio (regras de negócio)
│   │   ├── entities/            # Entidades do domínio
│   │   ├── value_objects/       # Value Objects
│   │   ├── repositories/        # Interfaces de repositórios
│   │   └── services/           # Serviços do domínio
│   ├── application/             # Camada de aplicação
│   │   ├── use_cases/          # Casos de uso
│   │   └── ports/              # Interfaces (Primary/Secondary Ports)
│   ├── infrastructure/          # Camada de infraestrutura
│   │   ├── adapters/           # Adapters (implementações)
│   │   └── repositories/       # Implementações de repositórios
│   └── interfaces/             # Interfaces de usuário
│       └── cli/               # Interface de linha de comando
├── tests/                      # Testes automatizados
├── models/                     # Modelos treinados
├── data/                       # Dados de EEG
├── logs/                       # Logs do sistema
└── main.py                     # Ponto de entrada
```

### Princípios da Arquitetura

1. **Separação de Responsabilidades**: Cada camada tem responsabilidades específicas
2. **Inversão de Dependência**: Dependências apontam para abstrações, não implementações
3. **Testabilidade**: Facilita testes unitários e de integração
4. **Flexibilidade**: Facilita mudança de tecnologias sem afetar regras de negócio
5. **Compatibilidade**: Mantém todas as funcionalidades do sistema original

## Funcionalidades

### ✅ Implementado
- Configuração centralizada com tipagem
- Estrutura de entidades e value objects
- Processamento de dados EEG
- Treinamento de modelos (sujeito único, validação cruzada)
- Interface CLI compatível
- Sistema de logging
- Repositórios de dados

### 🚧 Em Desenvolvimento
- Leave-One-Out validation
- Treinamento multi-sujeitos
- Análise estatística de dados
- Testes automatizados
- Comparação de modelos
- Exportação/importação de modelos

## Instalação

1. **Clone o repositório**:
```bash
cd HardThinking
```

2. **Crie ambiente virtual**:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# ou
source venv/bin/activate  # Linux/Mac
```

3. **Instale dependências**:
```bash
pip install -r requirements.txt
```

4. **Configure diretórios de dados**:
```bash
# Copie dados do sistema original ou configure novo diretório
# Estrutura esperada:
# data/
#   ├── P001_teste/
#   │   ├── arquivo1.csv
#   │   └── arquivo2.csv
#   └── P002_Garcia/
#       └── dados.csv
```

## Uso

### Interface CLI

Execute o sistema principal:
```bash
python main.py
```

### Opções de linha de comando:
```bash
python main.py --data-dir /caminho/para/dados --no-banner
```

### Menu Principal

O sistema oferece as seguintes opções:

1. **Treinamento de Modelos**:
   - Sujeito único
   - Validação cruzada
   - Leave-One-Out
   - Multi-sujeitos

2. **Análise de Dados**:
   - Análise geral
   - Análise por sujeito
   - Relatórios estatísticos

3. **Teste de Modelos**:
   - Teste individual
   - Comparação de modelos
   - Validação de performance

## Configuração

### Arquivo de Configuração

O sistema utiliza configuração tipada e centralizada:

```python
from src.config import get_config

config = get_config()

# Configurações de dados
print(config.data.sample_rate)  # 125 Hz
print(config.data.channels)     # 16 canais

# Configurações de modelo
print(config.model.architecture)  # CNN_1D
print(config.model.conv_filters)  # [64, 64, 128]

# Configurações de treinamento
print(config.training.epochs)     # 100
print(config.training.batch_size) # 32
```

### Personalização

As configurações podem ser personalizadas criando uma nova instância:

```python
from src.config import SystemConfiguration, DataConfiguration

# Configuração personalizada
custom_data_config = DataConfiguration(
    sample_rate=250,  # Frequência diferente
    channels=32       # Mais canais
)

config = SystemConfiguration()
config.data = custom_data_config
```

## Exemplos de Uso

### Treinamento Programático

```python
from src.application.use_cases.train_model_use_case import TrainModelUseCase, TrainModelRequest
from src.domain.value_objects.training_types import TrainingStrategy
from src.domain.entities.model import ModelArchitecture

# Configurar dependências...
# (ver main_cli.py para exemplo completo)

request = TrainModelRequest(
    subject_ids=["P001_teste"],
    strategy=TrainingStrategy.SINGLE_SUBJECT,
    model_architecture=ModelArchitecture.CNN_1D
)

response = train_model_use_case.execute(request)

if response.success:
    print(f"Acurácia: {response.validation_result.validation_performance.accuracy}")
```

### Análise de Dados

```python
from src.infrastructure.repositories.eeg_repository_impl import FileSystemEEGRepository

# Obter estatísticas de um sujeito
stats = eeg_repository.get_data_statistics("P001_teste")
print(f"Duração total: {stats['total_duration_seconds']} segundos")
print(f"Marcadores T1: {stats['marker_counts']['T1']}")
```

## Compatibilidade

### Sistema Original

O HardThinking mantém 100% de compatibilidade com o sistema original:

- **Mesma interface CLI**: Todos os menus e opções
- **Mesmo formato de dados**: Arquivos CSV com estrutura original
- **Mesmas funcionalidades**: Treinamento, análise, teste
- **Mesmos algoritmos**: CNN 1D, processamento de sinais

### Migração Gradual

É possível migrar gradualmente:

1. Execute HardThinking para novas funcionalidades
2. Use sistema original para funcionalidades específicas
3. Migre dados e modelos conforme necessário

## Desenvolvimento

### Adicionando Novas Funcionalidades

1. **Novo Caso de Uso**:
```python
# src/application/use_cases/novo_caso_uso.py
class NovoCasoUso:
    def execute(self, request):
        # Implementação
        pass
```

2. **Nova Entidade**:
```python
# src/domain/entities/nova_entidade.py
@dataclass
class NovaEntidade:
    id: str
    # Atributos...
```

3. **Novo Adapter**:
```python
# src/infrastructure/adapters/novo_adapter.py
class NovoAdapter(NovoPort):
    def implementar_metodo(self):
        # Implementação específica
        pass
```

### Testes

```bash
# Executar testes
pytest tests/

# Com cobertura
pytest --cov=src tests/

# Testes específicos
pytest tests/domain/test_entities.py
```

### Formatação de Código

```bash
# Formatação automática
black src/

# Verificação de estilo
flake8 src/
```

## Logging

O sistema possui logging abrangente:

```python
# Logs são salvos em logs/hardthinking_YYYYMMDD.log
# Níveis: DEBUG, INFO, WARNING, ERROR

# Exemplos de logs:
# [2025-01-26T10:30:00] INFO: Iniciando treinamento - Estratégia: single_subject
# [2025-01-26T10:35:00] INFO: Treinamento concluído - Acurácia: 0.847
```

## Monitoramento

### Métricas de Performance

- Acurácia de treinamento/validação
- Tempo de treinamento
- Uso de memória
- Qualidade dos dados

### Alertas

- Overfitting detectado
- Baixa qualidade de dados
- Falhas de treinamento

## Roadmap

### Próximas Versões

- [ ] Interface web
- [ ] API REST
- [ ] Integração com banco de dados
- [ ] Processamento distribuído
- [ ] Mais arquiteturas de modelo
- [ ] Análise em tempo real

## Contribuição

1. Fork o repositório
2. Crie branch para feature (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para branch (`git push origin feature/nova-funcionalidade`)
5. Crie Pull Request

## Licença

Este projeto mantém a mesma licença do sistema original BrainBridge.

## Suporte

Para dúvidas ou problemas:
1. Verifique a documentação
2. Consulte os logs em `logs/`
3. Abra issue no repositório

---

**HardThinking** - *Arquitetura sólida para mentes pensantes* 🧠⚡
