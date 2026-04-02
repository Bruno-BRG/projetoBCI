# Roadmap de Refatoracao

Ultima atualizacao: `2026-04-01`

## Objetivo
- tirar regra de negocio da GUI
- consolidar Clean Architecture de verdade
- reduzir o tamanho e o acoplamento do `streaming.py`
- fechar a migracao sem deixar legado escondido

## Modo de trabalho
- trabalhar por cortes pequenos e verificaveis
- cada tarefa so fecha com codigo ajustado, teste minimo e roadmap atualizado
- priorizar o que reduz acoplamento antes de cosmética
- evitar big bang; sempre migrar por fluxo funcional

## Status geral
- `Fase 2`: em progresso
- `Fase 3`: em progresso
- `Fase 4`: em progresso
- `Fase 5`: em progresso
- `Fase 6`: em progresso
- `Fase 7`: em progresso
- `Fase 8`: em progresso
- `Fase 9`: pendente
- `Fase 10`: em progresso

## Sprint atual
- [x] extrair `accuracy_presenter`
- [x] consolidar o restante do estado visual do `StreamingWidget`
- [x] revisar a UI e manter a composicao centralizada no `bootstrap`
- [ ] revisar caminhos de assets/modelos para refletir `brainbridge_v2/resources`
- [ ] decidir o papel de `resources` vs `data`

## Epic 1: Arquitetura base
- [x] criar slices de `recordings`, `sessions`, `markers`, `eeg`, `unity`, `esp32`
- [x] criar slices de `inference` e `training`
- [x] mover composicao para `bootstrap/container.py`
- [x] injetar controllers na UI
- [x] revisar pontos restantes de composition root improvisado

## Epic 2: StreamingWidget
- [x] remover acesso direto ao banco
- [x] remover import direto de TensorFlow
- [x] mover fluxo de inferencia para controller/use case/gateway
- [x] mover fluxo de treino para controller/use case/gateway
- [x] introduzir `StreamingSessionStateViewModel`
- [x] extrair `accuracy_presenter`
- [x] extrair estado visual restante de conexao e jogo
- [x] reduzir mais responsabilidades do `streaming.py`

## Epic 3: DTOs e Presenters
- [x] criar `StartRecordingRequest`
- [x] criar `StartSessionRequest`
- [x] criar view models tipados para `recording`
- [x] criar view models tipados para `session`
- [x] criar view models tipados para `markers`
- [x] criar view models tipados para `inference`
- [x] criar view models tipados para `training`
- [ ] revisar onde ainda sobra `dict` cru fora desses fluxos

## Epic 4: Persistencia e infraestrutura
- [x] criar `SQLiteRecordingRepository`
- [ ] quebrar `DatabaseManager`
- [ ] criar `SQLiteConnectionFactory`
- [ ] criar `SQLiteSchemaManager`
- [ ] criar `SQLiteSessionRepository`
- [ ] revisar estrutura final de `infrastructure/database`

## Epic 5: Resources e convencoes
- [x] criar `brainbridge_v2/resources`
- [x] reorganizar HTML, modelos e exemplos dentro de `resources`
- [ ] mapear quem ainda le caminho antigo
- [ ] alinhar discovery de modelos com `resources/models`
- [ ] alinhar assets visuais com `resources`
- [ ] decidir o que e asset versionado e o que e dado gerado em runtime

## Epic 6: Limpeza
- [x] remover wrappers legados obvios
- [x] remover arquivos vazios sem uso
- [x] consolidar `settings.py`
- [ ] remover imports mortos restantes
- [ ] revisar arquivos duplicados ou renomeados
- [ ] revisar nomes de modulos para consistencia

## Epic 7: Qualidade
- [x] criar testes unitarios para slices novos
- [x] criar testes para presenters principais
- [ ] criar testes de arquitetura para imports proibidos
- [ ] criar testes de contrato para gateways e repositories
- [ ] criar validacao de fluxo manual ponta a ponta

## Proximos passos sugeridos
1. ajustar runtime para a nova pasta `resources`
2. atacar `DatabaseManager`
3. criar testes de arquitetura
4. revisar onde ainda sobra `dict` cru
5. fazer passada final de limpeza

## Definicao de pronto
- nenhum widget importa infraestrutura concreta
- nenhum fluxo principal depende de sqlite, TensorFlow, socket ou serial dentro da GUI
- os fluxos centrais passam por use cases
- `resources`, `data` e `models` têm papeis claros
- arquitetura fica protegida por testes
