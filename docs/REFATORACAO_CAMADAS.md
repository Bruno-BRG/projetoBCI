# Refatoracao de Camadas (Concluida - Etapa Atual)

## Estrutura final ativa

Dentro de `brainbridge_v2/`:

```text
domain/
  entities/
application/
  ports/
  use_cases/
infrastructure/
  acquisition/
  communication/
  config/
  database/
  ml/
  signal_processing/
  repositories/
interface_adapters/
  controllers/
presentation/
  gui/
```

## Migracao aplicada

Fluxo de pacientes:

`PatientRegistrationWidget (presentation/gui)`
-> `PatientController` (interface adapters)
-> `RegisterPatientUseCase` / `ListPatientsUseCase` (application)
-> `SQLitePatientRepository` (infrastructure)
-> `DatabaseManager` (infrastructure/database)

Fluxo principal de app:

`presentation/main.py`
-> `presentation/gui/main_window.py`
-> widgets de interface em `presentation/gui/widgets`
-> adaptadores/controladores em camadas adequadas

## Legado removido

Diretorios legados removidos:

- `brainbridge_v2/core/`
- `brainbridge_v2/database/`
- `brainbridge_v2/utils/`
- `brainbridge_v2/gui/` (substituido por `presentation/gui`)
- `brainbridge_v2/acquisition/` (movido para `infrastructure/acquisition`)
- `brainbridge_v2/communication/` (movido para `infrastructure/communication`)
- `brainbridge_v2/ml/` (movido para `infrastructure/ml`)
- `brainbridge_v2/processing/` (movido para `infrastructure/signal_processing`)
- `brainbridge_v2/config/` (movido para `infrastructure/config`)
- `brainbridge_v2/main.py` (movido para `presentation/main.py`)

## Arquivos principais novos

- `brainbridge_v2/domain/entities/patient.py`
- `brainbridge_v2/application/ports/patient_repository.py`
- `brainbridge_v2/application/use_cases/patient_use_cases.py`
- `brainbridge_v2/infrastructure/repositories/sqlite_patient_repository.py`
- `brainbridge_v2/interface_adapters/controllers/patient_controller.py`

## Status de validacao

- `pytest`: `131 passed`
