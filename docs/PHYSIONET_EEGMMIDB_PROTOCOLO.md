# PhysioNet EEGMMIDB - mapeamento de runs

Fonte: PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0.

O dataset reutiliza os eventos `T0`, `T1` e `T2`, mas o significado de `T1`
e `T2` depende do run `Rxx`. Para o modelo BrainBridge de esquerda vs direita,
somente os runs unilaterais de punho devem entrar no treino generalizado.

## Usar para esquerda vs direita

| Runs | Tipo | T1 | T2 | Uso no BrainBridge |
| --- | --- | --- | --- | --- |
| R03, R07, R11 | Movimento real | mao esquerda | mao direita | Incluir |
| R04, R08, R12 | Movimento imaginado | mao esquerda | mao direita | Incluir |

## Nao usar nesse modelo

| Runs | Tipo | T1 | T2 | Motivo |
| --- | --- | --- | --- | --- |
| R01 | Baseline olhos abertos | n/a | n/a | Sem tarefa motora |
| R02 | Baseline olhos fechados | n/a | n/a | Sem tarefa motora |
| R05, R09, R13 | Movimento real | ambas as maos | ambos os pes | Nao representa esquerda/direita |
| R06, R10, R14 | Movimento imaginado | ambas as maos | ambos os pes | Nao representa esquerda/direita |

## Regra implementada

O modulo `brainbridge_v2.infrastructure.ml.physionet_eegmmidb_protocol`
centraliza essa tabela. O treino generalizado usa `left_right_only=True` por
padrao; quando o nome do arquivo segue `SxxxRxx`, runs que nao sao
esquerda/direita sao ignorados automaticamente.

Arquivos de pacientes reais sem padrao `SxxxRxx` continuam aceitos normalmente.
