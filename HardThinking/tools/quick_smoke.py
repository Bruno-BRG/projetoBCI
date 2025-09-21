import os
import sys

# Insere o diretório raiz do projeto no PYTHONPATH para permitir imports relativos ao pacote `src`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.application.use_cases.train_model_use_case import TrainModelUseCase, TrainModelRequest
from src.infrastructure.adapters.tensorflow_ml_adapter import TensorFlowMLAdapter
from src.domain.value_objects.training_types import TrainingStrategy
from src.domain.entities.model import ModelArchitecture

class DummyRepo:
    def get_eeg_data_by_subject(self, subject_id):
        return []

class DummyModelRepo:
    def save_model(self, model):
        return True

class DummySubjectRepo:
    pass

class DummyLogging:
    def log_info(self, msg):
        print('INFO:', msg)
    def log_error(self, msg, e=None):
        print('ERROR:', msg, e)

class DummyNotification:
    def notify_training_started(self, model_id, subjects):
        print('notify started')
    def notify_training_completed(self, model_id, metrics):
        print('notify completed')
    def notify_training_failed(self, model_id, error_msg):
        print('notify failed')

if __name__ == '__main__':
    ml_adapter = TensorFlowMLAdapter(config={})
    use_case = TrainModelUseCase(
        eeg_repository=DummyRepo(),
        model_repository=DummyModelRepo(),
        subject_repository=DummySubjectRepo(),
        ml_port=ml_adapter,
        logging_port=DummyLogging(),
        notification_port=DummyNotification()
    )

    req = TrainModelRequest(
        subject_ids=['S001'],
        strategy=TrainingStrategy.SINGLE_SUBJECT,
        model_architecture=ModelArchitecture.CNN_1D
    )
    try:
        resp = use_case.execute(req)
        print('Executed, success=', resp.success)
    except Exception as e:
        print('Exception during execute:', e)
