"""
Adapter para modelos de Machine Learning usando TensorFlow/Keras.

Import do TensorFlow é feito de forma "lazy" dentro da classe para evitar
que a importação do módulo falhe ao apenas importar o pacote (ex: erros de
DLL no Windows). Se o TensorFlow não estiver disponível, o adapter irá
marcar-se como indisponível e levantar erros claros quando métodos forem
invocados.
"""

import os
import importlib
import traceback
from typing import Dict, Any, Optional
import numpy as np
from ...application.ports.secondary_ports import MLModelPort


class TensorFlowMLAdapter(MLModelPort):
    """Adapter para TensorFlow/Keras com importação lazy.

    Se o TensorFlow não puder ser importado, os métodos irão levantar
    RuntimeError com instruções de diagnóstico.
    """

    def __init__(self, config):
        self.config = config
        self.available = False
        self._tf = None
        self._keras = None
        self._keras_layers = None
        self._keras_callbacks = None
        self.import_error = None
        self.import_traceback = None

        # tentar importar tensorflow quando instanciado, mas não falhar aqui
        try:
            self._tf = importlib.import_module('tensorflow')
            self._keras = self._tf.keras
            # importar símbolos usados internamente via keras
            self._keras_models = importlib.import_module('tensorflow.keras.models')
            self._keras_layers = importlib.import_module('tensorflow.keras.layers')
            self._keras_optimizers = importlib.import_module('tensorflow.keras.optimizers')
            self._keras_callbacks = importlib.import_module('tensorflow.keras.callbacks')
            self.available = True
        except Exception as e:
            # não propagar a exceção aqui — deixar o sistema inicializar e
            # falhar apenas ao tentar treinar/predizer, com mensagem útil.
            self.available = False
            # store exception and traceback for diagnostics
            self.import_error = e
            self.import_traceback = traceback.format_exc()

    def _require_tf(self):
        if not self.available or self._tf is None:
            msg = (
                "TensorFlow não disponível no ambiente. Instale uma versão compatível do "
                "TensorFlow e as dependências de runtime (ex: Microsoft Visual C++). "
                "Veja: https://www.tensorflow.org/install/errors"
            )
            if getattr(self, 'import_error', None) is not None:
                msg += f"\nCausa original: {repr(self.import_error)}"
                # keep traceback short but helpful
                if getattr(self, 'import_traceback', None):
                    tb_lines = self.import_traceback.strip().splitlines()
                    # include only last 8 lines of traceback
                    msg += "\nTraceback (most recent call last):\n" + "\n".join(tb_lines[-8:])

            raise RuntimeError(msg)

    def create_model(self, architecture: str, hyperparameters: Dict[str, Any]):
        self._require_tf()
        tf = self._tf
        Sequential = self._keras_models.Sequential
        Conv1D = getattr(self._keras_layers, 'Conv1D')
        MaxPooling1D = getattr(self._keras_layers, 'MaxPooling1D')
        Flatten = getattr(self._keras_layers, 'Flatten')
        Dense = getattr(self._keras_layers, 'Dense')
        Dropout = getattr(self._keras_layers, 'Dropout')
        BatchNormalization = getattr(self._keras_layers, 'BatchNormalization')
        Adam = getattr(self._keras_optimizers, 'Adam')

        if architecture == "CNN_1D":
            # parâmetros padrão
            input_shape = hyperparameters.get('input_shape', (250, 16))
            conv_filters = hyperparameters.get('conv_filters', [64, 64, 128])
            conv_kernels = hyperparameters.get('conv_kernels', [3, 3, 3])
            dense_units = hyperparameters.get('dense_units', 512)
            dropout_rates = hyperparameters.get('dropout_rates', [0.25, 0.25, 0.5])
            num_classes = hyperparameters.get('num_classes', 2)

            model = Sequential()
            model.add(Conv1D(filters=conv_filters[0], kernel_size=conv_kernels[0], activation='relu', input_shape=input_shape))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(dropout_rates[0]))

            model.add(Conv1D(filters=conv_filters[1], kernel_size=conv_kernels[1], activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(dropout_rates[1]))

            model.add(Conv1D(filters=conv_filters[2], kernel_size=conv_kernels[2], activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(dropout_rates[1]))

            model.add(Flatten())
            model.add(Dense(dense_units, activation='relu'))
            model.add(Dropout(dropout_rates[2]))
            model.add(Dense(num_classes, activation='softmax'))

            model.compile(optimizer=Adam(learning_rate=hyperparameters.get('learning_rate', 0.001)),
                          loss='categorical_crossentropy', metrics=['accuracy'])

            return model

        elif architecture == "EEGNET":
            # implementação simplificada
            input_shape = hyperparameters.get('input_shape', (250, 16))
            num_classes = hyperparameters.get('num_classes', 2)

            model = Sequential()
            model.add(Conv1D(filters=16, kernel_size=64, padding='same', input_shape=input_shape))
            model.add(BatchNormalization())
            model.add(Conv1D(filters=32, kernel_size=1))
            model.add(BatchNormalization())
            model.add(self._keras.layers.Activation('elu'))
            model.add(MaxPooling1D(pool_size=4))
            model.add(Dropout(0.25))

            model.add(Conv1D(filters=32, kernel_size=16, padding='same', groups=32))
            model.add(Conv1D(filters=32, kernel_size=1))
            model.add(BatchNormalization())
            model.add(self._keras.layers.Activation('elu'))
            model.add(MaxPooling1D(pool_size=8))
            model.add(Dropout(0.25))

            model.add(Flatten())
            model.add(Dense(num_classes, activation='softmax'))

            model.compile(optimizer=Adam(learning_rate=hyperparameters.get('learning_rate', 0.001)),
                          loss='categorical_crossentropy', metrics=['accuracy'])

            return model

        else:
            raise ValueError(f"Arquitetura não suportada: {architecture}")

    def train_model(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        self._require_tf()
        tf = self._tf
        # converte labels
        y_categorical = tf.keras.utils.to_categorical(y, num_classes=2)

        # reshape se necessário
        if len(X.shape) == 2:
            samples, features = X.shape
            channels = 16
            time_steps = features // channels
            X = X.reshape(samples, time_steps, channels)

        callbacks = [
            self._keras_callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            self._keras_callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        ]

        import time
        start_time = time.time()
        history = model.fit(X, y_categorical, epochs=100, batch_size=32, validation_split=0.2, callbacks=callbacks, verbose=1)
        training_time = time.time() - start_time

        epochs_trained = len(history.history.get('loss', []))

        return {
            'history': history.history,
            'final_loss': history.history.get('loss', [None])[-1],
            'final_accuracy': history.history.get('accuracy', [None])[-1],
            'val_loss': history.history.get('val_loss', [None])[-1],
            'val_accuracy': history.history.get('val_accuracy', [None])[-1],
            'training_time': training_time,
            'epochs_trained': epochs_trained
        }

    def predict(self, model, X: np.ndarray) -> np.ndarray:
        self._require_tf()
        if len(X.shape) == 2:
            samples, features = X.shape
            channels = 16
            time_steps = features // channels
            X = X.reshape(samples, time_steps, channels)

        predictions = model.predict(X, verbose=0)
        return np.argmax(predictions, axis=1)

    def predict_proba(self, model, X: np.ndarray) -> np.ndarray:
        self._require_tf()
        if len(X.shape) == 2:
            samples, features = X.shape
            channels = 16
            time_steps = features // channels
            X = X.reshape(samples, time_steps, channels)

        return model.predict(X, verbose=0)

    def evaluate_model(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        self._require_tf()
        tf = self._tf
        y_categorical = tf.keras.utils.to_categorical(y, num_classes=2)

        if len(X.shape) == 2:
            samples, features = X.shape
            channels = 16
            time_steps = features // channels
            X = X.reshape(samples, time_steps, channels)

        loss, accuracy = model.evaluate(X, y_categorical, verbose=0)
        y_pred = self.predict(model, X)

        from sklearn.metrics import precision_score, recall_score, f1_score
        return {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0)
        }

    def save_model(self, model, file_path: str) -> bool:
        self._require_tf()
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            model.save(file_path)
            return True
        except Exception as e:
            print(f"Erro ao salvar modelo: {str(e)}")
            return False

    def load_model(self, file_path: str):
        self._require_tf()
        try:
            if os.path.exists(file_path):
                return self._keras_models.load_model(file_path)
            else:
                print(f"Arquivo não encontrado: {file_path}")
                return None
        except Exception as e:
            print(f"Erro ao carregar modelo: {str(e)}")
            return None
