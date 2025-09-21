"""
Adapter para sistema de arquivos
"""

import os
import pandas as pd
import numpy as np
import json
import glob
from pathlib import Path
from typing import Dict, Any, List, Optional
from ...application.ports.secondary_ports import FileSystemPort

class LocalFileSystemAdapter(FileSystemPort):
    """Adapter para sistema de arquivos local"""
    
    def __init__(self, base_path: str = ""):
        self.base_path = Path(base_path) if base_path else Path.cwd()
    
    def read_csv_file(self, file_path: str) -> Dict[str, Any]:
        """Lê arquivo CSV"""
        try:
            full_path = self._get_full_path(file_path)
            
            # Lê CSV pulando cabeçalho (compatível com formato do sistema original)
            df = pd.read_csv(full_path, skiprows=4)
            
            # Extrai dados EEG (colunas 1-16)
            eeg_data = df.iloc[:, 1:17].values.astype(float)
            
            # Extrai anotações (última coluna)
            annotations = df.iloc[:, -1].values.tolist()
            
            # Extrai informações do cabeçalho
            header_info = self._read_csv_header(full_path)
            
            return {
                'eeg_data': eeg_data,
                'annotations': annotations,
                'header_info': header_info,
                'shape': eeg_data.shape,
                'file_path': str(full_path)
            }
            
        except Exception as e:
            raise IOError(f"Erro ao ler arquivo CSV {file_path}: {str(e)}")
    
    def _read_csv_header(self, file_path: Path) -> Dict[str, str]:
        """Lê informações do cabeçalho do CSV"""
        header_info = {}
        try:
            with open(file_path, 'r') as f:
                for i in range(4):  # Primeiras 4 linhas são cabeçalho
                    line = f.readline().strip()
                    if ',' in line:
                        parts = line.split(',', 1)
                        if len(parts) == 2:
                            header_info[f"header_{i}"] = parts[1]
                    else:
                        header_info[f"header_{i}"] = line
        except:
            pass
        return header_info
    
    def write_csv_file(self, file_path: str, data: Dict[str, Any]) -> bool:
        """Escreve arquivo CSV"""
        try:
            full_path = self._get_full_path(file_path)
            
            # Cria diretório se não existe
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            if 'dataframe' in data:
                # Se já é um DataFrame
                data['dataframe'].to_csv(full_path, index=False)
            elif 'eeg_data' in data and 'annotations' in data:
                # Cria DataFrame dos dados EEG
                eeg_data = data['eeg_data']
                annotations = data['annotations']
                
                # Cria DataFrame
                df = pd.DataFrame(eeg_data)
                df['annotations'] = annotations
                
                # Adiciona cabeçalho se fornecido
                if 'header_info' in data:
                    with open(full_path, 'w') as f:
                        for key, value in data['header_info'].items():
                            f.write(f"{key},{value}\n")
                        df.to_csv(f, index=False, header=False)
                else:
                    df.to_csv(full_path, index=False)
            else:
                raise ValueError("Formato de dados não suportado")
            
            return True
            
        except Exception as e:
            print(f"Erro ao escrever arquivo CSV: {str(e)}")
            return False
    
    def list_files(self, directory: str, pattern: str = "*") -> List[str]:
        """Lista arquivos em diretório"""
        try:
            dir_path = self._get_full_path(directory)
            
            if not dir_path.exists():
                return []
            
            files = list(dir_path.glob(pattern))
            return [str(f.relative_to(self.base_path)) for f in files if f.is_file()]
            
        except Exception as e:
            print(f"Erro ao listar arquivos: {str(e)}")
            return []
    
    def list_directories(self, directory: str) -> List[str]:
        """Lista subdiretórios"""
        try:
            dir_path = self._get_full_path(directory)
            
            if not dir_path.exists():
                return []
            
            dirs = [d for d in dir_path.iterdir() if d.is_dir()]
            return [str(d.relative_to(self.base_path)) for d in dirs]
            
        except Exception as e:
            print(f"Erro ao listar diretórios: {str(e)}")
            return []
    
    def create_directory(self, directory_path: str) -> bool:
        """Cria diretório"""
        try:
            full_path = self._get_full_path(directory_path)
            full_path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"Erro ao criar diretório: {str(e)}")
            return False
    
    def file_exists(self, file_path: str) -> bool:
        """Verifica se arquivo existe"""
        try:
            full_path = self._get_full_path(file_path)
            return full_path.exists() and full_path.is_file()
        except:
            return False
    
    def directory_exists(self, directory_path: str) -> bool:
        """Verifica se diretório existe"""
        try:
            full_path = self._get_full_path(directory_path)
            return full_path.exists() and full_path.is_dir()
        except:
            return False
    
    def delete_file(self, file_path: str) -> bool:
        """Remove arquivo"""
        try:
            full_path = self._get_full_path(file_path)
            if full_path.exists():
                full_path.unlink()
                return True
            return False
        except Exception as e:
            print(f"Erro ao deletar arquivo: {str(e)}")
            return False
    
    def copy_file(self, source_path: str, destination_path: str) -> bool:
        """Copia arquivo"""
        try:
            import shutil
            source = self._get_full_path(source_path)
            destination = self._get_full_path(destination_path)
            
            # Cria diretório de destino se necessário
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(source, destination)
            return True
        except Exception as e:
            print(f"Erro ao copiar arquivo: {str(e)}")
            return False
    
    def read_json_file(self, file_path: str) -> Dict[str, Any]:
        """Lê arquivo JSON"""
        try:
            full_path = self._get_full_path(file_path)
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise IOError(f"Erro ao ler arquivo JSON {file_path}: {str(e)}")
    
    def write_json_file(self, file_path: str, data: Dict[str, Any]) -> bool:
        """Escreve arquivo JSON"""
        try:
            full_path = self._get_full_path(file_path)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Erro ao escrever arquivo JSON: {str(e)}")
            return False
    
    def get_file_size(self, file_path: str) -> int:
        """Obtém tamanho do arquivo em bytes"""
        try:
            full_path = self._get_full_path(file_path)
            return full_path.stat().st_size
        except:
            return 0
    
    def get_file_modified_time(self, file_path: str) -> float:
        """Obtém timestamp de modificação do arquivo"""
        try:
            full_path = self._get_full_path(file_path)
            return full_path.stat().st_mtime
        except:
            return 0.0
    
    def _get_full_path(self, path: str) -> Path:
        """Converte caminho relativo para absoluto"""
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        else:
            return self.base_path / path_obj
