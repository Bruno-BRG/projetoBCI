#!/usr/bin/env python3
"""
build_installer.py – Script de automação de build do BrainBridge

Etapas:
  1. Compila o executável com PyInstaller (dist/BrainBridge/)
  2. (Opcional) Gera o instalador com Inno Setup

Uso:
  python build_installer.py                 # Build completo
  python build_installer.py --skip-installer # Só PyInstaller, sem Inno Setup
  python build_installer.py --clean         # Limpa artefatos antigos primeiro
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent
SPEC_FILE = REPO_ROOT / "brainbridge.spec"
ISS_FILE = REPO_ROOT / "installer.iss"
DIST_DIR = REPO_ROOT / "dist"
BUILD_DIR = REPO_ROOT / "build"
OUTPUT_DIR = REPO_ROOT / "Output"

# Possíveis caminhos do Inno Setup Compiler
ISCC_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"),
    Path(r"C:\Program Files\Inno Setup 6\ISCC.exe"),
    Path(r"C:\Program Files (x86)\Inno Setup 5\ISCC.exe"),
]


def find_iscc() -> Path | None:
    """Localiza o compilador Inno Setup (ISCC.exe)."""
    # Tenta via PATH primeiro
    iscc_path = shutil.which("iscc") or shutil.which("ISCC")
    if iscc_path:
        return Path(iscc_path)
    # Tenta caminhos conhecidos
    for candidate in ISCC_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def clean(verbose: bool = True):
    """Remove artefatos de builds anteriores."""
    for folder in (DIST_DIR, BUILD_DIR, OUTPUT_DIR):
        if folder.exists():
            if verbose:
                print(f"🗑  Removendo {folder.relative_to(REPO_ROOT)}/")
            shutil.rmtree(folder)
    if verbose:
        print("✓ Limpeza concluída\n")


def run_pyinstaller() -> bool:
    """Executa o PyInstaller usando o spec file."""
    print("=" * 60)
    print("  ETAPA 1 – PyInstaller: Compilando executável")
    print("=" * 60)

    if not SPEC_FILE.exists():
        print(f"✗ Arquivo spec não encontrado: {SPEC_FILE}")
        return False

    # Usa o pyinstaller do virtualenv atual
    pyinstaller_exe = shutil.which("pyinstaller")
    if not pyinstaller_exe:
        # Tenta no venv local
        venv_path = REPO_ROOT / ".venv" / "Scripts" / "pyinstaller.exe"
        if venv_path.exists():
            pyinstaller_exe = str(venv_path)
        else:
            print("✗ PyInstaller não encontrado. Instale com: pip install pyinstaller")
            return False

    cmd = [pyinstaller_exe, str(SPEC_FILE), "--noconfirm"]
    print(f"→ {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print("\n✗ PyInstaller falhou!")
        return False

    exe_path = DIST_DIR / "BrainBridge" / "BrainBridge.exe"
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"\n✓ Executável criado: {exe_path}")
        print(f"  Tamanho do .exe: {size_mb:.1f} MB")

        # Tamanho total da pasta
        total = sum(f.stat().st_size for f in (DIST_DIR / "BrainBridge").rglob("*") if f.is_file())
        total_mb = total / (1024 * 1024)
        print(f"  Tamanho total da pasta: {total_mb:.1f} MB")
    else:
        print(f"\n✗ Executável não encontrado em {exe_path}")
        return False

    return True


def run_inno_setup() -> bool:
    """Compila o instalador com Inno Setup."""
    print("\n" + "=" * 60)
    print("  ETAPA 2 – Inno Setup: Gerando instalador")
    print("=" * 60)

    iscc = find_iscc()
    if not iscc:
        print("⚠ Inno Setup não encontrado.")
        print("  Para gerar o instalador, instale o Inno Setup:")
        print("  https://jrsoftware.org/isdl.php")
        print(f"\n  Depois rode manualmente:")
        print(f'  ISCC.exe "{ISS_FILE}"')
        return False

    if not ISS_FILE.exists():
        print(f"✗ Script .iss não encontrado: {ISS_FILE}")
        return False

    cmd = [str(iscc), str(ISS_FILE)]
    print(f"→ {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print("\n✗ Inno Setup falhou!")
        return False

    # Verifica saída
    installer = OUTPUT_DIR / "BrainBridge_Setup_2.0.0.exe"
    if installer.exists():
        size_mb = installer.stat().st_size / (1024 * 1024)
        print(f"\n✓ Instalador criado: {installer}")
        print(f"  Tamanho: {size_mb:.1f} MB")
    else:
        # Pode estar com nome diferente
        installers = list(OUTPUT_DIR.glob("*.exe"))
        if installers:
            for inst in installers:
                size_mb = inst.stat().st_size / (1024 * 1024)
                print(f"\n✓ Instalador criado: {inst}")
                print(f"  Tamanho: {size_mb:.1f} MB")
        else:
            print("\n⚠ Instalador não encontrado na pasta Output/")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Build do BrainBridge – Executável + Instalador Windows"
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Limpa artefatos de builds anteriores antes de compilar"
    )
    parser.add_argument(
        "--skip-installer", action="store_true",
        help="Pula a geração do instalador (só compila com PyInstaller)"
    )
    parser.add_argument(
        "--clean-only", action="store_true",
        help="Apenas limpa artefatos, sem compilar"
    )
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════╗")
    print("║   BrainBridge – Build System v2.0.0         ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    # Limpar se solicitado
    if args.clean or args.clean_only:
        clean()
        if args.clean_only:
            return 0

    # Etapa 1: PyInstaller
    if not run_pyinstaller():
        return 1

    # Etapa 2: Inno Setup (opcional)
    if not args.skip_installer:
        run_inno_setup()
    else:
        print("\n⏭ Inno Setup pulado (--skip-installer)")

    print("\n" + "=" * 60)
    print("  BUILD CONCLUÍDO!")
    print("=" * 60)
    print(f"\n  Executável: dist/BrainBridge/BrainBridge.exe")
    if not args.skip_installer:
        print(f"  Instalador: Output/BrainBridge_Setup_2.0.0.exe")
    print(f"\n  Para testar: dist\\BrainBridge\\BrainBridge.exe")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
