from pathlib import Path


_CONFIG_FILE_PATH: Path = Path(__file__)
_CONFIG_DIR_PATH: Path = _CONFIG_FILE_PATH.parent
_SRC_DIR_PATH: Path = _CONFIG_DIR_PATH.parent.parent
_PROJECT_DIR_PATH: Path = _SRC_DIR_PATH.parent

_MOSS_DIR_PATH: Path = _PROJECT_DIR_PATH / 'moss'
_MOSS_EXEC_PATH: Path = _PROJECT_DIR_PATH / 'moss' / 'moss.jar'
_MOSS_INPUT_DIR_PATH: Path = _MOSS_DIR_PATH / 'input'
_MOSS_OUTPUT_DIR_PATH: Path = _MOSS_DIR_PATH / 'output'


CONFIG_FILE_PATH: str = str(_CONFIG_FILE_PATH.absolute())
CONFIG_DIR_PATH: str = str(_CONFIG_DIR_PATH.absolute())
SRC_DIR_PATH: str = str(_SRC_DIR_PATH.absolute())
PROJECT_DIR_PATH: str = str(_PROJECT_DIR_PATH.absolute())

MOSS_EXEC_PATH: str = str(_MOSS_EXEC_PATH.absolute())
MOSS_INPUT_DIR_PATH: str = str(_MOSS_INPUT_DIR_PATH.absolute())
MOSS_OUTPUT_DIR_PATH: str = str(_MOSS_OUTPUT_DIR_PATH.absolute())
