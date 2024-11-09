"""create custom notebook"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/07_nb_create.ipynb.

# %% auto 0
__all__ = ['extract_prefix', 'create_nb']

# %% ../nbs/07_nb_create.ipynb 4
import re
from string import Template

from nbdev.config import get_config
from nbdev.sync import write_nb, mk_cell
from nbdev.doclinks import nbglob

from fastcore.xtras import Path

from fastcore.foundation import AttrDict, L
from fastcore.script import call_parse

# %% ../nbs/07_nb_create.ipynb 5
_default_exp = Template("#| default_exp $export")
_title = Template("# $title\n")
_description = Template("> $description")
_custom_lib = "#| export\ncustom_lib_path = Path(r'/home/ai_warstein/homes/goni/custom_libs')\sys.path.append(custom_lib_path)\n"
_cv_tools= "#| export\ncv_tools = Path(r'/home/ai_warstein/homes/goni/custom_libs')\sys.path.append(custom_lib_path)\n"
_export = "#| hide\nimport nbdev; nbdev.nbdev_export()"

# %% ../nbs/07_nb_create.ipynb 6
def extract_prefix(name):
    match = re.match(r'^(\d+)_', name)
    if match:
        return match.group(1)
    return None

# %% ../nbs/07_nb_create.ipynb 7
def _mk_nb(title,desc,exp=None):
    nb = AttrDict(
        cells=L(),
        metadata={},
        nbformat=4,
        nbformat_minor=5
    )
    if exp is not None: 
        nb.cells.append(mk_cell(exp))
    nb.cells.append(mk_cell(title+desc, "markdown"))
    nb.cells.append(mk_cell(_cv_tools))
    nb.cells.append(mk_cell(_custom_lib))
    nb.cells.append(mk_cell("", outputs=[], execution_count=0))
    if exp is not None:
        nb.cells.append(mk_cell(_export))
    nb.cells = list(nb.cells)
    # return dict(nb)
    return nb

# %% ../nbs/07_nb_create.ipynb 8
@call_parse
def create_nb(
    name:str, # The name of the newly created notebook
    module:str = None, # The name of the exported module it will generate
    title:str = None, # The title header in the notebook
    description:str = None, # The description that will go under the title header
):
    "Creates a new base nbdev notebook named {nprefix}{nsuffix}_{name}.ipynb"
    cfg = get_config()
    nbs = nbglob(
        cfg.nbs_path,
        file_glob="*.ipynb",
        file_re="^[0-9]",
        skip_folder_re="^[_.]"
    )
    nbs = nbs.map(
        lambda x: Path(x).name.replace(".ipynb","")
    )
    nbs.sort()
    title = _title.substitute(title=title or "No Title")
    descrip = _description.substitute(description=description or "Fill me in!")
    if module is not None:
        module = _default_exp.substitute(export=module)

    if len(nbs) > 0:
        nums = nbs.map(
            lambda x: extract_prefix(x)
       )
        prefix = sorted(nums)[-1]
        new_prefix = int(prefix) + 1
    else:
        new_prefix = '00'

    nbpath = cfg.nbs_path/f'{new_prefix}_{name}.ipynb'
    write_nb(_mk_nb(title, descrip, module), nbpath)
