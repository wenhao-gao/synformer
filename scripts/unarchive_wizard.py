import pathlib
import subprocess
import sys
from functools import partial

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import track
from rich.prompt import Confirm
from rich.rule import Rule
from rich.table import Table

from synformer.chem.mol import read_mol_file
from synformer.utils.crypt import decrypt_message, load_encrypted_pack

_US_Stock_prefix = "Enamine_Rush-Delivery_Building_Blocks-US"

content = """
# Unarchive Wizard

This script will help you unarchive the **preprocessed** reaction matrices and building block fingerprints.

## Why this script?

The preprocessed data are derived from the Enamine's building block catalog, which are **available only upon request**.
Therefore, we have password-protected the preprocessed data, and to confirm that you own a copy of the Enamine's
catalog before we can reveal the password.

This script will verify whether you own a copy of the Enamine's catalog, and if you do, it will reveal the password
and unarchive the data.

## Instruction

1. Prepare the archived preprocessed data.
    - Download the archive from the link in the "Building Block Database" section of README.
    - Put the archive file `processed.zip` into the `data/building_blocks` directory.
2. Prepare Enamine's building block catalog.
    - Request the Enamine's building block catalog on
      [https://enamine.net/building-blocks/building-blocks-catalog](https://enamine.net/building-blocks/building-blocks-catalog)
    - Once you are approved, download the **US Stock** catalog from the same web page.
    - Unzip the catalog archive and put the `.sdf` file into the `data/building_blocks` directory.
3. Run this script.
    - This script will automatically verify the Enamine's catalog file in the `data` directory and unarchive the
      preprocessed data for you.
"""


def check_archive():
    archive_path = pathlib.Path("data/processed.zip")
    if archive_path.exists():
        return True, f"[green]:white_check_mark: File found: {archive_path}"
    else:
        return False, "[red]:x: File not found. Please follow the instruction above to download the file."


def check_enamine():
    data_dir = pathlib.Path("data/building_blocks")
    for file in data_dir.iterdir():
        if file.name.startswith(_US_Stock_prefix) and file.suffix == ".sdf":
            return file.absolute(), f"[green]:white_check_mark: File found: {file}"
        elif file.name.startswith("Enamine") and file.suffix == ".sdf":
            return file.absolute(), (
                f"[yellow]:construction: File found: {file}. "
                "It is possibly not the US Stock catalog and might not work."
            )
        elif file.name.startswith("Enamine") and file.suffix == ".zip":
            return False, f"[red]:stop_sign: A zipped catalog found: {file}. Please unzip the file first."
    return False, "[red]:x: File not found. Please follow the instruction above to download the file."


if __name__ == "__main__":
    console = Console()
    message = Markdown(content)
    console.print(message)
    console.print(Rule())

    table = Table(title="Checklist")

    table.add_column("Item")
    table.add_column("Status")

    archive_exists, archive_message = check_archive()
    enamine_path, enamine_message = check_enamine()
    table.add_row("Preprocessed data archive", archive_message)
    table.add_row("Enamine's building block catalog", enamine_message)

    console.print(table)

    if not (archive_exists and enamine_path):
        exit(1)

    console.print(Rule())

    if not enamine_path.name.startswith(_US_Stock_prefix):
        if not Confirm.ask(
            "The catalog file found is not the US Stock catalog. " "Do you want to continue with this file?",
            console=console,
        ):
            exit(1)

    enc_pack = load_encrypted_pack(pathlib.Path("data/processed.key"))
    out = decrypt_message(
        enc_pack,
        mols=read_mol_file(
            enamine_path,
            show_pbar=True,
            pbar_fn=partial(track, description="Reading building blocks", console=console),
        ),
        print_fn=console.log,
    )
    console.clear_live()

    if out is None:
        console.log("[red]Decryption failed.")
        console.print(
            Panel(
                "[red]Unarchive failed. Please check the Enamine's catalog file. "
                "If you believe this is an error, please contact the author."
            )
        )
        exit(1)

    password, _ = out

    console.log("[green]Decryption successful.")
    console.log("Start unarchiving the preprocessed data...")

    result = subprocess.run(
        ["unzip", "-P", password, "processed.zip", "-d", "./"],
        cwd=pathlib.Path("data").absolute(),
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stdout,
    )
    if result.returncode != 0:
        console.print(Panel("[red]Unarchive failed."))
        exit(2)

    console.print(Panel("[green]Unarchive successful."))
