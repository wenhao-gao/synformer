import base64
import hashlib
import pathlib
import pickle
import random
import sys
from collections.abc import Iterable
from typing import TypedDict

import click
from cryptography import fernet

from synformer.chem.mol import Molecule, read_mol_file


def encrypt(message: str, key: bytes) -> bytes:
    return fernet.Fernet(key).encrypt(message.encode())


def decrypt(message: bytes, key: bytes) -> str:
    return fernet.Fernet(key).decrypt(message).decode()


def generate_hints_from_mols(mols: list[Molecule]) -> set[bytes]:
    hints_md5: set[bytes] = {m.csmiles_md5 for m in mols}
    return hints_md5


def generate_key_from_mols(mols: list[Molecule]) -> bytes:
    smiles_list = [m.csmiles for m in mols]
    smiles_list.sort()
    smiles_joined = ";".join(smiles_list)
    key_bytes = hashlib.sha256(smiles_joined.encode()).digest()
    key_base64 = base64.b64encode(key_bytes)
    return key_base64


class EncryptedPack(TypedDict):
    encrypted: bytes
    hints: set[bytes]


def encrypt_message(
    message: str,
    mols: Iterable[Molecule],
    seed: int,
    num_keys: int,
) -> tuple[EncryptedPack, list[Molecule]]:
    key_mols = random.Random(seed).sample(list(mols), num_keys)
    key = generate_key_from_mols(key_mols)
    hints_md5 = generate_hints_from_mols(key_mols)
    enc = encrypt(message, key)
    pack: EncryptedPack = {"encrypted": enc, "hints": hints_md5}
    return pack, key_mols


def decrypt_message(
    pack: EncryptedPack,
    mols: Iterable[Molecule],
    print_fn=print,
) -> tuple[str, bytes] | None:
    key_mols: list[Molecule] = []
    for mol in mols:
        if mol.csmiles_md5 in pack["hints"]:
            key_mols.append(mol)
            print_fn(f"Found key molecule {len(key_mols)}/{len(pack['hints'])}.")
            if len(key_mols) == len(pack["hints"]):
                break

    if len(key_mols) == len(pack["hints"]):
        print_fn("All key molecules found!")
        print_fn("Starting decryption...")
    else:
        print_fn("Not all key molecules found, aborting.")
        return None

    key = generate_key_from_mols(key_mols)
    message = decrypt(pack["encrypted"], key)
    return message, key


def save_encrypted_pack(pack: EncryptedPack, path: pathlib.Path):
    with open(path, "wb") as f:
        f.write(base64.b64encode(pickle.dumps(pack)))


def load_encrypted_pack(path: pathlib.Path) -> EncryptedPack:
    with open(path) as f:
        pack = pickle.loads(base64.b64decode(f.read()))
    return pack


@click.group()
def cli():
    pass


@cli.command(name="encrypt")
@click.option(
    "--sdf-path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    required=True,
    default="data/Enamine_Rush-Delivery_Building_Blocks-US_223244cmpd_20231001.sdf",
)
@click.option(
    "--enc-path",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    default="data/encrypted_message.pkl",
)
@click.option("--seed", type=int, default=2024)
@click.option("--num-keys", type=int, default=10)
def encrypt_cli(sdf_path: pathlib.Path, enc_path: pathlib.Path, seed: int, num_keys: int):
    mols = list(read_mol_file(sdf_path))

    print("Enter message to encrypt:")
    message = sys.stdin.read()
    pack, key_mols = encrypt_message(
        message=message,
        mols=mols,
        seed=seed,
        num_keys=num_keys,
    )

    print("Selected key molecules:")
    for mol in key_mols:
        print(f" - {mol.csmiles}, {str(mol.csmiles_md5)}")

    with enc_path.open("wb") as f:
        pickle.dump(pack, f)
    print(f"Saved encrypted message to {enc_path}")


@cli.command(name="decrypt")
@click.option(
    "--sdf-path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    required=True,
    default="data/Enamine_Rush-Delivery_Building_Blocks-US_223244cmpd_20231001.sdf",
)
@click.option(
    "--enc-path",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    default="data/encrypted_message.pkl",
)
def decrypt_cli(sdf_path: pathlib.Path, enc_path: pathlib.Path):
    pack: EncryptedPack = pickle.load(open(enc_path, "rb"))
    out = decrypt_message(pack, read_mol_file(sdf_path))
    if out is None:
        print("Could not decrypt message, aborting.")
        return

    message, key = out
    print(f"Key: {str(key)}")
    print("----- BEGIN DECRYPTED MESSAGE -----\n")
    print(message)
    print("\n----- END DECRYPTED MESSAGE -----")


if __name__ == "__main__":
    cli()
