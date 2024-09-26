import base64
import hashlib
import pathlib
import pickle
import random
from collections.abc import Iterable
from typing import TypedDict

from cryptography import fernet

from synformer.chem.mol import Molecule


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
