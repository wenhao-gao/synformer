import dataclasses
import datetime
from typing import NoReturn

import git


@dataclasses.dataclass(frozen=True)
class _VersionControlInfo:
    head: str
    branch: str | None
    committed_at: datetime.datetime
    modified_files: list[str]
    new_files: list[str]
    staged_files: list[str]

    @property
    def has_changes(self) -> bool:
        return len(self.modified_files + self.new_files + self.staged_files) > 0

    @property
    def display_version(self) -> str:
        if self.branch is not None:
            return f"{self.head[:7]}@{self.branch}"
        else:
            return self.head[:7]

    @property
    def changes(self) -> list[str]:
        return self.modified_files + self.new_files + self.staged_files

    def disallow_changes(self, debug: bool = False, exitcode: int = 1) -> NoReturn | None:
        if self.has_changes and not debug:
            print("\n\nYou have uncommitted changes:")
            for fn in self.changes:
                print(f" - {fn}")
            print("Please commit your changes before running the script.\n\n")
            exit(exitcode)
        return None

    def asdict(self):
        return dataclasses.asdict(self)


def get_vc_info(path="./") -> _VersionControlInfo:
    repo = git.Repo(path)
    head = repo.head.object.hexsha
    branch: str | None = None
    try:
        branch = repo.active_branch.name
    except TypeError:
        pass
    modified_files = [str(f.a_path) for f in repo.index.diff(None)]
    new_files = repo.untracked_files
    staged_files = [str(f.a_path) for f in repo.index.diff("HEAD")]
    return _VersionControlInfo(
        head=head,
        branch=branch,
        committed_at=repo.head.object.committed_datetime,
        modified_files=modified_files,
        new_files=new_files,
        staged_files=staged_files,
    )
