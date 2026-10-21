"""Copy session data from this rig to the lab server (the Push Data button).

Who can push where comes from GUI/paths/<name>.json, one file per experimenter:

    {"initials": "SS", "data_path": "/Serkan/Head_Fixed_Behavior/Data/Raw"}

Only subjects whose ID starts with the initials are copied. data_path is looked for on every
mapped network drive, whatever its letter.

The copy only adds files: it never overwrites or deletes, on the server or on the rig. Each
file is copied under a temporary name and renamed once complete, so an interrupted push
leaves no half-written file that a later push would take for a finished one.
"""
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import psutil

PARTIAL_SUFFIX = '.partial'


def load_experimenters(folder):
    """{initials: config} from every *.json in folder. Files without initials and data_path
    are skipped with a message in the console window."""
    experimenters = {}
    for path in sorted(Path(folder).glob('*.json')):
        try:
            config = json.loads(path.read_text())
            initials, _ = config['initials'], config['data_path']
        except (ValueError, KeyError, TypeError) as e:
            print(f"Skipping {path}: it needs 'initials' and 'data_path' ({e!r})")
            continue
        experimenters[initials] = config
    return experimenters


def mapped_drives():
    """Roots of the network drives mapped on this PC, e.g. 'Z:\\'."""
    return [p.device for p in psutil.disk_partitions(all=True) if 'remote' in p.opts]


def find_server_folder(data_path, drives=None):
    """The first mapped drive on which data_path exists, as 'Z:/Serkan/...', or None."""
    for drive in mapped_drives() if drives is None else drives:
        candidate = drive[:2] + data_path
        if os.path.isdir(candidate):
            return candidate
    return None


@dataclass
class Report:
    copied: list = field(default_factory=list)      # relative paths copied this time
    already: int = 0                                 # already on the server with the same size
    different: list = field(default_factory=list)   # on the server with another size: left alone

    def summary(self, destination):
        lines = [f"Copied {len(self.copied)} new file(s) to {destination}.",
                 f"{self.already} file(s) were already there."]
        if self.different:
            shown = '\n'.join('  ' + p for p in self.different[:10])
            more = f"\n  ... and {len(self.different) - 10} more" if len(self.different) > 10 else ''
            lines.append(f"\n{len(self.different)} file(s) are on the server with a different size and "
                         f"were NOT overwritten (copied from an unfinished session?). Check them by hand:\n"
                         f"{shown}{more}")
        return '\n'.join(lines)


def copy_new_files(source, destination, prefix):
    """Copy every file under source/<subject>/ that destination lacks, for subjects starting with prefix."""
    report = Report()
    for subject in sorted(os.listdir(source)):
        if not subject.startswith(prefix) or not os.path.isdir(os.path.join(source, subject)):
            continue
        for folder, _, files in os.walk(os.path.join(source, subject)):
            relative = os.path.relpath(folder, source)
            target = os.path.join(destination, relative)
            for name in sorted(files):
                source_file, target_file = os.path.join(folder, name), os.path.join(target, name)
                if os.path.exists(target_file):
                    if os.path.getsize(target_file) == os.path.getsize(source_file):
                        report.already += 1
                    else:
                        report.different.append(os.path.join(relative, name))
                    continue
                os.makedirs(target, exist_ok=True)
                partial = target_file + PARTIAL_SUFFIX
                shutil.copy2(source_file, partial)
                os.replace(partial, target_file)
                report.copied.append(os.path.join(relative, name))
    return report
