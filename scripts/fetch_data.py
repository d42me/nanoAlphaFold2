"""Build the small multi-protein teaching dataset.

Teacher structures come from AlphaFold DB and MSAs from the ColabFold MMseqs2
API. Generated files are written under ``data/``, which is ignored by Git.
"""

import io
from pathlib import Path
import shutil
import tarfile
import time

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EXAMPLE_DIR = PROJECT_ROOT / "examples" / "tautomerase"

PROTEINS = {
    "crambin": "P01542",
    "bpti": "P00974",
    "hba": "P69905",
    "hbb": "P68871",
    "lysozyme": "P00698",
    "cytochrome_c": "P00004",
    "histone_h4": "P62805",
    "blg": "P02754",
    "rps27a": "P62979",
}


def fetch_afdb_entry(accession):
    response = requests.get(
        f"https://alphafold.ebi.ac.uk/api/prediction/{accession}", timeout=30
    )
    response.raise_for_status()
    entries = response.json()
    if not entries:
        raise RuntimeError(f"AlphaFold DB has no entry for {accession}")
    return entries[0]


def fetch_msa(name, sequence, output_dir, host="https://api.colabfold.com"):
    """Submit a sequence to MMseqs2 and merge returned A3M alignments."""
    output_dir = Path(output_dir)
    for attempt in range(5):
        try:
            response = requests.post(
                f"{host}/ticket/msa",
                data={"q": f">{name}\n{sequence}\n", "mode": "all"},
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            ticket = result.get("id")
            if not ticket:
                raise RuntimeError(f"MMseqs2 returned no ticket: {result}")

            for _ in range(120):
                time.sleep(5)
                status_response = requests.get(f"{host}/ticket/{ticket}", timeout=30)
                status_response.raise_for_status()
                status = status_response.json().get("status")
                if status == "COMPLETE":
                    break
                if status in {"ERROR", "UNKNOWN"}:
                    raise RuntimeError(f"MMseqs2 job failed with status {status}")
            else:
                raise RuntimeError("timed out waiting for MMseqs2")

            download = requests.get(f"{host}/result/download/{ticket}", timeout=120)
            download.raise_for_status()
            alignments = []
            with tarfile.open(
                fileobj=io.BytesIO(download.content), mode="r:*"
            ) as archive:
                for member in archive.getmembers():
                    if member.name.endswith(".a3m"):
                        extracted = archive.extractfile(member)
                        if extracted:
                            alignments.append(extracted.read().decode())
            if not alignments:
                raise RuntimeError("MMseqs2 result contained no A3M alignment")

            (output_dir / f"{name}.a3m").write_text("\n".join(alignments))
            return sum(alignment.count(">") for alignment in alignments)
        except Exception as error:
            print(f"  attempt {attempt + 1} failed: {error}", flush=True)
            time.sleep(20 * (attempt + 1))

    raise RuntimeError(f"MSA fetch failed for {name}")


def install_tautomerase_example():
    output_dir = DATA_DIR / "tautomerase"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(EXAMPLE_DIR / "alignment.a3m", output_dir / "tautomerase.a3m")
    shutil.copy2(EXAMPLE_DIR / "teacher.cif", output_dir / "teacher.cif")
    query = next(
        line.strip()
        for line in (EXAMPLE_DIR / "alignment.a3m").read_text().splitlines()
        if line and not line.startswith((">", "#"))
    )
    (output_dir / "seq.fasta").write_text(f">tautomerase\n{query}\n")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    for name, accession in PROTEINS.items():
        output_dir = DATA_DIR / name
        output_dir.mkdir(parents=True, exist_ok=True)
        if (output_dir / f"{name}.a3m").exists():
            print(f"{name}: already downloaded", flush=True)
            continue

        entry = fetch_afdb_entry(accession)
        sequence = entry["sequence"]
        cif = requests.get(
            f"https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v6.cif",
            timeout=60,
        )
        cif.raise_for_status()
        (output_dir / "teacher.cif").write_bytes(cif.content)
        (output_dir / "seq.fasta").write_text(f">{name}\n{sequence}\n")
        hits = fetch_msa(name, sequence, output_dir)
        print(
            f"{name}: {len(sequence)} aa, pLDDT "
            f"{entry['globalMetricValue']:.1f}, MSA hits ~{hits}",
            flush=True,
        )
        time.sleep(5)

    install_tautomerase_example()
    print("tautomerase: copied from examples/tautomerase")
    print("done.")


if __name__ == "__main__":
    main()
