"""Contagem CRUA dos XMLs do DDR — prova que o gap de caixas = dedup + degeneradas.

Reparseia os XMLs Pascal VOC oficiais (data/raw/.../lesion_detection/{train,valid,test})
com a MESMA lógica do 01_convert_annotations.py e quebra, por classe:
  raw          objetos brutos no XML (antes de qualquer filtro)
  degeneradas  descartadas por xmax<=xmin ou ymax<=ymin (linha 75 do convert)
  dup_exatas   descartadas por serem caixa idêntica repetida no mesmo XML (linhas 77-81)
  kept         o que sobra = o que vira label YOLO

Se 'raw' bater com o publicado (Li et al. / Santos Tabela 2: MA 10388, HE 13093, EX 23713,
SE 1558), fica demonstrado que teu GT lê exatamente as caixas oficiais e só remove
duplicatas/degeneradas — nada perdido por bug. 'kept' deve bater com o check_annotation_counts.

Uso (na Balerion, sem GPU, só data/ montado):
    sudo docker run --rm --entrypoint python -v "$PWD/data:/app/data" \
        -v "$PWD/scripts/check_xml_raw_counts.py:/app/scripts/check_xml_raw_counts.py" \
        retino-sahi scripts/check_xml_raw_counts.py
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DET_DIR = REPO_ROOT / "data" / "raw" / "oia-ddr" / "DDR-dataset" / "lesion_detection"
SPLITS = ["train", "valid", "test"]
CLASS_MAP = {"ma": 0, "ex": 1, "se": 2, "he": 3}
CLASS_NAMES = {0: "MA", 1: "EX", 2: "SE", 3: "HE"}
ORDER = ["MA", "HE", "EX", "SE"]
REF_BOXES = {"MA": 10388, "HE": 13093, "EX": 23713, "SE": 1558}


def main():
    raw, degen, dup, kept = Counter(), Counter(), Counter(), Counter()
    n_xml, unknown = 0, 0

    for split in SPLITS:
        for xml_path in sorted((DET_DIR / split).glob("*.xml")):
            n_xml += 1
            seen = set()  # dedup por imagem (mesmo escopo do parse_xml do convert)
            for obj in ET.parse(xml_path).getroot().findall("object"):
                name = (obj.findtext("name") or "").strip().lower()
                if name not in CLASS_MAP:
                    unknown += 1
                    continue
                cls = CLASS_NAMES[CLASS_MAP[name]]
                raw[cls] += 1
                b = obj.find("bndbox")
                xmin, ymin = float(b.findtext("xmin")), float(b.findtext("ymin"))
                xmax, ymax = float(b.findtext("xmax")), float(b.findtext("ymax"))
                if xmax <= xmin or ymax <= ymin:
                    degen[cls] += 1
                    continue
                key = (name, xmin, ymin, xmax, ymax)
                if key in seen:
                    dup[cls] += 1
                    continue
                seen.add(key)
                kept[cls] += 1

    print(f"XMLs lidos: {n_xml}  |  objetos com classe desconhecida: {unknown}\n")
    print(f"  {'classe':6} {'raw':>7} {'ref':>7} {'raw-ref':>8} {'degen':>6} {'dup':>6} {'kept':>7}")
    ok = True
    for cls in ORDER:
        r, ref = raw[cls], REF_BOXES[cls]
        d = r - ref
        ok &= (d == 0)
        print(f"  {cls:6} {r:>7} {ref:>7} {d:>+8} {degen[cls]:>6} {dup[cls]:>6} {kept[cls]:>7}")
    print(f"\n  totais: raw={sum(raw.values())}  ref={sum(REF_BOXES.values())}  "
          f"degen={sum(degen.values())}  dup={sum(dup.values())}  kept={sum(kept.values())}")

    if ok:
        print("\nRESULTADO: raw == publicado em TODAS as classes -> teu GT le exatamente as "
              "caixas oficiais do DDR; o deficit no 'kept' e' 100% dedup + degeneradas "
              "(limpeza correta). Pipeline de dados validado contra a literatura.")
    else:
        print("\nRESULTADO: 'raw' ainda diverge do publicado -> investigar (nome de classe "
              "nao mapeado, XML a mais/menos, ou o publicado conta objetos de outra forma).")


if __name__ == "__main__":
    main()
