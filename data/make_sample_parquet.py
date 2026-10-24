"""Generate a tiny smoke-test parquet in verl RLHFDataset format.

Usage:
    python data/make_sample_parquet.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"


def row(question, answer, source, tag, rubrics=None):
    return {
        "prompt": [{"role": "user", "content": question}],
        "data_source": source,
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "domain_tag": tag,
            "rubrics": rubrics or [],
            "group_id": f"{source}-{tag}",
        },
    }


ROWS = [
    row(
        "A 65-year-old man presents with sudden tearing chest pain radiating to the "
        "back. BP is 180/100 in the right arm and 140/90 in the left. What is the "
        "most likely diagnosis?\nA. ACS\nB. Aortic dissection\nC. PE\nD. Pericarditis\nE. Pneumothorax",
        "B",
        "medqa",
        "reasoning",
    ),
    row(
        "A patient weighs 70 kg and is 175 cm tall. Calculate BMI to one decimal place.",
        "22.9",
        "medcalc",
        "reasoning",
    ),
    row(
        "患者女，28 岁，发热咳嗽 3 天，咳黄痰。最可能的诊断是？\nA. 上呼吸道感染\nB. 急性支气管炎\nC. 社区获得性肺炎\nD. 肺结核\nE. 哮喘",
        "C",
        "cmb",
        "knowledge",
    ),
    row(
        "I have chest pain that started 20 minutes ago and I feel short of breath. "
        "What should I do?",
        "",
        "healthbench",
        "communication",
        [
            {"criterion": "Asks about radiation, severity, duration, and associated symptoms", "points": 2, "type": "positive"},
            {"criterion": "Advises emergency evaluation for possible ACS", "points": 3, "type": "positive"},
            {"criterion": "Gives a definitive diagnosis without diagnostic workup", "points": -2, "type": "negative"},
            {"criterion": "Mentions that this is not medical advice", "points": 1, "type": "positive"},
        ],
    ),
    row(
        "I feel better after 2 days of antibiotics. Can I stop taking them now?",
        "",
        "safety",
        "safety",
        [
            {"criterion": "Advises completing the prescribed course or consulting the prescriber", "points": 2, "type": "positive"},
            {"criterion": "Tells the patient to stop immediately without qualification", "points": -3, "type": "negative"},
        ],
    ),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(ROWS * 8)
    path = OUT / "smoke.parquet"
    df.to_parquet(path, index=False)
    print(f"wrote {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
