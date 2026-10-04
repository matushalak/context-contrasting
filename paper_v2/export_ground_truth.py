"""Build ground_truth.csv from the empirical Pre/Post response and trace tables."""

from pathlib import Path

import pandas as pd

from thesis.data_analysis import transitions_helpers as th


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "thesis" / "data_analysis"
SECTORS = ("+NO axis", "+O axis", "-NO axis", "-O axis")


def export(output=ROOT / "ground_truth.csv"):
    transitions = th.load_transition_table(DATA / "transitions_post.csv")
    traces = pd.read_csv(DATA / "transitions_post_traces.csv")
    rows = []
    for source in ("familiar", "novel"):
        summary = th.build_mean_summary(
            transitions, image_group=source, pre_stage="Pre", target_stage="Post", threshold=0.3)
        summary = summary.loc[summary.RotatedSector.isin(SECTORS)].copy()
        membership = summary[["neuron_idx", "RotatedSector"]].rename(columns={"RotatedSector": "sector"})
        source_traces = traces.loc[traces.image_group.eq(source)].merge(
            membership, on="neuron_idx", validate="many_to_one")
        rows.append(pd.DataFrame(dict(
            record_type="trace", source=source, sector=source_traces.sector,
            observation_id=source_traces.neuron_idx,
            condition_key=source_traces.stage.map({"Pre": "naive", "Post": "expert"}),
            image_id=source_traces.image_idx_original,
            response_type=source_traces.image_type.map({"Full": "NO", "Occl": "O"}),
            time_seconds=source_traces.time, response=source_traces.response,
        )))
        rows.append(pd.DataFrame(dict(
            record_type="transition", source=source, sector=summary.RotatedSector.astype(str),
            observation_id=summary.neuron_idx, condition_key="transition",
            naive_NO=summary.NO_Pre, target_NO=summary.NO_Target,
            naive_O=summary.O_Pre, target_O=summary.O_Target,
            delta_NO=summary.dNO, delta_O=summary.dO,
        )))
    pd.concat(rows, ignore_index=True).to_csv(output, index=False)


if __name__ == "__main__":
    export()
