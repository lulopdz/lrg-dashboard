import pandas as pd
import pytest

import parse_reports as pr


def _ieso_xml(delivery_date, resources):
    """Build a valid IESO DAScheduledEnergy2 XML string."""
    res_blocks = []
    for res_name, res_type, hours in resources:
        he_blocks = "\n".join(
            f"""          <HourlyEnergy>
            <DeliveryHour>{h}</DeliveryHour>
            <EnergyMW>{mw}</EnergyMW>
          </HourlyEnergy>"""
            for h, mw in hours
        )
        res_blocks.append(
            f"""      <Resource>
        <ResourceName>{res_name}</ResourceName>
        <ResourceType>{res_type}</ResourceType>
        <ScheduleEnergies>
          <HourlyEnergies>
{he_blocks}
          </HourlyEnergies>
        </ScheduleEnergies>
      </Resource>"""
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Document xmlns="http://www.ieso.ca/schema">\n'
        "  <DocBody>\n"
        f"    <DeliveryDate>{delivery_date}</DeliveryDate>\n"
        + "\n".join(res_blocks)
        + "\n  </DocBody>\n"
        "</Document>\n"
    )


def test_zone_of():
    assert pr.zone_of("OTTAWA_BID:HUB") == "OTTAWA"
    assert pr.zone_of("TORONTO_GEN:ZONE") == "TORONTO"
    assert pr.zone_of("NORTHWEST_LD:HUB") == "NORTHWEST"


def test_read_xml_reports_valid_and_malformed(tmp_path, capsys):
    valid_path = tmp_path / "r1.xml"
    valid_path.write_text(
        _ieso_xml(
            "2026-10-10",
            [
                ("OTTAWA_GEN:HUB", "GEN", [(1, 1.0), (18, 2.5)]),
                ("TORONTO_BID:HUB", "LD", [(5, -1.0)]),
            ],
        ),
        encoding="utf-8",
    )

    no_body_path = tmp_path / "nobody.xml"
    no_body_path.write_text(
        '<?xml version="1.0"?><Document xmlns="http://www.ieso.ca/schema"></Document>',
        encoding="utf-8",
    )

    df = pr.read_xml_reports([str(no_body_path), str(valid_path)])

    assert list(df.columns) == pr.POSITION_COLS
    assert len(df) == 3
    row0 = df[(df["resource_name"] == "OTTAWA_GEN:HUB") & (df["hour"] == 1)].iloc[0]
    assert row0["date"] == "2026-10-10"
    assert row0["resource_type"] == "GEN"
    assert row0["energy_mw"] == pytest.approx(1.0)

    malformed_path = tmp_path / "broken.xml"
    malformed_path.write_text("<Document><DocBody><unclosed>", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Failed to parse 1 XML report"):
        pr.read_xml_reports([str(malformed_path), str(valid_path)])
    assert "Error parsing" in capsys.readouterr().err


def test_update_positions_incremental_merge_and_override(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    positions_path = tmp_path / "portfolio_positions.csv"

    monkeypatch.setattr(pr, "DATA_DIR", tmp_path)
    monkeypatch.setattr(pr, "POSITIONS_PATH", positions_path)

    # Seed existing positions CSV
    existing = pd.DataFrame(
        [
            {"date": "2026-10-09", "hour": 1, "resource_name": "OTTAWA_GEN:HUB", "resource_type": "GEN", "energy_mw": 1.0},
            {"date": "2026-10-10", "hour": 1, "resource_name": "OTTAWA_GEN:HUB", "resource_type": "GEN", "energy_mw": 1.0},
        ]
    )
    existing.to_csv(positions_path, index=False)

    # Report 1 overrides 2026-10-10 HE1 and adds HE2
    (reports_dir / "01_first.xml").write_text(
        _ieso_xml("2026-10-10", [("OTTAWA_GEN:HUB", "GEN", [(1, 2.0), (2, 1.5)])]),
        encoding="utf-8",
    )
    # Later filename 02_second.xml overrides 01_first.xml for HE2
    (reports_dir / "02_second.xml").write_text(
        _ieso_xml("2026-10-10", [("OTTAWA_GEN:HUB", "GEN", [(2, 3.0)])]),
        encoding="utf-8",
    )

    df = pr.update_positions()

    assert len(df) == 3
    assert positions_path.exists()
    # Unchanged prior day
    assert df.loc[(df["date"] == "2026-10-09") & (df["hour"] == 1), "energy_mw"].iloc[0] == pytest.approx(1.0)
    # Existing row overwritten by XML
    assert df.loc[(df["date"] == "2026-10-10") & (df["hour"] == 1), "energy_mw"].iloc[0] == pytest.approx(2.0)
    # Earlier XML overwritten by later XML filename
    assert df.loc[(df["date"] == "2026-10-10") & (df["hour"] == 2), "energy_mw"].iloc[0] == pytest.approx(3.0)


def test_update_positions_without_xmls_keeps_existing(tmp_path, monkeypatch):
    positions_path = tmp_path / "portfolio_positions.csv"
    monkeypatch.setattr(pr, "DATA_DIR", tmp_path)
    monkeypatch.setattr(pr, "POSITIONS_PATH", positions_path)

    existing = pd.DataFrame(
        [{"date": "2026-10-09", "hour": 4, "resource_name": "OTTAWA_GEN:HUB", "resource_type": "GEN", "energy_mw": 1.0}]
    )
    existing.to_csv(positions_path, index=False)

    df = pr.update_positions()
    assert len(df) == 1
    assert df.iloc[0]["hour"] == 4


def test_parse_reports_computes_pnl_and_pending(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    positions_path = tmp_path / "portfolio_positions.csv"

    monkeypatch.setattr(pr, "DATA_DIR", tmp_path)
    monkeypatch.setattr(pr, "POSITIONS_PATH", positions_path)

    (reports_dir / "day.xml").write_text(
        _ieso_xml(
            "2026-10-10",
            [
                ("OTTAWA_GEN:HUB", "GEN", [(1, 2.0), (2, 0.0)]),
                ("TORONTO_BID:HUB", "LD", [(1, -1.0)]),
                ("NIAGARA_GEN:HUB", "GEN", [(1, 1.0)]),  # no prices -> Pending (NaN)
            ],
        ),
        encoding="utf-8",
    )

    dam = pd.DataFrame(
        [
            {"interval_start_local": "2026-10-10 00:00:00-05:00", "location": "OTTAWA", "lmp": 50.0},
            {"interval_start_local": "2026-10-10 01:00:00-05:00", "location": "OTTAWA", "lmp": 40.0},
            {"interval_start_local": "2026-10-10 00:00:00-05:00", "location": "TORONTO", "lmp": 30.0},
        ]
    )
    rtm = pd.DataFrame(
        [
            {"interval_start_local": "2026-10-10 00:00:00-05:00", "location": "OTTAWA", "lmp": 35.0},
            {"interval_start_local": "2026-10-10 01:00:00-05:00", "location": "OTTAWA", "lmp": 25.0},
            {"interval_start_local": "2026-10-10 00:00:00-05:00", "location": "TORONTO", "lmp": 45.0},
        ]
    )
    dam.to_csv(tmp_path / "ieso_dam_prices.csv", index=False)
    rtm.to_csv(tmp_path / "ieso_rtm_prices.csv", index=False)

    pr.parse_reports()

    out = pd.read_csv(tmp_path / "historical_pnl.csv")
    assert len(out) == 4

    ott_h1 = out[(out["resource_name"] == "OTTAWA_GEN:HUB") & (out["hour"] == 1)].iloc[0]
    assert ott_h1["spread"] == pytest.approx(15.0)
    assert ott_h1["pnl"] == pytest.approx(30.0)

    ott_h2 = out[(out["resource_name"] == "OTTAWA_GEN:HUB") & (out["hour"] == 2)].iloc[0]
    assert ott_h2["pnl"] == pytest.approx(0.0)

    tor_h1 = out[(out["resource_name"] == "TORONTO_BID:HUB") & (out["hour"] == 1)].iloc[0]
    assert tor_h1["spread"] == pytest.approx(-15.0)
    assert tor_h1["pnl"] == pytest.approx(15.0)  # -1 MW * (-15) = +15

    nia_h1 = out[(out["resource_name"] == "NIAGARA_GEN:HUB") & (out["hour"] == 1)].iloc[0]
    assert pd.isna(nia_h1["pnl"])
