import datetime as dt
import xml.etree.ElementTree as ET

import pytest

import check_freshness as cf
import parse_reports as pr
import update_prices_ieso as upi


def test_latest_per_zone_valid_and_invalid(tmp_path):
    csv_ok = tmp_path / "rtm_ok.csv"
    csv_ok.write_text(
        "interval_start_local,location,lmp\n"
        "2026-10-10 00:00:00-05:00,OTTAWA,20.0\n"
        "2026-10-10 03:00:00,OTTAWA,25.0\n"
        "2026-10-10 01:00:00-05:00,TORONTO,30.0\n",
        encoding="utf-8",
    )
    latest = cf.latest_per_zone(csv_ok)
    assert set(latest.keys()) == {"OTTAWA", "TORONTO"}
    assert latest["OTTAWA"] == dt.datetime(2026, 10, 10, 3, 0, tzinfo=cf.MARKET_TZ)

    csv_missing_col = tmp_path / "rtm_bad_col.csv"
    csv_missing_col.write_text("interval_start_local,lmp\n2026-10-10 00:00:00-05:00,20.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing expected columns"):
        cf.latest_per_zone(csv_missing_col)

    csv_empty_cell = tmp_path / "rtm_empty_cell.csv"
    csv_empty_cell.write_text("interval_start_local,location,lmp\n2026-10-10 00:00:00-05:00,,20.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty location or interval_start_local"):
        cf.latest_per_zone(csv_empty_cell)


def test_parse_reports_zone_of_and_invalid_hour(tmp_path):
    with pytest.raises(ValueError, match="Unexpected resource_name format"):
        pr.zone_of("OTTAWA")

    bad_hour = tmp_path / "bad_hour.xml"
    bad_hour.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Document xmlns="http://www.ieso.ca/schema">\n'
        "  <DocBody>\n"
        "    <DeliveryDate>2026-10-10</DeliveryDate>\n"
        "    <Resource>\n"
        "      <ResourceName>OTTAWA_GEN:HUB</ResourceName>\n"
        "      <ResourceType>GEN</ResourceType>\n"
        "      <ScheduleEnergies><HourlyEnergies><HourlyEnergy>\n"
        "        <DeliveryHour>25</DeliveryHour><EnergyMW>1.0</EnergyMW>\n"
        "      </HourlyEnergy></HourlyEnergies></ScheduleEnergies>\n"
        "    </Resource>\n"
        "  </DocBody>\n"
        "</Document>\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="DeliveryHour out of range"):
        pr.read_xml_reports([str(bad_hour)])


def test_update_prices_ieso_dam_and_rtm_parsing(monkeypatch):
    dam_xml = ET.fromstring(
        '<Document xmlns="http://www.ieso.ca/schema"><DocBody>'
        "<TransactionZone><ZoneName>OTTAWA:HUB</ZoneName>"
        "<Components><PriceComponent>Zonal Price</PriceComponent>"
        "<DeliveryHour><Hour>1</Hour><LMP>42.50</LMP></DeliveryHour>"
        "</Components>"
        "<Components><PriceComponent>Congestion</PriceComponent>"
        "<DeliveryHour><Hour>1</Hour><LMP>1.00</LMP></DeliveryHour>"
        "</Components>"
        "</TransactionZone></DocBody></Document>"
    )
    monkeypatch.setattr(upi, "fetch", lambda url: dam_xml)
    df_dam = upi.dam_rows(dt.date(2026, 10, 10))
    assert len(df_dam) == 1
    assert df_dam.iloc[0]["location"] == "OTTAWA"
    assert df_dam.iloc[0]["lmp"] == pytest.approx(42.5)

    rtm_xml = ET.fromstring(
        '<Document xmlns="http://www.ieso.ca/schema"><DocBody>'
        "<TransactionZone><ZoneName>TORONTO:HUB</ZoneName>"
        "<IntervalPrice><ZonalPrice>10.0</ZonalPrice></IntervalPrice>"
        "<IntervalPrice><ZonalPrice>20.0</ZonalPrice></IntervalPrice>"
        "</TransactionZone></DocBody></Document>"
    )
    monkeypatch.setattr(upi, "fetch", lambda url: rtm_xml if "2026101001" in url else None)
    df_rtm = upi.rtm_rows(dt.date(2026, 10, 10), [1, 2])
    assert len(df_rtm) == 1
    assert df_rtm.iloc[0]["location"] == "TORONTO"
    assert df_rtm.iloc[0]["lmp"] == pytest.approx(15.0)
