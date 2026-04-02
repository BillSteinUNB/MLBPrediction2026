import unittest

from OddsScraper.oddsportal_scraper import (
    OddsPortalEventInfo,
    OddsPortalScraper,
    _build_match_event_request_url,
    _build_results_archive_page_url,
    _decimal_to_american,
    _extract_primary_secondary_decimal_odds,
    _extract_primary_secondary_values,
    _epoch_seconds_to_iso8601,
    _event_from_results_archive_row,
    _event_id_from_url,
    _infer_favorite_side_from_prices,
    _normalize_oddsportal_team_code,
    _oddsportal_primary_secondary_sides,
    _parse_results_page_date_header,
    _raw_event_token_from_url,
    _resolve_signed_runline_point,
    _select_match_event_url,
)


SAMPLE_HTML = """
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": ["Event", "SportsEvent"],
  "name": "New York Yankees - Houston Astros",
  "startDate": "2023-08-05T01:05:00+02:00",
  "url": "https://www.oddsportal.com/baseball/usa/mlb-2023/new-york-yankees-houston-astros-thHHjUZt/",
  "homeTeam": {"name": "New York Yankees"},
  "awayTeam": {"name": "Houston Astros"}
}
</script>
<script>
var pageVar = '{"defaultBettingType":3,"defaultScope":1,"nav":{"3":{"1":["27","26","417","16"],"3":["16"]},"5":{"1":["26","417","16"],"17":["417"]},"2":{"1":["26","417","16"],"17":["417"]}}}';
</script>
"bettingTypes":{"3":{"name":"Home/Away","short-name":"Home/Away"},"5":{"name":"Asian Handicap","short-name":"AH"},"2":{"name":"Over/Under","short-name":"O/U"}},"bettingTypesOrder":[3,5,2]
"scopeNames":{"1":"FT including OT","3":"1st Half","17":"1st Inning"},"scopeSportNames":{"3":{"6":"1st Half Innings"}}
"""


class OddsPortalScraperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scraper = OddsPortalScraper()

    def test_parse_event_inventory(self) -> None:
        inventory = self.scraper.parse_event_inventory(SAMPLE_HTML)

        self.assertEqual(inventory.event.name, "New York Yankees - Houston Astros")
        self.assertEqual(inventory.event.home_team, "New York Yankees")
        self.assertEqual(inventory.event.away_team, "Houston Astros")
        self.assertEqual(inventory.default_betting_type_id, 3)
        self.assertEqual(inventory.default_scope_id, 1)

        market_names = {(scope.betting_type_name, scope.scope_name) for scope in inventory.scopes}
        self.assertIn(("Home/Away", "FT including OT"), market_names)
        self.assertIn(("Asian Handicap", "1st Inning"), market_names)
        self.assertIn(("Over/Under", "FT including OT"), market_names)

    def test_parse_event_inventory_honors_event_override(self) -> None:
        override = OddsPortalEventInfo(
            name="St.Louis Cardinals @ Los Angeles Dodgers",
            url="https://www.oddsportal.com/baseball/h2h/los-angeles-dodgers-nwPDBpVc/st-louis-cardinals-IDVz16ES/#YyBe1m05",
            start_date="2024-03-31T23:10:00",
            home_team="Los Angeles Dodgers",
            away_team="St.Louis Cardinals",
        )
        inventory = self.scraper.parse_event_inventory(SAMPLE_HTML, event_override=override)
        self.assertEqual(inventory.event, override)

    def test_resolve_target_market_requests_accepts_1x2_for_moneylines(self) -> None:
        inventory = self.scraper.parse_event_inventory(SAMPLE_HTML)
        targets = self.scraper._resolve_target_market_requests(inventory)
        target_keys = {(item["market_type"], item["betting_type_id"], item["scope_id"]) for item in targets}
        self.assertIn(("full_game_ml", 3, 1), target_keys)
        self.assertIn(("f5_ml", 3, 3), target_keys)

    def test_helper_functions(self) -> None:
        self.assertEqual(_normalize_oddsportal_team_code("St.Louis Cardinals"), "STL")
        self.assertEqual(_event_id_from_url("https://www.oddsportal.com/baseball/usa/mlb-2023/game-thHHjUZt/"), "oddsportal-thHHjUZt")
        self.assertEqual(_raw_event_token_from_url("https://www.oddsportal.com/baseball/h2h/a/b/#GhAdmyKi"), "GhAdmyKi")
        self.assertEqual(_parse_results_page_date_header("01 Nov 2023  - Play Offs").date().isoformat(), "2023-11-01")
        self.assertEqual(_decimal_to_american("2.50"), 150)
        self.assertEqual(_decimal_to_american("1.67"), -149)
        self.assertEqual(
            _build_match_event_request_url(
                "https://www.oddsportal.com/match-event/1-6-0IDPX5EM-3-1-a913daf5006bba0a27979f19a23b0ce8.dat?geo=CA&lang=en",
                betting_type_id=2,
                scope_id=3,
            ),
            "https://www.oddsportal.com/match-event/1-6-0IDPX5EM-2-3-a913daf5006bba0a27979f19a23b0ce8.dat?geo=CA&lang=en",
        )
        self.assertEqual(
            _select_match_event_url(
                [
                    "https://www.oddsportal.com/match-event/1-6-ERfwMJtB-3-1-a913daf5006bba0a27979f19a23b0ce8.dat?geo=CA&lang=en",
                    "https://www.oddsportal.com/match-event/1-6-YTAvKcBN-3-1-a913daf5006bba0a27979f19a23b0ce8.dat?geo=CA&lang=en",
                ],
                event_url="https://www.oddsportal.com/baseball/h2h/team-a/team-b/#YTAvKcBN",
            ),
            "https://www.oddsportal.com/match-event/1-6-YTAvKcBN-3-1-a913daf5006bba0a27979f19a23b0ce8.dat?geo=CA&lang=en",
        )

    def test_non_total_markets_map_primary_slot_to_home(self) -> None:
        self.assertEqual(_oddsportal_primary_secondary_sides("full_game_ml"), ("home", "away"))
        self.assertEqual(_oddsportal_primary_secondary_sides("full_game_rl"), ("home", "away"))
        self.assertEqual(_oddsportal_primary_secondary_sides("f5_total"), ("over", "under"))

        ml_payload = {
            "d": {
                "oddsdata": {
                    "back": {
                        "1": {
                            "handicapValue": None,
                            "odds": {
                                "417": [1.35, 3.5],
                            },
                        },
                    }
                }
            }
        }
        rl_payload = {
            "d": {
                "oddsdata": {
                    "back": {
                        "2": {
                            "handicapValue": "1.5",
                            "odds": {
                                "417": [1.91, 1.91],
                            },
                        },
                    }
                }
            }
        }

        ml_rows = self.scraper._build_rows_from_market_payload(
            payload=ml_payload,
            provider_name_map={"417": "SampleBook"},
            bookmaker_favorite_sides=None,
            event_id="oddsportal-test",
            game_date="2024-09-02",
            commence_time_utc="2024-09-02T19:05:00Z",
            away_team="CHW",
            home_team="BAL",
            game_type="regular",
            market_type="full_game_ml",
            fetched_at="2024-09-02T18:00:00Z",
        )
        self.assertEqual([(row.side, row.price) for row in ml_rows], [("home", -286), ("away", 250)])

        rl_rows = self.scraper._build_rows_from_market_payload(
            payload=rl_payload,
            provider_name_map={"417": "SampleBook"},
            bookmaker_favorite_sides={"OddsPortal:SampleBook": "away"},
            event_id="oddsportal-test",
            game_date="2024-09-02",
            commence_time_utc="2024-09-02T19:05:00Z",
            away_team="CHW",
            home_team="BAL",
            game_type="regular",
            market_type="full_game_rl",
            fetched_at="2024-09-02T18:00:00Z",
        )
        self.assertEqual(
            [(row.side, row.point, row.price) for row in rl_rows],
            [("home", 1.5, -110), ("away", -1.5, -110)],
        )

    def test_moneyline_favorite_map_and_runline_sign_helpers(self) -> None:
        ml_payload = {
            "d": {
                "oddsdata": {
                    "back": {
                        "1": {
                            "handicapValue": None,
                            "odds": {
                                "26": [1.74, 2.20],
                                "417": [2.35, 1.62],
                            },
                        },
                    }
                }
            }
        }

        favorite_map = self.scraper._build_bookmaker_favorite_map_from_moneyline_payload(
            payload=ml_payload,
            provider_name_map={"26": "BookA", "417": "BookB"},
            market_type="full_game_ml",
        )
        self.assertEqual(favorite_map["OddsPortal:BookA"], "home")
        self.assertEqual(favorite_map["OddsPortal:BookB"], "away")
        self.assertEqual(_infer_favorite_side_from_prices(home_price=-135, away_price=115), "home")
        self.assertEqual(_infer_favorite_side_from_prices(home_price=110, away_price=-130), "away")
        self.assertEqual(_resolve_signed_runline_point(side="home", point_value=1.5, favorite_side="away"), 1.5)
        self.assertEqual(_resolve_signed_runline_point(side="away", point_value=1.5, favorite_side="away"), -1.5)
        self.assertEqual(_resolve_signed_runline_point(side="home", point_value=-1.5, favorite_side=None), -1.5)
        self.assertEqual(_resolve_signed_runline_point(side="away", point_value=-1.5, favorite_side=None), 1.5)

    def test_extract_primary_secondary_decimal_odds_handles_1x2_dicts(self) -> None:
        self.assertEqual(
            _extract_primary_secondary_decimal_odds({"0": 1.69, "2": 2.6, "1": 8.5}, market_type="full_game_ml"),
            (1.69, 2.6),
        )
        self.assertEqual(
            _extract_primary_secondary_decimal_odds([1.39, 2.9], market_type="full_game_rl"),
            (1.39, 2.9),
        )
        self.assertEqual(
            _extract_primary_secondary_values({"0": 1711913875, "2": 1711913000, "1": 1711912000}, market_type="full_game_ml"),
            (1711913875, 1711913000),
        )
        self.assertEqual(_epoch_seconds_to_iso8601(1711913875), "2024-03-31T19:37:55Z")

    def test_results_archive_helpers(self) -> None:
        first_archive_url = (
            "https://www.oddsportal.com/ajax-sport-country-tournament-archive_/6/Sj67Y5TK/"
            "X202178560X0X0X0X0X0X0X0X0X0X0X0X0X134217728X0X0X0X0X0X8X512X32X0X0X0X0X0X0X0"
            "X536870912X2560X2048X0X33554560X8519680X0X0X0X524288/1/0/?_=1775062998875"
        )
        page_two_url = _build_results_archive_page_url(first_archive_url, 2)
        self.assertEqual(
            page_two_url,
            "https://www.oddsportal.com/ajax-sport-country-tournament-archive_/6/Sj67Y5TK/"
            "X202178560X0X0X0X0X0X0X0X0X0X0X0X0X134217728X0X0X0X0X0X8X512X32X0X0X0X0X0X0X0"
            "X536870912X2560X2048X0X33554560X8519680X0X0X0X524288/1/0/page/2/?_=1775062998875",
        )

        event = _event_from_results_archive_row(
            {
                "url": "/baseball/h2h/arizona-diamondbacks-8bP2bXmH/san-diego-padres-8poQN9Ud/#GhAdmyKi",
                "home-name": "Arizona Diamondbacks",
                "away-name": "San Diego Padres",
                "date-start-base": 1727637000,
            }
        )
        assert event is not None
        self.assertEqual(event.name, "San Diego Padres @ Arizona Diamondbacks")
        self.assertEqual(event.home_team, "Arizona Diamondbacks")
        self.assertEqual(event.away_team, "San Diego Padres")
        self.assertEqual(event.start_date, "2024-09-29T19:10:00")
        self.assertEqual(
            event.url,
            "https://www.oddsportal.com/baseball/h2h/arizona-diamondbacks-8bP2bXmH/san-diego-padres-8poQN9Ud/#GhAdmyKi",
        )


if __name__ == "__main__":
    unittest.main()
