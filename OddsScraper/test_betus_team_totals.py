import unittest

from OddsScraper.scraper import MLBOddsScraper


class BetUSTeamTotalParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scraper = MLBOddsScraper(output_dir="data")

    def test_parse_betus_total_offer_handles_half_lines(self) -> None:
        point, price = self.scraper._parse_betus_total_offer("O\xa03½\xa0 -140Added")
        self.assertEqual(point, 3.5)
        self.assertEqual(price, -140)

    def test_parse_betus_team_code_maps_athletics(self) -> None:
        self.assertEqual(self.scraper._parse_betus_team_code("Athletics"), "ATH")

    def test_build_betus_team_total_rows(self) -> None:
        rows = self.scraper._build_betus_team_total_rows(
            {
                "boardDateText": "Mon, Mar 30, 2026 EST",
                "games": [
                    {
                        "gameTimeText": "06:40 pm",
                        "matchupHref": "https://www.betus.com.pa/stats/baseball/mlb/matchup/684386/",
                        "marketHref": "https://www.betus.com.pa/sportsbook/mlb/nationals-vs-phillies/",
                        "awayTeam": "Washington Nationals",
                        "homeTeam": "Philadelphia Phillies",
                        "awayTeamTotalOverText": "O 3½ -140 Added",
                        "awayTeamTotalUnderText": "U 3½ +110 Added",
                        "homeTeamTotalOverText": "O 4½ -135 Added",
                        "homeTeamTotalUnderText": "U 4½ +105 Added",
                    }
                ],
            },
            fetched_at="2026-03-30T20:00:00+00:00",
        )

        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0].event_id, "betus-684386")
        self.assertEqual(rows[0].game_date, "2026-03-30")
        self.assertEqual(rows[0].away_team, "WSH")
        self.assertEqual(rows[0].home_team, "PHI")
        self.assertEqual(rows[0].market_type, "full_game_team_total_away")
        self.assertEqual(rows[0].side, "over")
        self.assertEqual(rows[0].point, 3.5)
        self.assertEqual(rows[0].price, -140)
        self.assertEqual(rows[0].bookmaker, "BetUS")
        self.assertTrue(rows[0].commence_time_utc.endswith("Z"))


if __name__ == "__main__":
    unittest.main()
