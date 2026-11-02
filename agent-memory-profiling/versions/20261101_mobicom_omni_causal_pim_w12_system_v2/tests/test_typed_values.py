from datetime import date, datetime
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from typed_values import Interval, parse_value


SESSION = (datetime(2025, 5, 20, 8), datetime(2025, 5, 20, 22))


def value(text, value_type):
    return {'name': 'v', 'type': value_type, 'value': text}


class TypedValuesTests(unittest.TestCase):
    def test_iso_chinese_and_english_dates(self):
        expected = Interval(date(2025, 5, 20).toordinal(),
                            date(2025, 5, 20).toordinal(), 'day')
        self.assertEqual(parse_value(value('2025-05-20', 'date'), SESSION), expected)
        self.assertEqual(parse_value(value('2025年5月20日', 'date'), SESSION), expected)
        self.assertEqual(parse_value(value('May 20, 2025', 'date'), SESSION), expected)
        self.assertEqual(parse_value(value('20 May 2025', 'date'), SESSION), expected)

    def test_month_day_without_year_uses_session_year(self):
        self.assertEqual(
            parse_value(value('5月20日', 'date'), SESSION),
            Interval(date(2025, 5, 20).toordinal(),
                     date(2025, 5, 20).toordinal(), 'day'))

    def test_cross_year_month_day_returns_whole_ambiguous_year_range(self):
        session = (datetime(2024, 12, 31, 20), datetime(2025, 1, 1, 2))
        self.assertEqual(
            parse_value(value('5月20日', 'date'), session),
            Interval(date(2024, 1, 1).toordinal(),
                     date(2025, 12, 31).toordinal(), 'day'))

    def test_relative_dates_follow_the_session_interval(self):
        self.assertEqual(
            parse_value(value('昨天', 'date'), SESSION),
            Interval(date(2025, 5, 19).toordinal(),
                     date(2025, 5, 19).toordinal(), 'day'))
        self.assertEqual(
            parse_value(value('last week', 'date'), SESSION),
            Interval(date(2025, 5, 13).toordinal(),
                     date(2025, 5, 19).toordinal(), 'day'))

    def test_numbers_durations_and_unknown_values(self):
        self.assertEqual(parse_value(value('1,234', 'number'), SESSION),
                         Interval(1234, 1234, 'number'))
        self.assertEqual(parse_value(value('一千二百三十四', 'number'), SESSION),
                         Interval(1234, 1234, 'number'))
        self.assertEqual(parse_value(value('2个月', 'duration'), SESSION),
                         Interval(56, 62, 'day'))
        self.assertEqual(parse_value(value('两年', 'duration'), SESSION),
                         Interval(730, 732, 'day'))
        self.assertIsNone(parse_value(value('不确定', 'date'), SESSION))
        self.assertIsNone(parse_value(value('13月40日', 'date'), SESSION))
        self.assertIsNone(parse_value(value('文本', 'text'), SESSION))


if __name__ == '__main__':
    unittest.main()
