"""Parse source-located witness values into conservative typed intervals."""
from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import re


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float
    unit: str


MONTHS = {
    name: index for index, name in enumerate(
        ('january', 'february', 'march', 'april', 'may', 'june', 'july',
         'august', 'september', 'october', 'november', 'december'), 1)
}
MONTHS.update({name[:3]: index for name, index in list(MONTHS.items())})
CHINESE_DIGITS = {
    '零': 0, '〇': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
}
CHINESE_UNITS = {'十': 10, '百': 100, '千': 1000}
CHINESE_NUMBER_RE = re.compile(r'[零〇一二两三四五六七八九十百千万]+')
ARABIC_NUMBER_RE = re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?')
ISO_DATE_RE = re.compile(r'(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)')
CHINESE_DATE_RE = re.compile(r'(?<!\d)(\d{4})年(\d{1,2})月(\d{1,2})[日号]')
CHINESE_MONTH_DAY_RE = re.compile(r'(?<!\d)(\d{1,2})月(\d{1,2})[日号]')
ENGLISH_DATE_RE = re.compile(
    r'\b(?:(?P<month1>[A-Za-z]+)\s+(?P<day1>\d{1,2})(?:st|nd|rd|th)?'
    r'(?:\s*,?\s*(?P<year1>\d{4}))?|'
    r'(?P<day2>\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?'
    r'(?P<month2>[A-Za-z]+)(?:\s*,?\s*(?P<year2>\d{4}))?)\b',
    re.IGNORECASE)
RELATIVE_DAYS = {'前天': -2, '昨天': -1, '今天': 0, '明天': 1, '后天': 2}
RELATIVE_ENGLISH_DAYS = {
    'day before yesterday': -2, 'yesterday': -1, 'today': 0,
    'tomorrow': 1, 'day after tomorrow': 2,
}
DURATION_RE = re.compile(
    r'(?P<number>\d[\d,]*(?:\.\d+)?|[零〇一二两三四五六七八九十百千万]+)\s*'
    r'(?P<unit>星期|周|天|日|个月|月|年)')


def _point(value: float, unit: str) -> Interval:
    return Interval(float(value), float(value), unit)


def _parse_chinese_integer(text: str) -> int | None:
    if not text or not re.fullmatch(r'[零〇一二两三四五六七八九十百千万]+', text):
        return None
    total = section = number = 0
    for char in text:
        if char in CHINESE_DIGITS:
            number = CHINESE_DIGITS[char]
            continue
        if char == '万':
            section += number
            total += section * 10000
            section = number = 0
            continue
        unit = CHINESE_UNITS.get(char)
        if unit is None:
            return None
        if number == 0:
            number = 1
        section += number * unit
        number = 0
    return total + section + number


def parse_number_text(text: str) -> float | None:
    match = ARABIC_NUMBER_RE.search(text or '')
    if match:
        try:
            return float(match.group(0).replace(',', ''))
        except ValueError:
            return None
    match = CHINESE_NUMBER_RE.search(text or '')
    if match:
        value = _parse_chinese_integer(match.group(0))
        return float(value) if value is not None else None
    return None


def _datetime_point(value: datetime) -> Interval:
    seconds = (value.toordinal() * 86400 + value.hour * 3600
               + value.minute * 60 + value.second + value.microsecond / 1_000_000)
    return _point(seconds, 'second')


def _date_day(value: date) -> Interval:
    return _point(value.toordinal(), 'day')


def _session_dates(session_interval):
    if (not isinstance(session_interval, tuple) or len(session_interval) != 2
            or not all(isinstance(value, datetime) for value in session_interval)):
        return None
    start, end = session_interval
    if end < start:
        return None
    return start.date(), end.date()


def _month_day_interval(month: int, day: int, session_interval) -> Interval | None:
    dates = _session_dates(session_interval)
    if dates is None:
        return None
    start, end = dates
    try:
        if start.year == end.year:
            return _date_day(date(start.year, month, day))
    except ValueError:
        return None
    lo, hi = sorted((start.year, end.year))
    try:
        date(lo, month, day)
        date(hi, month, day)
    except ValueError:
        return None
    return Interval(date(lo, 1, 1).toordinal(), date(hi, 12, 31).toordinal(), 'day')


def _shift_month(year: int, month: int, offset: int) -> tuple[int, int]:
    index = year * 12 + month - 1 + offset
    return index // 12, index % 12 + 1


def _month_interval(year: int, month: int, *, last_day: bool) -> int:
    day = calendar.monthrange(year, month)[1] if last_day else 1
    return date(year, month, day).toordinal()


def _relative_date(text: str, session_interval) -> Interval | None:
    dates = _session_dates(session_interval)
    if dates is None:
        return None
    start, end = dates
    lower = text.lower()
    for cue, offset in RELATIVE_ENGLISH_DAYS.items():
        if cue in lower:
            shift = timedelta(days=offset)
            return Interval((start + shift).toordinal(), (end + shift).toordinal(), 'day')
    for cue, offset in RELATIVE_DAYS.items():
        if cue in text:
            shift = timedelta(days=offset)
            return Interval((start + shift).toordinal(), (end + shift).toordinal(), 'day')
    if '本周' in text or '这周' in text or 'this week' in lower:
        return Interval(start.toordinal(), end.toordinal(), 'day')
    if '上周' in text or 'last week' in lower:
        return Interval((start - timedelta(days=7)).toordinal(),
                        (end - timedelta(days=1)).toordinal(), 'day')
    if '本月' in text or '这个月' in text or 'this month' in lower:
        return Interval(_month_interval(start.year, start.month, last_day=False),
                        _month_interval(end.year, end.month, last_day=True), 'day')
    if '上个月' in text or 'last month' in lower:
        start_year, start_month = _shift_month(start.year, start.month, -1)
        end_year, end_month = _shift_month(end.year, end.month, -1)
        return Interval(_month_interval(start_year, start_month, last_day=False),
                        _month_interval(end_year, end_month, last_day=True), 'day')
    if '今年' in text or 'this year' in lower:
        return Interval(date(start.year, 1, 1).toordinal(),
                        date(end.year, 12, 31).toordinal(), 'day')
    if '去年' in text or 'last year' in lower:
        return Interval(date(start.year - 1, 1, 1).toordinal(),
                        date(end.year - 1, 12, 31).toordinal(), 'day')
    return None


def _parse_date(text: str, value_type: str, session_interval) -> Interval | None:
    relative = _relative_date(text, session_interval)
    if relative is not None:
        return relative
    if value_type == 'datetime' and (('T' in text or ' ' in text)
                                     and not re.search(r'[zZ]|[+-]\d\d:?\d\d$', text)):
        try:
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is None:
                return _datetime_point(parsed)
        except ValueError:
            pass
    match = ISO_DATE_RE.search(text)
    if match:
        try:
            return _date_day(date(*map(int, match.groups())))
        except ValueError:
            return None
    match = CHINESE_DATE_RE.search(text)
    if match:
        try:
            return _date_day(date(*map(int, match.groups())))
        except ValueError:
            return None
    match = CHINESE_MONTH_DAY_RE.search(text)
    if match:
        month, day = map(int, match.groups())
        return _month_day_interval(month, day, session_interval)
    match = ENGLISH_DATE_RE.search(text)
    if match:
        month_name = (match.group('month1') or match.group('month2') or '').lower()
        month = MONTHS.get(month_name)
        if month is None:
            return None
        day = int(match.group('day1') or match.group('day2'))
        year = match.group('year1') or match.group('year2')
        if year:
            try:
                return _date_day(date(int(year), month, day))
            except ValueError:
                return None
        return _month_day_interval(month, day, session_interval)
    return None


def _parse_duration(text: str) -> Interval | None:
    match = DURATION_RE.search(text)
    if not match:
        return None
    number_text = match.group('number')
    amount = (float(number_text.replace(',', ''))
              if re.fullmatch(r'\d[\d,]*(?:\.\d+)?', number_text)
              else _parse_chinese_integer(number_text))
    if amount is None:
        return None
    unit = match.group('unit')
    if unit in {'天', '日'}:
        days = float(amount)
    elif unit in {'周', '星期'}:
        days = float(amount) * 7
    elif unit in {'月', '个月'}:
        return Interval(float(amount) * 28, float(amount) * 31, 'day')
    elif unit == '年':
        return Interval(float(amount) * 365, float(amount) * 366, 'day')
    else:
        return None
    return Interval(days, days, 'day')


def parse_value(v: dict, session_interval: tuple[datetime, datetime]) -> Interval | None:
    if not isinstance(v, dict):
        return None
    value = v.get('value')
    value_type = v.get('type')
    if not isinstance(value, str) or not value or not isinstance(value_type, str):
        return None
    if value_type in {'date', 'datetime'}:
        return _parse_date(value, value_type, session_interval)
    if value_type == 'duration':
        return _parse_duration(value)
    if value_type in {'number', 'count'}:
        number = parse_number_text(value)
        if number is None:
            return None
        return _point(number, 'count' if value_type == 'count' else 'number')
    return None
