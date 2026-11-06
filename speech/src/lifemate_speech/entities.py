"""WEAK regex labels on reference; no semantic repair or transcript rewriting.

Strict entity sentence match compares multisets of extracted NFC/whitespace
spans, preserving case/punctuation. Missing reference category => not_present;
false-positive-only hypotheses are separately counted. Heuristics miss unseen
English terms, unsupported spoken phone/date forms and lowercase names, and may overcapture
addresses. This is diagnostic evidence, not ground truth entity annotation.

Evaluation v3: a single trailing digit after a money scale is interpreted as
tenths of that scale (hai triệu tám = 2,800,000); rưỡi adds half a single scale.
Multi-digit unsuffixed tails and compound-scale rưỡi are ambiguous, never guessed.
Fractional expressions are captured IN FULL, even when parsing fails. Relaxed
comparison explicitly rejects a fractional-presence mismatch, even if numeric
values happen to agree; this deliberately conservative policy can reject an
equivalent unscaled VND spelling (see fraction_presence_mismatch).
Clock values handle kém/rưỡi/minutes; absent daypart does not infer AM/PM.
Calendar dates exclude a following 'lần'; missing month/year remain unknown.
Mai after contact titles is a name; bare Mai can still be name or tomorrow.
It remains a weak relative-time label WITH an uncertainty flag. No context,
calendar resolution, ASR changes or transcript rewriting is performed.
"""
from collections import Counter
from decimal import Decimal, InvalidOperation
from datetime import date
import re
from .normalize import normalize

ENTITY_RULES_VERSION = 'weak-regex-v3'
MONEY_RELAXED_VERSION = 'vnd-fractions-guard-v2'
RELATIVE_TIME_RULES_VERSION = 'relative-phrase-mai-v2'
TIME_RULES_VERSION = 'clock-kem-ruoi-v1'
CATEGORIES = ('money','time','date','phone','name/contact','address','english_term')
WORDS = 'không|một|mốt|hai|ba|bốn|tư|năm|lăm|sáu|bảy|tám|chín|mười|mươi|trăm|linh|lẻ'
NUMBER = rf'(?:\d[\d.,]*|(?:{WORDS})(?:\s+(?:{WORDS}))*)'
UNIT = r'(?:nghìn|ngàn|triệu|tỷ|tỉ)'
MONEY = rf'(?<!\w){NUMBER}\s*{UNIT}(?:\s+{NUMBER}\s*{UNIT})*(?:\s+(?:rưỡi|{NUMBER}))?(?:\s*(?:đồng|vnđ|vnd|đ))?(?!\w)|(?<!\w){NUMBER}\s*(?:đồng|vnđ|vnd|đ)(?!\w)'
DAY_NUMBER = rf'(?:(?:hai|ba)\s+(?:mốt|một|hai|ba|bốn|tư|năm|lăm|sáu|bảy|tám|chín)|{NUMBER})'
TIME_PATTERN = rf'(?<!\w)(?:\d{{1,2}}[:h]\d{{2}}|{NUMBER}\s+(?:giờ(?:\s+(?:kém\s+{NUMBER}|rưỡi|{NUMBER})(?:\s+phút)?)?|rưỡi))(?:\s+(?:sáng|trưa|chiều|tối|đêm))?(?!\w)'
CAP = r'[A-ZÀ-ỸĐ][a-zà-ỹđ]+'
DIGIT_WORD = r'(?:không|một|mốt|hai|ba|bốn|tư|năm|lăm|sáu|bảy|tám|chín)'
RULES = {
    'money': re.compile(MONEY, re.I),
    'time': re.compile(TIME_PATTERN,re.I),
    'date': re.compile(rf'(?<!\w)(?:\d{{1,2}}[/-]\d{{1,2}}(?:[/-]\d{{2,4}})?|(?:ngày|hôm)\s+(?:mùng\s+)?{DAY_NUMBER}(?:\s+tháng\s+{NUMBER})?(?:\s+năm\s+\d{{4}})?|thứ\s+(?:hai|ba|tư|năm|sáu|bảy|[2-7])|chủ nhật)(?!\w)', re.I),
    'phone': re.compile(rf'(?<!\w)(?:(?:\+84|0)\d(?:[ .-]?\d){{8,9}}(?!\d)|không(?:[\s,.-]+{DIGIT_WORD}){{8,11}}(?!\w))', re.I),
    'name/contact': re.compile(rf'(?<!\w)(?i:anh|chị|cô|chú|bác|ông|bà|sếp|bạn|em)\s+{CAP}(?:\s+{CAP}){{0,3}}(?!\w)'),
    'address': re.compile(r'(?<!\w)(?:đường|phố|quận|huyện|phường|số nhà|địa chỉ)\s+[^,;.!?]{1,80}', re.I),
    'english_term': re.compile(r'(?<!\w)(?:google meet|microsoft teams|wi-fi|wifi|bluetooth|facebook|youtube|spotify|netflix|gmail|email|online|offline|zoom|tiktok|app)(?!\w)', re.I),
}
RELATIVE_PHRASES = ('ngay bây giờ','ngày hôm sau','ngày kia','ngày mai','ngày mốt',
                    'sáng mai','trưa mai','chiều mai','tối mai','sáng mốt','chiều mốt','tối mốt',
                    'sáng nay','trưa nay','chiều nay','tối nay','đêm nay','hôm nay','hôm qua','tối qua',
                    'tuần sau','tuần tới','tháng sau','tháng tới','năm sau','năm tới','bây giờ','mai','mốt')
RELATIVE_PATTERN = re.compile(r'(?<!\w)(?:'+'|'.join(map(re.escape, sorted(RELATIVE_PHRASES,key=len,reverse=True)))+r')(?!\w)', re.I)


def extract_entities(text):
    text = normalize(text)
    result = {}
    for kind,rule in RULES.items():
        spans=[]
        for match in rule.finditer(text):
            if kind=='date' and re.match(r'\s+lần\b',text[match.end():],re.I):
                continue
            span=dict(text=match.group(),start=match.start(),end=match.end())
            parser = {'money':parse_money,'time':parse_time,'date':parse_date}.get(kind)
            if parser:
                span['parser']=parser(span['text'])
            spans.append(span)
        result[kind]=spans
    return result


def entity_comparison(reference, hypothesis):
    ref, hyp = extract_entities(reference), extract_entities(hypothesis)
    errors = {}
    for kind in CATEGORIES:
        r, h = Counter(s['text'] for s in ref[kind]), Counter(s['text'] for s in hyp[kind])
        errors[kind] = dict(status='not_present' if not r else ('correct' if r == h else 'incorrect'),
                            reference_spans=ref[kind], hypothesis_spans=hyp[kind],
                            false_positive_only=not bool(r) and bool(h))
    return errors


def relative_time_spans(text):
    text = normalize(text)
    spans = []
    for match in RELATIVE_PATTERN.finditer(text):
        if match.group().casefold() == 'mai' and re.search(r'(?i)(?<!\w)(?:anh|chị|cô|chú|bác|ông|bà|sếp|bạn|em)(?:\s+(?:giáo|gái|trai|sĩ))?\s+$', text[:match.start()]):
            continue
        spans.append(dict(text=match.group(), start=match.start(), end=match.end(), canonical=match.group().casefold(),
                          uncertainties=['bare_mai_name_or_tomorrow'] if match.group().casefold()=='mai' else []))
    return spans


def relative_time_comparison(reference, hypothesis):
    r, h = relative_time_spans(reference), relative_time_spans(hypothesis)
    rc, hc = Counter(s['canonical'] for s in r), Counter(s['canonical'] for s in h)
    return dict(status='not_present' if not rc else ('correct' if rc == hc else 'incorrect'),
                reference_spans=r, hypothesis_spans=h, missing=list((rc-hc).elements()),
                extra=list((hc-rc).elements()), false_positive_only=not bool(rc) and bool(hc))


def small_number(words):
    digits = dict(không=0, một=1, mốt=1, hai=2, ba=3, bốn=4, tư=4, năm=5, lăm=5,
                  sáu=6, bảy=7, tám=8, chín=9)
    if not words:
        return None
    total, last, previous_digit = 0, 0, False
    for word in words:
        if word in digits:
            if previous_digit:
                return None  # Reject ambiguous adjacent digits, e.g. 'năm hai'.
            last = digits[word]
            previous_digit=True
        elif word == 'trăm':
            total += (last or 1)*100
            last = 0
            previous_digit=False
        elif word in ('mươi','mười'):
            total += (last or 1)*10
            last = 0
            previous_digit=False
        elif word not in ('linh','lẻ'):
            return None
    return total+last


def numeric_value(text):
    if re.fullmatch(r'\d+(?:[.,]\d+)*', text):
        try:
            if re.fullmatch(r'\d{1,3}(?:[.,]\d{3})+', text):
                return Decimal(re.sub(r'[.,]','',text))
            return Decimal(text.replace(',','.'))
        except InvalidOperation:
            return None
    value = small_number(text.split())
    return Decimal(value) if value is not None else None


def parse_money(text):
    text = normalize(text).casefold()
    text = re.sub(r'\s*(?:đồng|vnđ|vnd|đ)$','',text).strip()
    result=dict(value_vnd=None,has_fraction='rưỡi' in text,uncertainties=[],fraction_rule=None)
    units = {'nghìn':1000,'ngàn':1000,'triệu':1000000,'tỷ':1000000000,'tỉ':1000000000}
    total, previous_end, last_multiplier = Decimal(0), 0, float('inf')
    matches = list(re.finditer(r'\b(nghìn|ngàn|triệu|tỷ|tỉ)\b', text))
    if not matches:
        value = numeric_value(text)
        result['value_vnd']=int(value) if value is not None and value==int(value) else None
        if result['value_vnd'] is None:
            result['uncertainties'].append('invalid_or_noninteger_vnd')
        return result
    for match in matches:
        multiplier = units[match.group()]
        if multiplier >= last_multiplier:
            result['uncertainties'].append('non_descending_scales')
            return result
        amount = numeric_value(text[previous_end:match.start()].strip())
        if amount is None:
            result['uncertainties'].append('unsupported_number_words')
            return result
        if amount!=int(amount):
            result['has_fraction']=True
        total += amount*multiplier
        last_multiplier, previous_end = multiplier, match.end()
    tail=text[previous_end:].strip()
    if tail:
        result['has_fraction']=True
        if tail=='rưỡi' and len(matches)==1:
            total+=Decimal(last_multiplier)/2
            result['fraction_rule']='half_of_single_scale'
        elif tail=='rưỡi':
            result['uncertainties'].append('half_after_compound_scales')
            return result
        else:
            digit=numeric_value(tail)
            if digit is None or digit!=int(digit) or not 0<=digit<=9:
                result['uncertainties'].append('ambiguous_unsuffixed_tail')
                return result
            total+=digit*Decimal(last_multiplier)/10
            result['fraction_rule']='single_tail_digit_is_tenths_of_last_scale'
    first_scale=units[matches[0].group()]
    if total%first_scale:
        result['has_fraction']=True
    if total!=int(total):
        result['uncertainties'].append('noninteger_vnd')
    else:
        result['value_vnd']=int(total)
    return result


def money_value(text):
    return parse_money(text)['value_vnd']


def _int_number(text):
    value=numeric_value(text.strip())
    return int(value) if value is not None and value==int(value) else None


def parse_time(text):
    """Clock diagnostic only, no context AM/PM inference. kém subtracts minutes."""
    text=normalize(text).casefold()
    result=dict(clock=None,minutes_since_midnight=None,uncertainties=[])
    period_match=re.search(r'\s+(sáng|trưa|chiều|tối|đêm)$',text)
    period=period_match.group(1) if period_match else None
    if period_match:
        text=text[:period_match.start()]
    subtract=False
    digital=re.fullmatch(r'(\d{1,2})[:h](\d{2})',text)
    if digital:
        hour,minute=map(int,digital.groups())
    else:
        match=re.fullmatch(rf'({NUMBER})\s+(giờ|rưỡi)(?:\s+(kém\s+{NUMBER}|rưỡi|{NUMBER})(?:\s+phút)?)?',text,re.I)
        if not match:
            result['uncertainties'].append('unsupported_clock_form')
            return result
        hour=_int_number(match.group(1))
        tail=match.group(3) or ''
        if match.group(2)=='rưỡi' or tail=='rưỡi':
            minute=30
        elif tail.startswith('kém '):
            minute=_int_number(tail[4:])
            subtract=True
        elif tail:
            minute=_int_number(tail)
        else:
            minute=0
    if hour is None or minute is None or not 0<=hour<=23 or not 0<=minute<=59:
        result['uncertainties'].append('invalid_clock_range')
        return result
    if period in ('chiều','tối') and 1<=hour<12:
        hour+=12
    elif period=='sáng' and hour==12:
        hour=0
    elif period=='trưa' and hour<11:
        result['uncertainties'].append('ambiguous_noon_hour')
    elif period=='đêm':
        result['uncertainties'].append('ambiguous_night_day_boundary')
    if period is None and hour<=12:
        result['uncertainties'].append('daypart_unspecified')
    value=hour*60+(-minute if subtract else minute)
    if value<0:
        result['uncertainties'].append('previous_day_wrap')
    value%=24*60
    result.update(clock=f'{value//60:02d}:{value%60:02d}',minutes_since_midnight=value)
    return result


def parse_date(text):
    """Day/month diagnostic; no inferred year/month or two-digit-century guess."""
    text=normalize(text).casefold()
    result=dict(day=None,month=None,year=None,weekday_iso=None,uncertainties=[])
    weekday=re.fullmatch(r'thứ (hai|ba|tư|năm|sáu|bảy|[2-7])|chủ nhật',text)
    if weekday:
        names={'hai':1,'ba':2,'tư':3,'năm':4,'sáu':5,'bảy':6}
        word=weekday.group(1)
        result['weekday_iso']=7 if word is None else (int(word)-1 if word.isdigit() else names[word])
        return result
    digital=re.fullmatch(r'(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?',text)
    if digital:
        day,month=int(digital.group(1)),int(digital.group(2))
        year=int(digital.group(3)) if digital.group(3) else None
        if year is not None and year<100:
            result['uncertainties'].append('two_digit_year_century_unknown')
    else:
        match=re.fullmatch(rf'(?:ngày|hôm)\s+(?:mùng\s+)?({DAY_NUMBER})(?:\s+tháng\s+({NUMBER}))?(?:\s+năm\s+(\d{{4}}))?',text)
        if not match:
            result['uncertainties'].append('unsupported_date_form')
            return result
        day=_int_number(match.group(1))
        if day is None:
            parts=match.group(1).split()
            if len(parts)==2 and parts[0] in ('hai','ba'):
                ones=_int_number(parts[1])
                day=(20 if parts[0]=='hai' else 30)+ones if ones is not None else None
        month=_int_number(match.group(2)) if match.group(2) else None
        year=int(match.group(3)) if match.group(3) else None
    if day is None or not 1<=day<=31 or (month is not None and not 1<=month<=12):
        result['uncertainties'].append('invalid_calendar_range')
        return result
    if month is not None:
        try:
            date(year if year is not None and year>=100 else 2000,month,day)
        except ValueError:
            result['uncertainties'].append('invalid_calendar_date')
            return result
    if month is None:
        result['uncertainties'].append('month_unspecified')
    if year is None:
        result['uncertainties'].append('year_unspecified')
    result.update(day=day,month=month,year=year)
    return result


def relaxed_money_comparison(reference, hypothesis):
    r = extract_entities(reference)['money']
    h = extract_entities(hypothesis)['money']
    rp,hp=[s['parser'] for s in r],[s['parser'] for s in h]
    rv, hv = [s['value_vnd'] for s in rp], [s['value_vnd'] for s in hp]
    comparable = None not in rv and None not in hv
    fraction_mismatch=sum(s['has_fraction'] for s in rp)!=sum(s['has_fraction'] for s in hp)
    # Multiple amounts must not hide a missing fraction by moving the marker
    # to a different amount with an otherwise matching multiset of VND values.
    if comparable and Counter(rv)==Counter(hv):
        fraction_mismatch = fraction_mismatch or Counter((s['value_vnd'],s['has_fraction']) for s in rp)!=Counter((s['value_vnd'],s['has_fraction']) for s in hp)
    return dict(status='not_present' if not r else ('incorrect' if fraction_mismatch else ('unparseable' if not comparable else ('correct' if Counter(rv)==Counter(hv) else 'incorrect'))),
                reference_values_vnd=rv, hypothesis_values_vnd=hv, reference_spans=r, hypothesis_spans=h,
                reference_parses=rp,hypothesis_parses=hp,fraction_presence_mismatch=fraction_mismatch,
                comparable=comparable, false_positive_only=not bool(r) and bool(h))


def relaxed_time_comparison(reference,hypothesis):
    r,h=extract_entities(reference)['time'],extract_entities(hypothesis)['time']
    rv,hv=[s['parser']['clock'] for s in r],[s['parser']['clock'] for s in h]
    comparable=None not in rv and None not in hv
    return dict(status='not_present' if not r else ('unparseable' if not comparable else ('correct' if Counter(rv)==Counter(hv) else 'incorrect')),
                reference_clocks=rv,hypothesis_clocks=hv,reference_spans=r,hypothesis_spans=h,
                comparable=comparable,false_positive_only=not bool(r) and bool(h))


def parser_uncertainties(text):
    """Review flags, not new labels or rules. Includes bare-Mai ambiguity."""
    result=[]
    for kind,spans in extract_entities(text).items():
        for span in spans:
            reasons=span.get('parser',{}).get('uncertainties',[])
            if reasons:
                result.append(dict(kind=kind,text=span['text'],start=span['start'],end=span['end'],reasons=reasons))
    for span in relative_time_spans(text):
        if span['uncertainties']:
            result.append(dict(kind='relative_time',text=span['text'],start=span['start'],end=span['end'],reasons=span['uncertainties']))
    return result
