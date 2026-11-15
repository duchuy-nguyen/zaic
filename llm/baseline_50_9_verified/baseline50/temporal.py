"""Conservative Vietnamese temporal hints. No labels, state access, or tool execution.

Keep raw text and source spans. Calendar-relative resolution is permitted only
for the current utterance with a timezone-aware context clock. Historical user
utterances have no reliable utterance timestamp and must NOT use today's anchor.
"""
from datetime import datetime, timedelta, date, timezone
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

VERSION = 'vi-temporal-conservative-v1'
_DIGITS = {'không':0,'một':1,'mốt':1,'hai':2,'ba':3,'bốn':4,'tư':4,
           'năm':5,'lăm':5,'sáu':6,'bảy':7,'tám':8,'chín':9}
_NW = '|'.join(sorted([*_DIGITS, 'mười','mươi','linh','lẻ'],key=len,reverse=True))
_N = rf'(?:\d+|(?:{_NW})(?:\s+(?:{_NW})){{0,4}})'
_PER = r'(?:buổi\s+)?(?:rạng\s+sáng|sáng\s+sớm|nửa\s+đêm|sáng|trưa|chiều|tối|đêm|khuya)'
_REL = r'(?:hôm\s+nay|hôm\s+qua|hôm\s+kia|ngày\s+mai|ngày\s+kia|ngày\s+mốt|hôm\s+sau|hôm\s+trước|mai|nay|qua|mốt)'
_CLOCK = re.compile(rf'(?<![\w:/.-])(?P<h>{_N})\s*(?P<sep>giờ|h(?=\d|\s|$)|:|rưỡi)'
    rf'(?P<tail>(?:\s*kém\s+{_N}(?:\s*phút)?|\s*rưỡi|\s*(?:{_N})(?:\s*phút)?)?)'
    rf'(?P<sec>(?:\s*{_N}\s*giây|:\d{{1,2}})?)', re.I)
_DURATION = re.compile(rf'(?<!\w)(?P<n>{_N}|nửa|một\s+phần\s+tư|ba\s+phần\s+tư)\s*'
    r'(?P<u>giây|phút|tiếng|giờ|ngày|tuần)(?:\s*(?P<half>rưỡi))?(?!\w)',re.I)
_DATE = re.compile(rf'(?<!\w)(?P<rel>{_REL})(?!\w)|'
    r'(?<![\w/.-])(?P<d>\d{1,2})[/-](?P<m>\d{1,2})(?:[/-](?P<y>\d{4}))?(?![\w/.-])|'
    rf'(?<!\w)ngày\s+(?P<wd>{_N})\s+tháng\s+(?P<wm>\d{{1,2}}|mười\s+(?:một|hai)|mười|{"|".join(_DIGITS)})(?:\s+năm\s+(?P<wy>\d{{4}}))?(?!\w)',re.I)
_WEEKDAY = re.compile(r'(?<!\w)(?:thứ\s+(hai|ba|tư|năm|sáu|bảy|[2-7])|chủ\s+nhật)'
    r'(?:\s+(tuần\s+(?:này|sau|tới|trước)))?(?!\w)',re.I)
_PER_RE = re.compile(_PER,re.I)


def number(s):
    """Strict time-sized integers; never interpret a phone/money digit string as words."""
    s=' '.join(s.lower().split())
    if s.isdigit(): return int(s)
    ts=s.split()
    if len(ts)==1 and ts[0] in _DIGITS:return _DIGITS[ts[0]]
    if len(ts)==2 and ts[0]=='mười' and ts[1] in _DIGITS:return 10+_DIGITS[ts[1]]
    if ts==['mười']:return 10
    if len(ts) in (2,3) and ts[0] in _DIGITS and ts[1]=='mươi':
        v=10*_DIGITS[ts[0]]
        if len(ts)==3:
            if ts[2] not in _DIGITS:raise ValueError(s)
            v+=_DIGITS[ts[2]]
        return v
    if len(ts)==2 and ts[0] in ('linh','lẻ') and ts[1] in _DIGITS:return _DIGITS[ts[1]]
    raise ValueError(s)


def anchor(context):
    if not context or not isinstance(context.get('current_time'),str):return None
    try:
        dt=datetime.fromisoformat(context['current_time'].replace('Z','+00:00'))
        if dt.tzinfo is None:return None
        tz=context.get('timezone')
        if tz:
            try:dt=dt.astimezone(ZoneInfo(tz))
            except ZoneInfoNotFoundError:
                # Windows may lack the IANA database. These zones have a fixed
                # +07 offset in the supported contemporary contest dates.
                if tz not in ('Asia/Ho_Chi_Minh','Asia/Bangkok'):return None
                dt=dt.astimezone(timezone(timedelta(hours=7)))
            except ValueError:return None
        return dt
    except (ValueError,TypeError):return None


def clause(text,start,end):
    """Do not borrow a date/period across a sentence, semicolon, or newline."""
    left=max((text.rfind(c,0,start) for c in ';!?\n'),default=-1)+1
    right=min((p for c in ';!?\n' if (p:=text.find(c,end))>=0),default=len(text))
    # A dot between digits is not a sentence boundary (e.g.8.30 is left unsupported).
    for m in re.finditer(r'(?<!\d)\.|\.(?!\d)',text):
        if m.end()<=start:left=max(left,m.end())
        elif m.start()>=end:right=min(right,m.start());break
    return left,right


def qualifiers(text,start,end):
    l,r=clause(text,start,end); pre=text[l:start].lower(); post=text[end:r].lower()
    out={}
    if re.search(r'(?:không\s+phải|chẳng\s+phải)\s*(?:lúc\s*)?$',pre):out['negated']=True
    if re.search(r'(?:đừng|không\s+được|hủy|huỷ|bỏ)\b[^,;.!?]{0,35}$',pre):out['nonaffirmative_context']=True
    if re.search(r'(?:khoảng|tầm|cỡ|chừng|gần|xấp\s+xỉ)\s*(?:lúc\s*)?$',pre):out['approximate']=True
    if re.search(r'(?:không\s+muộn\s+hơn|chậm\s+nhất)\s*(?:lúc\s*)?$',pre):out['relation']='not_later_than'
    elif re.search(r'(?:không\s+sớm\s+hơn|sớm\s+nhất)\s*(?:lúc\s*)?$',pre):out['relation']='not_earlier_than'
    elif re.search(r'trước\s*(?:lúc\s*)?$',pre):out['relation']='before'
    elif re.search(r'sau\s*(?:lúc\s*)?$',pre):out['relation']='after'
    if re.search(r'(?<!\w)(?:à\s*(?:thôi|không)?|mà|đổi\s+(?:sang|thành)|thay\s+bằng)\s*(?:lúc\s*)?$',pre):out['correction_candidate']=True
    if re.search(r'(?:hàng|hằng|mỗi|cứ|định\s+kỳ|cách\s+ngày)\b',pre+' '+post):out['recurring_context']=True
    if re.search(r'(?:nếu|khi\s+nào|chỉ\s+khi)\b',pre):out['conditional_context']=True
    if re.search(r'(?:từ|giữa)\s*(?:lúc\s*)?$',pre) and re.search(rf'^\s*(?:{_PER}\s*)?(?:đến|tới|và|[-–])',post):out['range_boundary']='start'
    if re.search(r'(?:đến|tới|[-–])\s*(?:lúc\s*)?$',pre) and re.search(r'\b(?:từ|giữa)\b',pre):out['range_boundary']='end'
    return out


def date_hints(text,now):
    out=[]; offsets={'hôm nay':0,'nay':0,'hôm qua':-1,'qua':-1,'hôm kia':-2,'ngày mai':1,'mai':1,'ngày kia':2,'ngày mốt':2,'mốt':2}
    for m in _DATE.finditer(text):
        if m['rel']:
            relative=m['rel'].lower()
            # Bare "qua" is also a preposition, and Mai can be a contact name.
            before=text[max(0,m.start()-25):m.start()].lower()
            if relative=='qua' and not re.search(rf'{_PER}\s*$',before):continue
            if relative=='mai' and re.search(r'(?:anh|chị|cô|chú|bác|bạn|em|ông|bà|cho|gửi|gọi|tên)\s*$',before):continue
            if relative=='mai' and m['rel'][0].isupper() and m.start()>0 and not re.search(rf'{_PER}\s*$',before):continue
            if relative=='mai' and re.search(r'(?:gặp|nhắn|liên\s+hệ)\s*$',before):continue
        h={'kind':'date','source':m.group(),'span':[m.start(),m.end()],**qualifiers(text,*m.span())}
        try:
            if m['rel']:
                v=' '.join(m['rel'].lower().split())
                if v in offsets:
                    h['day_offset_from_utterance']=offsets[v]
                    if now:h['date']=(now.date()+timedelta(days=offsets[v])).isoformat()
                else:h['unresolved_anchor']='referenced_day'
            else:
                d=int(m['d']) if m['d'] else number(m['wd'])
                mo=int(m['m']) if m['m'] else number(m['wm'])
                y=m['y'] or m['wy']
                h.update(day=d,month=mo)
                if y or now:
                    y=int(y) if y else now.year
                    h['date']=date(y,mo,d).isoformat()
                    if not (m['y'] or m['wy']):h['year_assumed_from_context']=True
                else:
                    # Validate month/day without inventing a year. Feb29 stays possible.
                    date(2000,mo,d);h['unresolved_anchor']='year'
            out.append(h)
        except (ValueError,OverflowError):
            out.append({'kind':'unresolved_time','source':m.group(),'span':[m.start(),m.end()],'reason':'invalid_calendar_date'})
    weekdays={'hai':0,'ba':1,'tư':2,'năm':3,'sáu':4,'bảy':5}
    for m in _WEEKDAY.finditer(text):
        v=(m.group(1) or '').lower(); wd=6 if not v else (int(v)-2 if v.isdigit() else weekdays[v])
        h={'kind':'weekday','source':m.group(),'span':[m.start(),m.end()],'weekday_monday_zero':wd,**qualifiers(text,*m.span())}
        if now and m.group(2):
            wk=m.group(2).lower();shift=-7 if wk.endswith('trước') else (7 if wk.endswith(('sau','tới')) else 0)
            h['date']=(now.date()-timedelta(days=now.weekday())+timedelta(days=wd+shift)).isoformat()
        else:h['unresolved_anchor']='week_or_recurrence'
        out.append(h)
    return out


def period_near(text,start,end):
    """Accept an adjacent period before or after, never one across another clock."""
    l,r=clause(text,start,end)
    before=text[l:start].lower();after=text[end:r].lower()
    suffix=re.match(rf'^\s*(?:{_REL}\s+)?(?P<p>{_PER})(?!\w)',after)
    prefix=re.search(rf'(?<!\w)(?P<p>{_PER})(?:\s+{_REL})?(?:\s+lúc)?\s*$',before)
    vals=[x['p'].lower().removeprefix('buổi ') for x in (prefix,suffix) if x]
    return vals[0] if vals and len(set(vals))==1 else None


def hour24(hour,period):
    if period is None:return hour,hour==0 or hour>12
    if period in ('sáng','sáng sớm','rạng sáng'):
        if hour==12:return 0,True
        return hour,0<=hour<12
    if period=='trưa':
        if hour==12:return 12,True
        if hour==1:return 13,True
        return hour,False
    if period in ('chiều','tối'):
        if 1<=hour<12:return hour+12,True
        if 13<=hour<=23:return hour,True
        return hour,False
    if period in ('đêm','khuya','nửa đêm'):
        if hour==12:return 0,True
        if hour in (0,1,2,3,4) or hour>=19:return hour,True
        if 7<=hour<=11:return hour+12,True
    return hour,False


def clock_hints(text,now,dates):
    out=[]
    for m in _CLOCK.finditer(text):
        try:
            hour=number(m['h']); tail=m['tail'].strip().lower();second=0;minute=0
            if hour>23:raise ValueError('hour')
            if m['sep'].lower()=='rưỡi':minute=30
            elif tail=='rưỡi':minute=30
            elif tail:
                negative=tail.startswith('kém')
                value=re.sub(r'^kém\s+|\s*phút$','',tail).strip()
                minute=number(value)
                if not 0<=minute<=59:raise ValueError('minute')
                if negative:minute=-minute
            if m['sec']:second=number(re.sub(r'^:|\s*giây$','',m['sec']).strip())
            if not 0<=second<=59:raise ValueError('second')
            period=period_near(text,*m.span())
            l,r=clause(text,*m.span());pre=text[l:m.start()].lower();post=text[m.end():r].lower()
            if re.search(r'phần\s*$',pre):continue
            if not period and m['sep'].lower()=='giờ' and (re.match(r'^\s*nữa\b',post) or re.search(r'(?:sau|trong|mỗi|cứ|cách)\s*$',pre)):
                continue
            hour,resolved=hour24(hour,period)
            total=hour*60+minute;day_shift,total=divmod(total,24*60);hh,mm=divmod(total,60)
            # Without a day period, "9 giờ kém10" also has a21h interpretation.
            h={'kind':'clock','source':m.group(),'span':[m.start(),m.end()],
               'hour':hh,'minute':mm,'period':period,'meridiem_resolved':resolved,**qualifiers(text,*m.span())}
            if second:h['second']=second
            if day_shift:h['day_shift_from_named_hour']=day_shift
            l,r=clause(text,*m.span())
            local=[d for d in dates if d['kind'] in ('date','weekday') and l<=d['span'][0] and d['span'][1]<=r]
            dated=[d for d in local if 'date' in d]
            if len(local)==1 and len(dated)==1 and resolved and now:
                d=dated[0];dt=datetime.combine(date.fromisoformat(d['date']),datetime.min.time(),tzinfo=now.tzinfo)
                dt+=timedelta(days=day_shift,hours=hh,minutes=mm,seconds=second)
                h['datetime']=dt.isoformat();h['date_source']=d['source']
            if not resolved:h['unresolved_anchor']='day_period'
            out.append(h)
        except (ValueError,OverflowError):
            out.append({'kind':'unresolved_time','source':m.group(),'span':[m.start(),m.end()],'reason':'invalid_or_unsupported_clock'})
    return out


def duration_hints(text,now,clocks):
    out=[];matches=[]
    units={'giây':1,'phút':60,'giờ':3600,'tiếng':3600,'ngày':86400,'tuần':604800}
    for m in _DURATION.finditer(text):
        # A clock can contain "8 giờ30phút". Neither component is a duration.
        if any(m.start()<h['span'][1] and m.end()>h['span'][0] for h in clocks):continue
        if m['u'].lower()=='tiếng' and re.match(r'\s+(?:đàn|chim|việt|anh|nhật|hàn|trung)\b',text[m.end():],re.I):continue
        try:
            n=m['n'].lower();v={'nửa':.5,'một phần tư':.25,'ba phần tư':.75}.get(n)
            if v is None:v=number(n)
            if m['half']:v+=.5
            if v<0:continue
            matches.append((m,int(v*units[m['u'].lower()])))
        except ValueError:continue
    # Merge adjacent components: "một tiếng hai mươi phút" is one80min span.
    grouped=[]
    for m,seconds in matches:
        if grouped and re.fullmatch(r'\s*(?:và\s*)?',text[grouped[-1][1]:m.start()]):
            grouped[-1][1]=m.end();grouped[-1][2]+=seconds
        else:grouped.append([m.start(),m.end(),seconds])
    for start,end,seconds in grouped:
        l,r=clause(text,start,end);pre=text[l:start].lower();post=text[end:r].lower()
        h={'kind':'duration','source':text[start:end],'span':[start,end],'seconds':seconds,**qualifiers(text,start,end)}
        if re.search(r'(?:trong\s+vòng|trong)\s*$',pre) and not re.match(r'^\s*nữa',post):h['relation']='within'
        elif re.search(r'(?:sớm\s+hơn|muộn\s+hơn|hoãn|lùi|đẩy\s+sớm|trước)\s*$',pre):h.update(relation='offset_from_reference',unresolved_anchor='referenced_event_or_time')
        elif re.search(r'sau\s*$',pre) or re.match(r'^\s*nữa\b',post):
            h['relation']='after'
            # "sau10phút kể từ cuộc họp" must not be anchored to NOW.
            if re.match(r'^\s*(?:kể\s+từ|tính\s+từ|sau\s+khi|trước\s+khi)',post):h['unresolved_anchor']='referenced_event_or_time'
            elif now and not any(h.get(k) for k in ('recurring_context','nonaffirmative_context','negated','conditional_context')):
                h['anchor']='context.current_time';h['datetime']=(now+timedelta(seconds=seconds)).isoformat()
            else:h['unresolved_anchor']='utterance_time_or_condition'
        elif re.search(r'\b(trước|sau)\b[^,;.!?]{1,60}$',pre):
            ref=re.search(r'\b(trước|sau)\s+([^,;.!?]{1,60})$',pre)
            h.update(relation='offset_from_reference',unresolved_anchor='referenced_event_or_time')
            if ref:h.update(offset_direction=-1 if ref[1]=='trước' else 1,reference_text=ref[2].strip())
        elif re.match(r'^\s*trước\b',post):h.update(relation='before',unresolved_anchor='utterance_time_or_referenced_event')
        else:h['unresolved_anchor']='duration_role_or_reference'
        out.append(h)
    return out


def temporal_hints(text,context=None):
    # Caller already normalizes NFC/whitespace; keep source offsets in that text.
    now=anchor(context);dates=date_hints(text,now);clocks=clock_hints(text,now,dates)
    hints=dates+clocks+duration_hints(text,now,clocks)
    occupied=[h['span'] for h in hints]
    for m in re.finditer(r'(?<!\w)(?:nửa\s+đêm|giữa\s+trưa|lát\s+nữa|tí\s+nữa|chút\s+nữa|giờ\s+đó|lúc\s+ấy|đúng\s+giờ\s+[^,;.!?]+|ngay\s+bây\s+giờ|bây\s+giờ)(?!\w)',text,re.I):
        if any(m.start()<b and m.end()>a for a,b in occupied):continue
        source=m.group();s=source.lower();h={'kind':'temporal_reference','source':source,'span':[m.start(),m.end()],**qualifiers(text,*m.span())}
        if s in ('bây giờ','ngay bây giờ'):
            h['anchor']='context.current_time'
            if now:h['datetime']=now.isoformat()
        elif s in ('nửa đêm','giữa trưa'):h.update(hour=0 if s=='nửa đêm' else 12,minute=0,unresolved_anchor='calendar_day')
        else:h['unresolved_anchor']='context_or_exact_time'
        hints.append(h)
    # Keep token overhead bounded on ordinary commands; no synthetic time defaults.
    return sorted(hints,key=lambda h:(h['span'][0],h['span'][1],h['kind']))
