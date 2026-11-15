"""Synthetic language tests: no competition ID, transcript, label, or state."""
import unittest
from baseline50.temporal import temporal_hints
from baseline50.policy import enriched_body, TfidfExamples, number_hints, legacy_number_hints

CTX={'current_time':'2028-02-28T23:40:00+07:00','timezone':'Asia/Ho_Chi_Minh'}

class TemporalTests(unittest.TestCase):
    def clocks(self,s,ctx=None):return [h for h in temporal_hints(s,ctx) if h['kind']=='clock']

    def test_clock_forms(self):
        cases=[('8 giờ 30 tối',20,30),('8 giờ 15 sáng',8,15),('21 giờ 45 phút',21,45),
            ('tám giờ ba mươi tối',20,30),('tám giờ rưỡi tối',20,30),('tám rưỡi tối',20,30),
            ('chín giờ kém mười tối',20,50),('chín giờ kém mười lăm sáng',8,45),
            ('tám giờ linh năm sáng',8,5),('tám giờ lẻ năm sáng',8,5),
            ('08:30',8,30),('8h30 tối',20,30),('tối mai lúc 8 giờ 30',20,30),
            ('12 giờ trưa',12,0),('12 giờ đêm',0,0),('1 giờ khuya',1,0),('0 giờ kém mười',23,50)]
        for s,h,m in cases:
            with self.subTest(s=s):
                hs=self.clocks(s);self.assertEqual(len(hs),1,hs);self.assertEqual((hs[0]['hour'],hs[0]['minute']),(h,m))

    def test_minutes_grid(self):
        for hour in range(0,24):
            for minute in (0,1,5,15,29,30,45,59):
                h=self.clocks(f'{hour} giờ {minute} phút')[0]
                self.assertEqual((h['hour'],h['minute']),(hour,minute))

    def test_kem_grid(self):
        for hour in (1,7,10,12,19,23):
            for minute in (5,15,20,45):
                h=self.clocks(f'{hour} giờ kém {minute} phút')[0]
                expected=(hour*60-minute)%(24*60)
                self.assertEqual(h['hour']*60+h['minute'],expected)

    def test_dates_and_rollover(self):
        hs=temporal_hints('mai lúc 8 giờ sáng',CTX)
        self.assertEqual(next(h['datetime'] for h in hs if h['kind']=='clock'),'2028-02-29T08:00:00+07:00')
        hs=temporal_hints('ngày mốt lúc 8 giờ tối',CTX)
        self.assertEqual(next(h['datetime'] for h in hs if h['kind']=='clock'),'2028-03-01T20:00:00+07:00')
        hs=temporal_hints('ngày 29 tháng hai năm 2028 lúc 8 giờ sáng',CTX)
        self.assertEqual(next(h['datetime'] for h in hs if h['kind']=='clock'),'2028-02-29T08:00:00+07:00')
        self.assertTrue(any(h.get('date')=='2028-02-29' for h in temporal_hints('29/02/2028',CTX)))
        self.assertTrue(any(h.get('reason')=='invalid_calendar_date' for h in temporal_hints('31/02/2028',CTX)))

    def test_durations(self):
        cases=[('sau nửa tiếng',1800),('sau một tiếng rưỡi',5400),('sau một tiếng hai mươi phút',4800),
            ('sau 10 phút',600),('90 giây nữa',90),('hai ngày nữa',172800),('sau một phần tư giờ',900)]
        for s,v in cases:
            with self.subTest(s=s):
                hs=temporal_hints(s,CTX);d=[h for h in hs if h['kind']=='duration']
                self.assertEqual(len(d),1,hs);self.assertEqual(d[0]['seconds'],v)
                self.assertIn('datetime',d[0],d)
        h=temporal_hints('sau 30 phút',CTX)[0]
        self.assertEqual(h['datetime'],'2028-02-29T00:10:00+07:00')
        self.assertFalse(self.clocks('sau một giờ hai mươi phút'))

    def test_ambiguous_and_history_no_anchor(self):
        for s in ['tám rưỡi','mai lúc tám giờ','giờ đó','lát nữa','trước cuộc họp 15 phút']:
            with self.subTest(s=s):
                self.assertFalse(any('datetime' in h for h in temporal_hints(s)))
        for s in ['nếu rảnh thì sau 10 phút','mỗi 8 giờ','sau 10 phút kể từ cuộc họp']:
            self.assertFalse(any('datetime' in h for h in temporal_hints(s,CTX)),s)
        self.assertFalse(any('datetime' in h for h in temporal_hints('8 giờ',CTX)))
        view={'id':'synthetic','context':{**CTX,'installed_apps':{}},'latest_text':'mai lúc 8 giờ sáng',
              'history':[{'role':'user','text':'mai lúc 8 giờ sáng'}]}
        body=enriched_body(view,TfidfExamples([]))
        hs=body['number_time_hints_not_new_requests']
        self.assertTrue(any('datetime' in h for h in hs['current']))
        self.assertFalse(any('datetime' in h for h in hs['history'][0]['hints']))

    def test_boundaries_negation_correction(self):
        hs=self.clocks('không phải 8 giờ sáng mà 9 giờ sáng')
        self.assertEqual(len(hs),2);self.assertTrue(hs[0]['negated']);self.assertTrue(hs[1]['correction_candidate'])
        self.assertEqual(self.clocks('trước 8 giờ sáng')[0]['relation'],'before')
        self.assertEqual(self.clocks('chậm nhất lúc 8 giờ sáng')[0]['relation'],'not_later_than')
        self.assertTrue(self.clocks('khoảng 8 giờ sáng')[0]['approximate'])
        hs=self.clocks('từ 8 giờ sáng đến 10 giờ sáng')
        self.assertEqual([h.get('range_boundary') for h in hs],['start','end'])
        for s in ['24 giờ 00 phút','8 giờ 60 phút','8 giờ kém 80 phút','99:99']:
            self.assertFalse(self.clocks(s),s)

    def test_no_false_clock_or_money_change(self):
        for s in ['gọi cô Mai qua Zalo','sau đó gửi tin','đọc lịch trước rồi đặt xe','mua hai tiếng đàn','số điện thoại 0912345678']:
            self.assertEqual(temporal_hints(s,CTX),[],s)
        for s in ['một triệu hai','hai trăm nghìn đồng','8 giờ tối chuyển ba triệu','không chuyển một triệu']:
            old=[h for h in legacy_number_hints(s) if h['kind']=='money']
            new=[h for h in number_hints(s,CTX) if h['kind']=='money']
            self.assertEqual(old,new)

    def test_timezone_not_machine_clock(self):
        c={'current_time':'2028-02-28T18:40:00Z','timezone':'Asia/Ho_Chi_Minh'}
        self.assertEqual(temporal_hints('bây giờ',c)[0]['datetime'],'2028-02-29T01:40:00+07:00')
        self.assertFalse(any('datetime' in h for h in temporal_hints('mai lúc 8 giờ sáng',{'current_time':'2028-01-01T00:00:00'})))

    def test_entity_correction_and_seconds_regressions(self):
        self.assertFalse(any(h['kind']=='date' for h in temporal_hints('Hẹn gặp Mai lúc tám giờ sáng',CTX)))
        self.assertFalse(any(h.get('correction_candidate') for h in self.clocks('8 giờ sáng ngày mai và 9 giờ tối ngày mốt')))
        self.assertEqual(self.clocks('08:30:15')[0]['second'],15)
        self.assertFalse(self.clocks('08:30:99'))
        h=temporal_hints('trước cuộc họp mười lăm phút',CTX)[0]
        self.assertEqual(h['offset_direction'],-1);self.assertEqual(h['reference_text'],'cuộc họp')
        self.assertNotIn('datetime',h)

if __name__=='__main__':unittest.main()
