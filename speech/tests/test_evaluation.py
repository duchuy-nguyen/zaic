import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from lifemate_speech.metrics import aggregate, edit_counts, score_utterance
from lifemate_speech.io_utils import speech_work_dir
from lifemate_speech.entities import (entity_comparison, extract_entities, money_value,
                                     parse_money,parse_time,parse_date,parser_uncertainties,
                                     relaxed_money_comparison,relaxed_time_comparison,relative_time_comparison)

spec = importlib.util.spec_from_file_location('eval_asr',ROOT/'scripts'/'eval_asr.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class EvaluationTests(unittest.TestCase):
    def test_known_edits_and_empty_reference(self):
        row = score_utterance('a b c','a x c d')
        self.assertEqual(row['edit_ops'],{'S':1,'D':0,'I':1})
        self.assertAlmostEqual(row['wer'],2/3)
        self.assertEqual(score_utterance('a b','')['edit_ops'],{'S':0,'D':2,'I':0})
        self.assertEqual(score_utterance('','')['wer'],0)
        self.assertEqual(score_utterance('','x y')['wer'],2)
        self.assertEqual(edit_counts('abc','axc')['distance'],1)
        self.assertAlmostEqual(score_utterance('a b','a x')['cer'],1/3)
        summary = aggregate([score_utterance('a','x'),score_utterance('a b c','a b c')])
        self.assertEqual(summary['wer_micro'],.25)
        self.assertEqual(summary['wer_macro'],.5)

    def test_money_relaxation_separate_from_strict_and_wer(self):
        ref, hyp = 'nạp 50 nghìn','nạp năm mươi ngàn'
        self.assertEqual(entity_comparison(ref,hyp)['money']['status'],'incorrect')
        self.assertEqual(relaxed_money_comparison(ref,hyp)['status'],'correct')
        self.assertGreater(score_utterance(ref,hyp)['wer'],0)
        self.assertEqual(money_value('hai triệu năm trăm nghìn'),2500000)
        self.assertEqual(money_value('1,5 triệu'),1500000)
        self.assertEqual(money_value('50.000 đồng'),50000)
        self.assertEqual(money_value('một triệu rưỡi'),1500000)
        self.assertEqual(relaxed_money_comparison('50 nghìn','60 nghìn')['status'],'incorrect')

    def test_v3_fractional_money_spans_and_omission_guard(self):
        for text,value in [('hai triệu tám',2800000),('một triệu hai',1200000),
                           ('ba mươi hai triệu rưỡi',32500000),('2,8 triệu',2800000)]:
            with self.subTest(text=text):
                spans=extract_entities('chuyển '+text+' nhé')['money']
                self.assertEqual(len(spans),1)
                self.assertEqual(spans[0]['text'],text)
                self.assertEqual(spans[0]['parser']['value_vnd'],value)
                self.assertTrue(spans[0]['parser']['has_fraction'])
        for fractional,base in [('hai triệu tám','hai triệu'),('một triệu hai','một triệu'),
                                ('ba mươi hai triệu rưỡi','ba mươi hai triệu'),
                                ('hai triệu không','hai triệu'),('hai triệu tám mươi','hai triệu')]:
            for ref,hyp in [(fractional,base),(base,fractional)]:
                result=relaxed_money_comparison(ref,hyp)
                self.assertEqual(result['status'],'incorrect')
                self.assertTrue(result['fraction_presence_mismatch'])
        self.assertEqual(relaxed_money_comparison('hai triệu tám','2,8 triệu')['status'],'correct')
        self.assertEqual(relaxed_money_comparison('một triệu rưỡi','1,5 triệu')['status'],'correct')
        swapped=relaxed_money_comparison('hai triệu tám và 50 nghìn','2800000 đồng và 0,05 triệu')
        self.assertEqual(swapped['status'],'incorrect')
        self.assertTrue(swapped['fraction_presence_mismatch'])
        unsupported=extract_entities('hai triệu tám mươi')['money'][0]
        self.assertEqual(unsupported['text'],'hai triệu tám mươi')
        self.assertIsNone(unsupported['parser']['value_vnd'])
        self.assertEqual(relaxed_money_comparison('hai triệu tám mươi','hai triệu tám mươi')['status'],'unparseable')
        self.assertEqual(relaxed_money_comparison('ba mươi hai triệu rưỡi','ba mươi hai triệu rưỡi')['status'],'correct')
        self.assertEqual(relaxed_money_comparison('hai triệu tám mươi','hai triệu tám mươi')['status'],'unparseable')
        self.assertEqual(relaxed_money_comparison('ba mươi hai triệu rưỡi','ba mươi hai triệu rưỡi')['status'],'correct')
        self.assertIsNone(money_value('hai triệu năm trăm nghìn rưỡi'))
        self.assertIsNone(money_value(''))

    def test_v3_clock_and_date(self):
        for text,value in [('8 giờ kém 15','07:45'),('tám giờ kém mười lăm','07:45'),
                           ('8 giờ rưỡi','08:30'),('tám rưỡi','08:30'),
                           ('tám giờ mười lăm','08:15'),('8 giờ ba mươi','08:30'),
                           ('8 giờ kém 15 tối','19:45')]:
            with self.subTest(text=text):
                self.assertEqual(extract_entities(text)['time'][0]['text'],text)
                self.assertEqual(parse_time(text)['clock'],value)
        self.assertEqual(relaxed_time_comparison('8 giờ kém 15','7:45')['status'],'correct')
        self.assertIsNone(parse_time('8 giờ kém 90')['clock'])
        self.assertIn('previous_day_wrap',parse_time('không giờ kém mười lăm')['uncertainties'])
        for text in ['ngày hai lần','ngày ba lần','mỗi ngày 2 lần','uống thuốc ngày hai mươi lần']:
            self.assertEqual(extract_entities(text)['date'],[])
        self.assertEqual(extract_entities('hôm hai chín')['date'][0]['text'],'hôm hai chín')
        self.assertEqual(parse_date('hôm hai chín')['day'],29)
        day=parse_date('ngày mùng một tháng mười')
        self.assertEqual((day['day'],day['month']),(1,10))
        self.assertEqual(extract_entities('ngày mùng một tháng mười')['date'][0]['text'],'ngày mùng một tháng mười')
        self.assertIn('year_unspecified',day['uncertainties'])
        self.assertIn('invalid_calendar_date',parse_date('31/2')['uncertainties'])

    def test_v3_mai_uncertainty(self):
        for title in ('cô','chị','bác','anh','cô giáo','chị gái'):
            self.assertEqual(relative_time_comparison(title+' Mai','')['status'],'not_present')
        self.assertEqual(relative_time_comparison('ngày mai','Ngày mai')['status'],'correct')
        self.assertFalse(parser_uncertainties('ngày mai'))
        self.assertTrue(parser_uncertainties('Mai Anh gọi tôi'))

    def test_weak_categories_and_relative_time(self):
        text = 'Gọi cô Mai 0912345678 lúc 08:30 ngày 5/11, ở đường Lê Lợi, bật Wi-Fi, trả 50 nghìn.'
        found = extract_entities(text)
        for category in found:
            self.assertTrue(found[category],category)
        rel = relative_time_comparison('Chiều nay gọi cô Mai, mai gặp nhau','chiều nay gọi cô mai, mốt gặp nhau')
        self.assertEqual(rel['missing'],['mai'])
        self.assertEqual(rel['extra'],['mốt'])
        self.assertEqual(rel['status'],'incorrect')
        self.assertEqual(relative_time_comparison('mai mai','mai')['missing'],['mai'])

    def test_join_history_missing_denominator_and_input_rejection(self):
        work=speech_work_dir()
        work.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work,prefix='unit_eval_') as tmp:
            root = Path(tmp)
            (root/'data'/'train').mkdir(parents=True)
            gold = dict(id='one',audio='same.wav',audio_text='current text',history=[
                dict(role='user',audio='same.wav',audio_text='first history'),
                dict(role='assistant',text='ignore'),
                dict(role='user',audio='same.wav',audio_text='last history')])
            (root/'data'/'train'/'train.jsonl').write_text(json.dumps(gold)+'\n',encoding='utf-8')
            rows = [dict(sample_id='one',split='train',kind='history',turn_index=2,text_raw='last history',error=None,asr_latency_s=1),
                    dict(sample_id='one',split='train',kind='current',turn_index=None,text_raw='current text',error=None,asr_latency_s=1)]
            path = root/'transcripts.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            result = evaluation.evaluate(root/'data',[path],root/'eval',scope='transcript_samples')
            self.assertEqual(result['counts']['total_audio'],3)
            self.assertEqual(result['counts']['missing'],1)
            self.assertAlmostEqual(result['wer_cer']['all']['wer_micro'],1/3)
            details = [json.loads(line) for line in (root/'eval'/'per_utterance.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(next(r for r in details if r['turn_index']==2)['wer'],0)
            self.assertEqual(next(r for r in details if r['turn_index']==0)['edit_ops']['D'],2)
            self.assertEqual(result['latency']['incomplete_sample_ids'],['one'])
            rows.append(rows[0])
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            with self.assertRaises(ValueError):
                evaluation.evaluate(root/'data',[path],root/'bad',scope='transcript_samples')
            rows = [dict(rows[0],split='public_test')]
            path.write_text(json.dumps(rows[0])+'\n',encoding='utf-8')
            with self.assertRaises(ValueError):
                evaluation.evaluate(root/'data',[path],root/'public',scope='transcript_samples')


if __name__=='__main__':
    unittest.main()
