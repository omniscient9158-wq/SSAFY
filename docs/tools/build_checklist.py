"""Fill the supplied checklist without changing its four-page layout.
PDF authoring environment: reportlab, pdfplumber, pypdf, pillow.
"""
from pathlib import Path
from io import BytesIO
from html import escape
from collections import Counter
import re
import pdfplumber
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / '2회차_셀프·상호_체크리스트.pdf'
OUT = ROOT
pdfmetrics.registerFont(TTFont('Korean', 'C:/Windows/Fonts/malgun.ttf'))
pdfmetrics.registerFont(TTFont('KoreanBold', 'C:/Windows/Fonts/malgunbd.ttf'))
INK = HexColor('#243044')
GREEN = HexColor('#DCF0CE')
GRAY = HexColor('#E4E7EB')
W,H = 841.8898,595.2756
ROWS = {}
for line in (ROOT / 'docs/04_체크리스트_현재상태.md').read_text(encoding='utf-8').splitlines():
    f=[x.strip() for x in line.strip('|').split('|')]
    if len(f)==5 and f[0].isdigit() and f[3] in {'통과','실패','미확인','해당없음'}:
        ROWS[int(f[0])] = f
assert len(ROWS)==27
COUNTS=Counter(v[3] for v in ROWS.values())

# Concise cell entries; detailed records remain in the Markdown checklist.
NOTES={
1:('01·02 설계서\n06 검증 결과','Train 100건 색인\nTest 색인 제외'),
2:('04 체크리스트\n06 검증 결과','재실행: 신규 0건\n메타 100 / 문헌 9'),
3:('03 출처 검토\n04 체크리스트','검색어·포함/제외 기록\n선별과 승인은 분리'),
4:('04 체크리스트\n06 검증 결과','동일 질문 재실행\n수집 캐시 사용'),
5:('02 설계서\n04 체크리스트','1000/200: 3청크\n150/30: 7청크'),
6:('10_taurine_vector_\nevidence.png','Top-k 1: 1건\nTop-k 3: 2건'),
7:('src/rag_graph.py\n부록 10: 그래프 코드','retrieve → generate\n두 노드 구성'),
8:('11_taurine_answer_\nui.png · 부록 5','인용 번호·제공 근거\n원출처 링크 대조'),
9:('04 체크리스트\n06 검증 결과','무근거 pKa 질문\n생성 보류 확인'),
10:('04_compound_\nswitch.png','항목 전환 후 이력\n분리·복원 확인'),
11:('05_condition_\nsearch.png · 부록 9','분자량≤300, LogP<2\nTrain 3,175건'),
12:('12_measurement_\ncomparison.png','LogS·문헌값의\n단위·온도 구분'),
13:('13_missing_key_\nnotice.png / 14_empty_\nindex_notice.png','키·색인 누락 안내\n기존 격리 실행 증빙'),
14:('01 요구사항 정의서','교육용 데이터 탐색\n건강·금융 조언 제외'),
15:('01 요구사항 정의서','F2013 미선택\nStreamlit 꾸미기만 적용'),
16:('01 요구사항 정의서','F2014 미선택'),
17:('06 검증 결과\nrag_comparison_vector_\n20261022.json','고정 5문항 비교\n오류·보류 사례 기록'),
18:('06 검증 결과','질문·답변·인용·시간\n비교 결과 포함'),
19:('01 요구사항 정의서','F2016 미선택'),
20:('aspirin_browser_\ntimings.json','첫 생성 4.85초\n후속 생성 18.72초'),
21:('13_missing_key_\nnotice.png\n제출 ZIP 검사','실제 .env 제외\n키 문자열 포함 검사'),
22:('src/README.md','수집·색인·검색·생성\nUI 모듈 분리'),
23:('external_request_\npolicy_check.json','호출 간격·재시도 확인\nWikimedia 신규 미실행'),
24:('03 출처 검토 보고서\n승인 출처 CSV','DOI·원출처·이용 조건\n문서별 기록'),
25:('03 출처 검토 보고서\n04 체크리스트','승인 14종/17발췌\n색인 6종/9발췌'),
26:('16_home_updated.jpg\n17_taurine_updated.jpg\n18_local_evidence_\nupdated.jpg','새 화면 3장·기존 증빙\n부록 및 파일 첨부'),
27:('10 제출물 점검표','개인 제출 문서 포함\n1인 팀: 상호 양식 제외'),
}
SHOTS=[
('16_home_updated.jpg','수정 후 첫 화면','2026-10-23 새 캡처. 테마·검색 카드·데이터 범위 접는 영역. 항목 26.'),
('17_taurine_updated.jpg','타우린 실험 기록','2026-10-23 새 캡처. B-524 실험 LogS -0.19와 참고 눈금. 항목 26.'),
('18_local_evidence_updated.jpg','출처 확인 문헌','2026-10-23 새 캡처. 로컬 검색 2건, 값·단위·온도와 원출처 링크. 항목 8·24·26.'),
('10_taurine_vector_evidence.png','의미 유사도 검색','UI 수정 전 기능 증빙. Top-k 비교 기록은 04 체크리스트를 함께 확인. 항목 6·8.'),
('11_taurine_answer_ui.png','생성 답변과 인용','UI 수정 전 기능 증빙. 답변의 근거 번호와 원문 발췌 대조. 이번에 생성 API를 재호출하지 않음. 항목 7·8.'),
('12_measurement_comparison.png','문헌 기록 비교','UI 수정 전 기능 증빙. 실험 LogS와 문헌의 단위·온도를 구분. 항목 12.'),
('13_missing_key_notice.png','키 누락 안내','UI 수정 전 격리 실행 증빙. 실제 키 값 비노출. 항목 13·21.'),
('14_empty_index_notice.png','빈 색인 안내','UI 수정 전 격리 실행 증빙. 기존 작업 색인을 삭제하지 않고 확인. 항목 13.'),
('05_condition_search.png','조건 검색','UI 수정 전 기능 증빙. 분자량 300 이하·LogP 2 미만, Train 3,175건. 항목 11.'),
]

def box_text(c, text, bbox, size=7.2):
    x0,t,x1,b=bbox
    for s in [size,size-.3,size-.6,6.2]:
        style=ParagraphStyle('cell',fontName='Korean',fontSize=s,leading=s*1.35,textColor=INK,wordWrap='CJK')
        p=Paragraph(escape(text).replace('\n','<br/>'),style)
        _,ph=p.wrap(x1-x0-9,b-t-6)
        if ph<=b-t-6:
            p.drawOn(c,x0+4.5,H-t-4-ph)
            return
    raise ValueError(f'Text overflows cell: {text}')

def tick(c, word):
    x,y=word['x0']+.8,H-word['bottom']+1.7
    c.setStrokeColor(INK);c.setLineWidth(1)
    c.lines([(x,y+2,x+1.7,y),(x+1.7,y,x+5.3,y+5.5)])

pl=pdfplumber.open(SOURCE)
original=PdfReader(SOURCE)
assert original.get_fields() is None  # supplied source is a static form
for mutual in [False]:
    writer=PdfWriter()
    for page_i,(p,source_page) in enumerate(zip(pl.pages,original.pages)):
        buf=BytesIO();c=canvas.Canvas(buf,pagesize=(W,H));c.setFillColor(INK)
        c.setFillColor(HexColor('#FFFFFF'));c.rect(643,H-30,169,15,fill=1,stroke=0)
        c.setFillColor(INK);c.setFont('Korean',7.5)
        c.drawRightString(808,H-26,'앱 작성자 김지은     점검자 김지은 (셀프)')
        if page_i==0:
            type_words=[w for w in p.extract_words() if w['text']=='□' and 95<w['top']<115]
            tick(c,type_words[1 if mutual else 0])
            form=p.find_tables()[0]
            box_text(c,'미실시 · 동료 점검 후 작성' if mutual else '2026-10-23 · 기존 실행 기록 포함',form.rows[3].cells[1],8)
            box_text(c,'김지은 · 서울 08반',form.rows[1].cells[1],8)
            box_text(c,'김지은 (셀프 점검)',form.rows[2].cells[1],8)
        table=p.find_tables()[1 if page_i==0 else 0]
        for row,values in zip(table.rows,table.extract()):
            if not values[0] or not values[0].strip().isdigit(): continue
            n=int(values[0]); result_cell=row.cells[4]
            if not mutual:
                squares=[w for w in p.crop(result_cell).extract_words() if w['text']=='□']
                assert len(squares)==4
                tick(c,squares[['통과','실패','미확인','해당없음'].index(ROWS[n][3])])
                box_text(c,NOTES[n][0],row.cells[5])
                box_text(c,NOTES[n][1],row.cells[6])
        if page_i==3:
            if not mutual:
                cells=p.find_tables()[1].rows[1].cells
                nums=[COUNTS['통과'],COUNTS['실패'],COUNTS['미확인'],COUNTS['해당없음'],0]
                for cell,value in zip(cells[1:6],nums): box_text(c,str(value),cell,10)
            feedback=p.find_tables()[2]
            box_text(c,'1인 프로젝트 개인 제출 범위: 별도 상호 점검 양식 제외',feedback.rows[1].cells[1],8)
            box_text(c,'작성자·셀프 점검자: 김지은 · 서울 08반. 실제 상호 점검 결과를 기재한 것은 아님.',feedback.rows[2].cells[1],8)
        c.save();buf.seek(0)
        # Clone each source for independent self/mutual overlays.
        source=PdfReader(SOURCE).pages[page_i]
        source.merge_page(PdfReader(buf).pages[0]);writer.add_page(source)
    if not mutual:
        appendix=BytesIO();c=canvas.Canvas(appendix,pagesize=(W,H))
        for idx,(filename,heading,caption) in enumerate(SHOTS,1):
            c.setFillColor(INK);c.setFont('Korean',7.5)
            c.drawString(34,H-26,'[SSAFY] AI 코어 프로젝트 2회차 - 셀프·상호 체크리스트 / 화면 증빙')
            c.setFont('KoreanBold',17);c.drawString(40,H-65,f'부록 {idx}. {heading}')
            c.setFillColor(GREEN);c.rect(34,H-104,774,21,fill=1,stroke=0)
            box_text(c,caption,(38,85,804,103),8)
            path=ROOT/'screenshots'/filename
            iw,ih=Image.open(path).size;scale=min(774/iw,385/ih)
            c.drawImage(str(path),34+(774-iw*scale)/2,88+(385-ih*scale)/2,width=iw*scale,height=ih*scale)
            c.setFillColor(INK);c.setFont('Korean',7.5);c.drawString(34,58,filename)
            c.drawString(34,23,'새 UI 캡처와 기존 기능 실행 증빙을 구분합니다.');c.drawRightString(808,23,f'{idx+4} 쪽');c.showPage()
        # Code evidence for the exact graph built by the app.
        c.setFillColor(INK);c.setFont('KoreanBold',17);c.drawString(40,H-65,'부록 10. LangGraph 두 노드 구성')
        c.setFillColor(GREEN);c.rect(34,H-104,774,21,fill=1,stroke=0)
        box_text(c,'항목 7 · src/rag_graph.py의 실제 그래프 구성 코드. 모델 호출 없이 소스를 확인함.',(38,85,804,103),8)
        source=(ROOT/'src/rag_graph.py').read_text(encoding='utf-8')
        code=source.split('def build_rag_graph():',1)[1].split('\ndef run_rag',1)[0]
        c.setFillColor(GRAY);c.rect(70,220,700,230,fill=1,stroke=0)
        c.setFillColor(INK);c.setFont('Korean',12)
        y=425
        for line in ('def build_rag_graph():'+code).splitlines():c.drawString(85,y,line);y-=19
        c.setFont('Korean',10);c.drawString(85,170,'START → retrieve → generate → END')
        c.setFont('Korean',7.5);c.drawString(34,23,'그래프 구성 확인은 생성 답변 정확도 평가와 별개입니다.');c.drawRightString(808,23,'14 쪽');c.showPage();c.save();appendix.seek(0)
        for page in PdfReader(appendix).pages:writer.add_page(page)
    target=OUT/'체크리스트.pdf'
    writer.add_metadata({'/Title':'AqSolDB 상호 점검용 양식 (미실시)' if mutual else '김지은 서울 08반 - AqSolDB 셀프 체크리스트 및 증빙'})
    with target.open('wb') as out:writer.write(out)
    check=PdfReader(target)
    assert len(check.pages)==(4 if mutual else 14)
    assert not check.get_fields()
    print(target.name,len(check.pages),'pages')
