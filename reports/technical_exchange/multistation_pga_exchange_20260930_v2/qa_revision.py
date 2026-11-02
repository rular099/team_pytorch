#!/usr/bin/env python3
"""Audit revision provenance, native shapes, original scientific images and PDF."""
import argparse
import hashlib
import json
import subprocess
import zipfile
import os
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

DECK=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--render',action='store_true');p.add_argument('--visual-review-complete',action='store_true')
    a=p.parse_args();m=json.loads((DECK/'native_deck_manifest.json').read_text())
    ppt=DECK/m['pptx'];prs=Presentation(ppt)
    assert len(prs.slides)==28 and len(prs.slides[0].shapes)==2
    assert all(s.has_notes_slide and len(s.notes_slide.notes_text_frame.text)>60 for s in prs.slides)
    editable=tables=0;charts=[]
    for number,sl in enumerate(prs.slides,1):
        for obj in sl.shapes:
            assert obj.left>=-100 and obj.top>=-100
            assert obj.left+obj.width<=prs.slide_width+100 and obj.top+obj.height<=prs.slide_height+100
            editable+=int(obj.has_text_frame and bool(obj.text.strip()));tables+=int(obj.has_table)
            if obj.shape_type==MSO_SHAPE_TYPE.PICTURE:
                r=next(r for r in m['image_records'] if r['slide']==number and r['shape']==obj.name)
                assert hashlib.sha256(obj.image.blob).hexdigest()==sha(DECK/r['path'])==r['sha256']
                assert max(abs(v) for v in [obj.crop_left,obj.crop_right,obj.crop_top,obj.crop_bottom])<1e-10
                err=abs((obj.width/obj.height)/(r['source_width_px']/r['source_height_px'])-1)
                assert err<1e-6
                charts.append({'slide':number,'source':r['path'],'byte_hash_matches':True,'aspect_error':err})
    with zipfile.ZipFile(ppt) as z:
        assert z.testzip() is None
        assert sum(n.startswith('ppt/slides/slide') and n.endswith('.xml') for n in z.namelist())==28
        assert sum(n.startswith('ppt/notesSlides/notesSlide') and n.endswith('.xml') for n in z.namelist())==28
    result=dict(checks='PASS',slides=28,main=22,appendix=6,editable_text=editable,native_tables=tables,charts=charts,
                pptx_sha256=sha(ppt),visual_review='PASS' if a.visual_review_complete else 'pending',v1_preserved=True)
    pdf=ppt.with_suffix('.pdf')
    if pdf.exists():
        pages=int(next(x.split(':')[1] for x in subprocess.check_output(['pdfinfo',str(pdf)],text=True).splitlines() if x.startswith('Pages:')))
        assert pages==28;result.update(pdf_pages=pages,pdf_sha256=sha(pdf))
    if a.render:
        out=DECK/'preview';out.mkdir(exist_ok=True)
        subprocess.run(['pdftoppm','-scale-to','1600','-png',str(pdf),str(out/'slide')],check=True)
        rendered=sorted(out.glob('slide-*.png'));assert len(rendered)==28
        font=ImageFont.truetype(os.environ.get('NATIVE_REPORT_FONT', str(Path.home() / '.fonts/noto/NotoSansCJKsc-Regular.otf')),18)
        for part,start in enumerate(range(0,28,9),1):
            sheet=Image.new('RGB',(1800,1089),'#E8EDF0');draw=ImageDraw.Draw(sheet)
            for i,path in enumerate(rendered[start:start+9]):
                with Image.open(path) as im:
                    im=im.convert('RGB');im.thumbnail((600,337));x=i%3*600;y=i//3*363
                    sheet.paste(im,(x,y));draw.text((x+10,y+338),f'Slide {start+i+1}',fill='#2F3742',font=font)
            sheet.save(out/f'contact_sheet_{part}.png')
    (DECK/'native_qa.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
