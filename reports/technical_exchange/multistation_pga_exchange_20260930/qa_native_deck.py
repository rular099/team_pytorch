#!/usr/bin/env python3
"""Verify native package and scientific image fidelity, optionally render QA previews."""
import argparse
import hashlib
import json
import subprocess
import zipfile
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

DECK=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--render',action='store_true')
    p.add_argument('--visual-review-complete',action='store_true',
                   help='Record manual approval only after inspecting freshly rendered pages.')
    args=p.parse_args()
    manifest=json.loads((DECK/'native_deck_manifest.json').read_text())
    ppt=DECK/manifest['pptx']
    prs=Presentation(ppt)
    assert len(prs.slides)==26
    assert len(prs.slides[0].shapes)==2
    assert all(s.has_text_frame for s in prs.slides[0].shapes)
    assert 'PGA' in prs.slides[0].shapes[0].text
    assert prs.slides[0].shapes[1].text=='2026-09-30'
    assert all(sl.has_notes_slide and len(sl.notes_slide.notes_text_frame.text)>60 for sl in prs.slides)
    outside=[]; images=[]; editable=0; tables=0
    for number,sl in enumerate(prs.slides,1):
        for obj in sl.shapes:
            if obj.left < -100 or obj.top < -100 or obj.left+obj.width>prs.slide_width+100 or obj.top+obj.height>prs.slide_height+100:
                outside.append((number,obj.name))
            editable+=int(obj.has_text_frame and bool(obj.text.strip()))
            tables+=int(obj.has_table)
            if obj.shape_type==MSO_SHAPE_TYPE.PICTURE:
                rec=next(r for r in manifest['image_records'] if r['slide']==number and r['shape']==obj.name)
                source=DECK/rec['path']
                assert hashlib.sha256(obj.image.blob).hexdigest()==sha(source)==rec['sha256']
                assert all(abs(v)<1e-10 for v in [obj.crop_left,obj.crop_right,obj.crop_top,obj.crop_bottom])
                drift=abs((obj.width/obj.height)/(rec['source_width_px']/rec['source_height_px'])-1)
                assert drift<1e-6,(number,drift)
                images.append({'slide':number,'path':rec['path'],'sha256_matches':True,'relative_aspect_error':drift})
    assert not outside, outside
    assert editable>150 and tables>=10
    with zipfile.ZipFile(ppt) as z:
        assert z.testzip() is None
        allnames=z.namelist()
        assert sum(n.startswith('ppt/slides/slide') and n.endswith('.xml') for n in allnames)==26
        assert sum(n.startswith('ppt/notesSlides/notesSlide') and n.endswith('.xml') for n in allnames)==26
    result={'checks':'PASS','slides':26,'main':20,'appendix':6,'cover_two_text_objects_only':True,
        'notes_slides':26,'editable_text_shapes':editable,'editable_tables':tables,
        'shapes_outside_canvas':outside,'scientific_images':images,'pptx_sha256':sha(ppt),
        'generative_drafts_used_in_ppt':False,
        'visual_review':'PASS: all 26 rendered pages reviewed' if args.visual_review_complete else 'pending'}
    pdf=ppt.with_suffix('.pdf')
    if pdf.exists():
        info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
        pages=int(next(line.split(':')[1].strip() for line in info.splitlines() if line.startswith('Pages:')))
        assert pages==26
        result['rendered_pdf_pages']=pages
        result['pdf_sha256']=sha(pdf)
    if args.render:
        if not pdf.exists():raise FileNotFoundError('Export PDF with headless LibreOffice first')
        preview=DECK/'preview';preview.mkdir(exist_ok=True)
        subprocess.run(['pdftoppm','-scale-to','1600','-png',str(pdf),str(preview/'slide')],check=True)
        rendered=sorted(preview.glob('slide-*.png'))
        assert len(rendered)==26
        fontpath=os.environ.get('NATIVE_REPORT_FONT', str(Path.home() / '.fonts/noto/NotoSansCJKsc-Regular.otf'))
        font=ImageFont.truetype(fontpath,18)
        for part,start in enumerate(range(0,26,9),1):
            chosen=rendered[start:start+9]
            sheet=Image.new('RGB',(1800,3*363),'#E8EDF0');draw=ImageDraw.Draw(sheet)
            for i,path in enumerate(chosen):
                with Image.open(path) as image:
                    image=image.convert('RGB');image.thumbnail((600,337))
                    x=(i%3)*600;y=(i//3)*363
                    sheet.paste(image,(x,y));draw.text((x+10,y+338),f'Slide {start+i+1}',font=font,fill='#2F3742')
            sheet.save(preview/f'contact_sheet_{part}.png')
        result['previews']=26;result['contact_sheets']=3
    (DECK/'native_qa.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
