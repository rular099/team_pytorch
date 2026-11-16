"""Print local HTML via an already running Firefox WebDriver BiDi endpoint.

Browser must be launched separately with a temporary profile. No external URL is
opened. This produces a browser PDF, not a successful LaTeX compile.
"""
import base64
import json
import websocket
from common import PAPER,write_json,sha

ws=websocket.create_connection('ws://127.0.0.1:9237/session',timeout=90,suppress_origin=True)
counter=0
log=[]
def call(method,params):
    global counter
    counter+=1;ws.send(json.dumps(dict(id=counter,method=method,params=params)))
    while True:
        result=json.loads(ws.recv())
        if result.get('id')!=counter:continue
        if result.get('type')=='error':raise RuntimeError(result)
        log.append(dict(method=method,status='success'))
        return result['result']

call('session.new',{'capabilities':{}})
context=call('browsingContext.getTree',{})['contexts'][0]['context']
call('browsingContext.setViewport',dict(context=context,viewport=dict(width=1440,height=1800),devicePixelRatio=1))
for filename,dest in [('preview.html','paper_review.pdf'),('preview_supplement.html','supplement_review.pdf')]:
    call('browsingContext.navigate',dict(context=context,url=(PAPER/filename).as_uri(),wait='complete'))
    call('script.evaluate',dict(expression='document.fonts.ready.then(()=>true)',target=dict(context=context),awaitPromise=True))
    shot=call('browsingContext.captureScreenshot',dict(context=context,origin='viewport'))
    (PAPER/'qa'/filename.replace('.html','.png')).write_bytes(base64.b64decode(shot['data']))
    pdf=call('browsingContext.print',dict(context=context,background=True,orientation='portrait',
        page=dict(width=21.0,height=29.7),margin=dict(top=1.8,bottom=1.8,left=1.8,right=1.8),
        scale=1.0,shrinkToFit=True))
    output=PAPER/dest;output.write_bytes(base64.b64decode(pdf['data']))
    log.append(dict(output=dest,bytes=output.stat().st_size,sha256=sha(output.read_bytes())))
call('session.end',{});ws.close()
build=json.loads((PAPER/'qa/build_checks.json').read_text())
build.update(browser_pdf='GENERATED using Firefox local HTML print; visual inspection separate',
    latex_pdf='NOT_BUILT; no TeX engine installed',browser='Firefox 160.0')
write_json(PAPER/'qa/build_checks.json',build)
write_json(PAPER/'qa/browser_render.json',dict(status='PASS',local_only=True,log=log))
print('PASS: local browser paper/supplement PDFs and viewport screenshots; no LaTeX compile claimed')
