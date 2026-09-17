"""Build a self-contained HTML viewer for judging LiDAR<->camera alignment by eye.

  python scripts/calib_overlay_viewer.py --frames 0 1600 500 2000

The page embeds, per frame, the ERP, the pre-rotated dual fisheye and the LiDAR xyz.
Projection runs in the browser, so opacity, point size, range window, the extrinsic
(p_cam = Rz(yaw) Ry(pitch) Rx(roll) p + t) and the fisheye lens parameters are all live.
Sliders start at the values in the config. Paste the JSON box back to record a new setting.
"""
import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np

from bevloc import config as C
from bevloc.run import context, frame_names


def jpg_b64(rgb, q=90):
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode()


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    ap.add_argument("--frames", nargs="+", default=["0", "1600", "500", "2000"])
    ap.add_argument("--out", default="experiments/00_calib/overlay_viewer.html")
    ap.add_argument("--max-points", type=int, default=130000)
    ap.add_argument("--max-range", type=float, default=80.0)
    a = ap.parse_args()
    cfg = C.load(a.config)
    ds, _, _ = context(cfg, need_mask=False)
    ds.erp_size = (2048, 1024)
    rng, frames = np.random.default_rng(0), []
    for n in frame_names(ds, a.frames):
        p = ds.points(n, 1.0)
        p = p[np.linalg.norm(p, axis=1) < a.max_range]
        if len(p) > a.max_points:
            p = p[rng.choice(len(p), a.max_points, replace=False)]
        p = p[np.argsort(np.linalg.norm(p, axis=1))]          # sorted by range -> colour batching in JS
        frames.append(dict(name=n, n=int(len(p)), xyz=base64.b64encode(p.astype("<f4").tobytes()).decode(),
                           erp=jpg_b64(ds.erp(n)), fish=jpg_b64(ds.fisheye(n), 95)))
        print(n, len(p), "points")

    c, lp = cfg.calib, cfg.erp.lens_params
    init = dict(roll=c.cam_from_lidar_rpy_deg[0], pitch=c.cam_from_lidar_rpy_deg[1], yaw=c.cam_from_lidar_rpy_deg[2],
                tx=c.cam_from_lidar_t_m[0], ty=c.cam_from_lidar_t_m[1], tz=c.cam_from_lidar_t_m[2],
                fin=lp.fov_inner_deg, fout=lp.fov_outer_deg, knee=lp.alpha)
    html = TEMPLATE.replace("/*DATA*/", json.dumps(frames)).replace("/*INIT*/", json.dumps(init))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB  ->  xdg-open {out}")


TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><title>LiDAR-camera overlay</title>
<style>
 html,body{margin:0;height:100%;background:#111;color:#ddd;font:13px system-ui,sans-serif;overflow:hidden}
 #side{position:fixed;left:0;top:0;bottom:0;width:290px;padding:10px;box-sizing:border-box;background:#1b1b1b;overflow-y:auto}
 #cv{position:fixed;left:290px;top:0}
 label{display:flex;justify-content:space-between;margin-top:7px} label span{color:#8cf;font-variant-numeric:tabular-nums}
 input[type=range]{width:100%} h3{margin:14px 0 2px;font-size:12px;color:#999;text-transform:uppercase;letter-spacing:.05em}
 select,button{background:#2a2a2a;color:#ddd;border:1px solid #444;padding:4px 6px;border-radius:4px;margin:2px 2px 2px 0}
 textarea{width:100%;height:96px;background:#0c0c0c;color:#9f9;border:1px solid #333;font:11px monospace;box-sizing:border-box}
 .hint{color:#888;font-size:11.5px;line-height:1.45;margin-top:8px}
</style></head><body>
<div id="side">
 <h3>Frame / view</h3>
 <select id="frame"></select>
 <select id="view">
  <option value="fish">Dual fisheye + lens model (live)</option>
  <option value="erp">ERP (config lens) + ideal sphere</option>
 </select>
 <h3>Points</h3>
 <label>opacity <span id="o_alpha"></span></label><input id="alpha" type="range" min="0" max="1" step="0.01" value="0.55">
 <label>size px <span id="o_size"></span></label><input id="size" type="range" min="0.5" max="5" step="0.25" value="1.5">
 <label>min range m <span id="o_rmin"></span></label><input id="rmin" type="range" min="1" max="80" step="0.5" value="1">
 <label>max range m <span id="o_rmax"></span></label><input id="rmax" type="range" min="2" max="80" step="0.5" value="40">
 <label>colour span m <span id="o_cspan"></span></label><input id="cspan" type="range" min="5" max="80" step="1" value="30">
 <h3>Extrinsic: p_cam = R·p_lidar + t</h3>
 <label>roll ° <span id="o_roll"></span></label><input id="roll" type="range" min="-5" max="5" step="0.05">
 <label>pitch ° <span id="o_pitch"></span></label><input id="pitch" type="range" min="-5" max="5" step="0.05">
 <label>yaw ° <span id="o_yaw"></span></label><input id="yaw" type="range" min="-5" max="5" step="0.05">
 <label>tx fwd m <span id="o_tx"></span></label><input id="tx" type="range" min="-1" max="1" step="0.01">
 <label>ty left m <span id="o_ty"></span></label><input id="ty" type="range" min="-1" max="1" step="0.01">
 <label>tz up m <span id="o_tz"></span></label><input id="tz" type="range" min="-1" max="1" step="0.01">
 <h3>Fisheye lens (fisheye view only)</h3>
 <label>inner FOV ° <span id="o_fin"></span></label><input id="fin" type="range" min="185" max="215" step="0.25">
 <label>outer FOV ° <span id="o_fout"></span></label><input id="fout" type="range" min="190" max="215" step="0.25">
 <label>knee alpha rad <span id="o_knee"></span></label><input id="knee" type="range" min="0.3" max="1.7" step="0.01">
 <div><button id="p_cfg">config values</button><button id="p_eq">equidistant 203</button><button id="p_zero">zero extrinsic</button></div>
 <h3>Current parameters</h3><textarea id="out" readonly></textarea>
 <div class="hint">wheel = zoom · drag = pan · <b>space</b> = blink points · <b>f</b> = next frame · <b>v</b> = switch view · <b>0</b> = fit.<br>
 Colour = range (blue near → red far). The ERP image is pre-rendered with the config lens, so lens sliders apply to the fisheye view only.</div>
</div>
<canvas id="cv"></canvas>
<script>
const F=/*DATA*/, INIT=/*INIT*/;
const $=id=>document.getElementById(id), cv=$('cv'), ctx=cv.getContext('2d');
const ids=['alpha','size','rmin','rmax','cspan','roll','pitch','yaw','tx','ty','tz','fin','fout','knee'];
function setInit(){Object.keys(INIT).forEach(k=>$(k).value=INIT[k]);}
setInit();
let cur=0, show=true, sc=1, ox=0, oy=0, imgs={}, pts=[];
function b64f32(s){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Float32Array(u.buffer);}
F.forEach((f,i)=>{pts[i]=b64f32(f.xyz);const o=document.createElement('option');o.value=i;o.textContent=f.name+'  ('+f.n+' pts)';$('frame').appendChild(o);
  ['erp','fish'].forEach(k=>{const im=new Image();im.onload=()=>{if(i==0)fit();};im.src='data:image/jpeg;base64,'+f[k];imgs[i+k]=im;});});
function turbo(t){t=Math.min(1,Math.max(0,t));const r=Math.round(255*Math.min(1,Math.max(0,1.5-Math.abs(4*t-3)))),g=Math.round(255*Math.min(1,Math.max(0,1.5-Math.abs(4*t-2)))),b=Math.round(255*Math.min(1,Math.max(0,1.5-Math.abs(4*t-1))));return `rgb(${r},${g},${b})`;}
function P(){const o={};ids.forEach(k=>o[k]=parseFloat($(k).value));return o;}
function resize(){cv.width=innerWidth-290;cv.height=innerHeight;draw();}
function fit(){const im=imgs[cur+$('view').value];if(!im||!im.width)return;sc=Math.min(cv.width/im.width,cv.height/im.height);ox=(cv.width-im.width*sc)/2;oy=(cv.height-im.height*sc)/2;draw();}
function draw(){
  const p=P(),view=$('view').value,im=imgs[cur+view];ids.forEach(k=>$('o_'+k).textContent=p[k]);
  $('out').value=JSON.stringify({frame:F[cur].name,cam_from_lidar_rpy_deg:[p.roll,p.pitch,p.yaw],cam_from_lidar_t_m:[p.tx,p.ty,p.tz],fov_inner_deg:p.fin,fov_outer_deg:p.fout,alpha:p.knee},null,1);
  ctx.setTransform(1,0,0,1,0,0);ctx.fillStyle='#111';ctx.fillRect(0,0,cv.width,cv.height);
  if(!im||!im.width)return;
  ctx.setTransform(sc,0,0,sc,ox,oy);ctx.imageSmoothingEnabled=sc<2;ctx.drawImage(im,0,0);
  if(!show)return;
  const W=im.width,H=im.height,d=Math.PI/180,cr=Math.cos(p.roll*d),sr=Math.sin(p.roll*d),cp=Math.cos(p.pitch*d),sp=Math.sin(p.pitch*d),cy=Math.cos(p.yaw*d),sy=Math.sin(p.yaw*d);
  const R=[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr, sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr, -sp,cp*sr,cp*cr];
  const a=2/(p.fin*d),half=p.fout*d/2,b=(1-a*p.knee)/(half-p.knee),dd=1-b*half;
  const rad=x=>{const s=1/(1+Math.exp(-20*(x-p.knee)));return (1-s)*a*x+s*(b*x+dd);};
  const X=pts[cur],n=X.length/3,s=p.size/sc*Math.max(1,sc*0.6),hs=s/2;ctx.globalAlpha=p.alpha;
  let bin=-1;
  for(let i=0;i<n;i++){
    const x0=X[3*i],y0=X[3*i+1],z0=X[3*i+2],rr=Math.hypot(x0,y0,z0);if(rr<p.rmin)continue;if(rr>p.rmax)break;
    const x=R[0]*x0+R[1]*y0+R[2]*z0+p.tx,y=R[3]*x0+R[4]*y0+R[5]*z0+p.ty,z=R[6]*x0+R[7]*y0+R[8]*z0+p.tz;
    let u,v;
    if(view=='erp'){u=(Math.atan2(-y,x)/(2*Math.PI)+0.5)*W;v=(0.5-Math.asin(z/Math.hypot(x,y,z))/Math.PI)*H;}
    else{const Yu=z,Zr=-y,th=Math.atan2(Yu,Zr),phi=Math.atan(Math.hypot(Yu,Zr)/(x+1e-6)),front=x>0,r=rad(Math.abs(phi));
      let xx=(front?r:-r)*Math.cos(th);const yy=r*Math.sin(th);xx=front?(xx+1)/2:(xx-1)/2;u=(xx+1)/2*W;v=(-yy+1)/2*H;}
    const nb=Math.min(31,Math.floor((rr-p.rmin)/p.cspan*32));if(nb!=bin){bin=nb;ctx.fillStyle=turbo(nb/31);}
    ctx.fillRect(u-hs,v-hs,s,s);
  }
  ctx.globalAlpha=1;
}
ids.forEach(k=>$(k).addEventListener('input',draw));
$('frame').onchange=e=>{cur=+e.target.value;draw();};$('view').onchange=()=>fit();
$('p_cfg').onclick=()=>{setInit();draw();};
$('p_eq').onclick=()=>{$('fin').value=203;$('fout').value=203;$('knee').value=1.0;draw();};
$('p_zero').onclick=()=>{['roll','pitch','yaw','tx','ty','tz'].forEach(k=>$(k).value=0);draw();};
cv.addEventListener('wheel',e=>{e.preventDefault();const k=Math.exp(-e.deltaY*0.0015),mx=e.offsetX,my=e.offsetY;ox=mx-(mx-ox)*k;oy=my-(my-oy)*k;sc*=k;draw();},{passive:false});
let drag=null;cv.onmousedown=e=>drag=[e.clientX-ox,e.clientY-oy];onmouseup=()=>drag=null;onmousemove=e=>{if(drag){ox=e.clientX-drag[0];oy=e.clientY-drag[1];draw();}};
onkeydown=e=>{if(e.code=='Space'){show=!show;draw();e.preventDefault();}
  if(e.key=='f'){cur=(cur+1)%F.length;$('frame').value=cur;draw();}
  if(e.key=='v'){$('view').value=$('view').value=='erp'?'fish':'erp';fit();}
  if(e.key=='0')fit();};
onresize=resize;resize();
</script></body></html>
"""

if __name__ == "__main__":
    main()
