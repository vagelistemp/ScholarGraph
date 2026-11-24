// Markdown (a small dialect) -> PDF through headless Chrome.
//   node docs/build/md2pdf.js input.md output.pdf "Footer text"
// Dialect:   % Title | Subtitle | meta line          first line: title block
//            # H1   ## H2   ### H3                    headings
//            - item / 1. item                         lists (consecutive lines)
//            ```  ...  ```                            code block
//            :::note Title ... :::                    callout box; its lines starting with "- " are the bullets
//            | a | b |  with a |---| separator row     table (column widths are proportional to the longest cell)
//            ![caption](file.png|width_px)            figure from docs/figures
// Inline:    `code`, **bold**, *italic*
const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');
const { H, renderDoc, setImgDir } = require('./html_backend.js');

const [, , input, output, footer = ''] = process.argv;
const FIG_DIR = path.join(__dirname, '..', 'figures');
setImgDir(FIG_DIR);
const lines = fs.readFileSync(input, 'utf8').replace(/\r/g, '').split('\n');
const blocks = [];
let i = 0;
const isTableLine = (l) => /^\s*\|.*\|\s*$/.test(l);

function flushPara(buf) { if (buf.length) { blocks.push(H.body(buf.join(' '))); buf.length = 0; } }
const para = [];
while (i < lines.length) {
  const l = lines[i];
  if (i === 0 && l.startsWith('% ')) {
    const [t1, t2 = '', t3 = ''] = l.slice(2).split('|').map((s) => s.trim());
    blocks.push(`<div class="title"><div class="t1">${t1}</div><div class="t2">${t2}</div><div class="t3">${t3}</div></div>`);
    i++; continue;
  }
  if (l.startsWith('```')) {
    flushPara(para);
    const code = []; i++;
    while (i < lines.length && !lines[i].startsWith('```')) { code.push(lines[i]); i++; }
    i++; blocks.push(...H.code(code)); continue;
  }
  if (l.startsWith(':::note')) {
    flushPara(para);
    const title = l.slice(7).trim(); const items = []; i++;
    let open = false;                                   // is the last item still being continued by following lines?
    while (i < lines.length && !lines[i].startsWith(':::')) {
      const t = lines[i];
      if (t.startsWith('- ')) { items.push(t.slice(2)); open = true; }
      else if (!t.trim()) open = false;                 // a blank line ends an item
      else if (open) items[items.length - 1] += ' ' + t.trim();
      else { items.push(t.trim()); open = true; }       // a plain paragraph inside the box becomes its own item
      i++;
    }
    i++; blocks.push(H.callout(title, items)); continue;
  }
  if (isTableLine(l) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
    flushPara(para);
    const split = (s) => s.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
    const header = split(l); i += 2; const rows = [];
    while (i < lines.length && isTableLine(lines[i])) { rows.push(split(lines[i])); i++; }
    const widths = header.map((h, c) => Math.max(6, Math.min(60, Math.max(h.length, ...rows.map((r) => (r[c] || '').length)))));
    blocks.push(H.table(widths, header, rows, { size: 16 })); blocks.push(H.gap()); continue;
  }
  const fig = l.match(/^!\[(.*)\]\((.+?)\|(\d+)\)\s*$/);
  if (fig) { flushPara(para); blocks.push(H.png(fig[2], +fig[3])); blocks.push(H.caption(fig[1])); i++; continue; }
  if (/^### /.test(l)) { flushPara(para); blocks.push(H.h2(l.slice(4))); i++; continue; }
  if (/^## /.test(l)) { flushPara(para); blocks.push(H.h2(l.slice(3))); i++; continue; }
  if (/^# /.test(l)) { flushPara(para); blocks.push(H.h1(l.slice(2))); i++; continue; }
  if (/^- /.test(l)) { flushPara(para); blocks.push(H.bullet(l.slice(2))); i++; continue; }
  if (/^\d+\. /.test(l)) { flushPara(para); blocks.push(H.num(l.replace(/^\d+\. /, ''))); i++; continue; }
  if (!l.trim()) { flushPara(para); i++; continue; }
  para.push(l.trim()); i++;
}
flushPara(para);

const html = renderDoc(blocks.flat(), { title: footer || path.basename(input), footer: footer || path.basename(input), imgDir: FIG_DIR });
const htmlPath = output.replace(/\.pdf$/i, '.tmp.html');
fs.writeFileSync(htmlPath, html, 'utf8');
const chrome = process.env.CHROME || (process.platform === 'win32' ? 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe' : process.platform === 'darwin' ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
const r = spawnSync(chrome, ['--headless=new', '--disable-gpu', '--no-pdf-header-footer', `--print-to-pdf=${path.resolve(output)}`, 'file:///' + path.resolve(htmlPath).replace(/\\/g, '/')], { encoding: 'utf8' });
fs.unlinkSync(htmlPath);
if (r.status !== 0) { console.error(r.stderr); process.exit(1); }
console.log('wrote', output);
