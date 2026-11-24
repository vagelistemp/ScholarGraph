// HTML backend for the recap PDF (rendered to PDF with headless Chrome).
const fs = require('fs');
const path = require('path');

const HERE = __dirname;
let IMG_DIR = path.join(HERE, 'img');
const IMG = () => 'file:///' + IMG_DIR.replace(/\\/g, '/');
const CONTENT_PX = 642; // A4 minus 2 cm margins at 96 dpi

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
function inline(s) {
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\^[^^]+\^|`[^`]+`)/g;
  let out = '', last = 0, m;
  while ((m = re.exec(s))) {
    out += esc(s.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith('**')) out += '<b>' + esc(tok.slice(2, -2)) + '</b>';
    else if (tok.startsWith('*')) out += '<i>' + esc(tok.slice(1, -1)) + '</i>';
    else if (tok.startsWith('`')) out += '<code>' + esc(tok.slice(1, -1)) + '</code>';
    else out += '<sup>' + esc(tok.slice(1, -1)) + '</sup>';
    last = m.index + tok.length;
  }
  return out + esc(s.slice(last));
}

const H = {
  body: (s) => `<p>${inline(s)}</p>`,
  h1: (s) => `<h1>${inline(s)}</h1>`,
  h2: (s) => `<h2>${inline(s)}</h2>`,
  bullet: (s) => `<ul class="b"><li>${inline(s)}</li></ul>`,
  num: (s) => `<ol class="b"><li>${inline(s)}</li></ol>`,
  code: (lines) => [`<pre>${lines.map(esc).join('\n')}</pre>`],
  caption: (s) => `<div class="cap">${inline(s)}</div>`,
  gap: () => '<div class="gap"></div>',
  png: (file, wPx) => `<div class="fig"><img src="${IMG()}/${file}" style="width:${Math.min(100, (100 * wPx) / CONTENT_PX).toFixed(1)}%"></div>`,
  table: (cols, header, rows, o = {}) => {
    const sum = cols.reduce((a, b) => a + b, 0);
    const col = cols.map((c) => `<col style="width:${((100 * c) / sum).toFixed(2)}%">`).join('');
    const cell = (c) => String(c).split('\n').map(inline).join('<br>');
    const cls = o.size && o.size <= 16 ? 'xs' : o.size && o.size <= 17 ? 's' : '';
    return `<table class="${cls}"><colgroup>${col}</colgroup><thead><tr>${header.map((h) => `<th>${cell(h)}</th>`).join('')}</tr></thead>` +
      `<tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${cell(c)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
  },
  callout: (title, lines) => `<div class="callout"><div class="ct">${inline(title)}</div><ul>${lines.map((l) => `<li>${inline(l)}</li>`).join('')}</ul></div>`,
  titleBlock: () => '',
};

const CSS_TEMPLATE = `
@page { size: A4; margin: 20mm 20mm 20mm 20mm;
  @bottom-center { content: "__FOOTER__ — page " counter(page) " / " counter(pages); font: 8pt Cambria, serif; color: #808080; } }
html { font-family: Cambria, "Times New Roman", serif; font-size: 10.5pt; color: #1F2933; }
body { margin: 0; line-height: 1.42; }
p { margin: 0 0 6pt 0; text-align: justify; hyphens: auto; }
h1 { font-size: 14pt; color: #2B6CB0; margin: 16pt 0 6pt; padding-bottom: 2pt; border-bottom: 1pt solid #9DB3C8; break-after: avoid; }
h2 { font-size: 11.5pt; margin: 11pt 0 4pt; break-after: avoid; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #F1F4F8; padding: 0 2pt; border-radius: 2pt; }
pre { font-family: Consolas, monospace; font-size: 8.6pt; background: #F3F5F8; border-left: 3pt solid #9DB3C8; padding: 5pt 8pt; margin: 4pt 0 8pt; white-space: pre-wrap; line-height: 1.3; }
ul.b, ol.b { margin: 0 0 3pt 0; padding-left: 16pt; text-align: justify; }
ul.b li, ol.b li { margin: 0; }
.cap { text-align: center; font-size: 9pt; font-style: italic; color: #4A5568; margin: 2pt 0 9pt; }
.fig { text-align: center; margin: 6pt 0 0; break-inside: avoid; }
.fig img { max-width: 100%; }
.gap { height: 5pt; }
.title { border-bottom: 2pt solid #2B6CB0; padding-bottom: 7pt; margin-bottom: 10pt; }
.t1 { font-size: 20pt; font-weight: bold; }
.t2 { font-size: 13pt; font-style: italic; color: #4A5568; margin: 2pt 0 4pt; }
.t3 { font-size: 9.5pt; color: #4A5568; }
.callout { background: #F4F8FC; border: 0.5pt solid #BFC8D2; border-left: 4pt solid #2B6CB0; padding: 7pt 10pt 4pt; margin: 4pt 0 8pt; break-inside: avoid; }
.callout .ct { font-weight: bold; color: #2B6CB0; font-size: 11.5pt; margin-bottom: 3pt; }
.callout ul { margin: 0; padding-left: 14pt; }
.callout li { margin: 0 0 3pt; text-align: justify; }
table { width: 100%; border-collapse: collapse; margin: 3pt 0 6pt; font-size: 9pt; }
table.s { font-size: 8.6pt; } table.xs { font-size: 8.2pt; }
th { background: #E4ECF4; text-align: left; }
th, td { border: 0.5pt solid #BFC8D2; padding: 3pt 4.5pt; vertical-align: middle; }
tr { break-inside: avoid; }
thead { display: table-header-group; }
`;


function renderDoc(blocks, { title, footer, imgDir }) {
  if (imgDir) IMG_DIR = imgDir;
  const css = CSS_TEMPLATE.replace('__FOOTER__', footer);
  return `<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>${title}</title><style>${css}</style></head><body>${blocks.join(String.fromCharCode(10))}</body></html>`;
}
function setImgDir(d) { IMG_DIR = d; }
module.exports = { H, renderDoc, inline, setImgDir };
