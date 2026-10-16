/*
 * QuADA-GS project page: interactive parts.
 * Plain JavaScript, no dependencies. Reads window.QUADA_DATA (data.js) and window.QuADA (core.js).
 */
(function () {
  'use strict';

  var D = window.QUADA_DATA, Q = window.QuADA;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var fmt = function (n) { return Math.round(n).toLocaleString('en-US'); };
  var lerp = function (a, b, t) { return a + (b - a) * t; };
  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var C = { l0: '#A9BBF3', l1: '#2434E0', l2: '#F0206B', ink: '#1B1C24', ink2: '#51546A', line: '#D3D7E8', gray: '#9EA3B8' };
  var LEVEL_NAME = ['x1', 'x2', 'x4'];
  var LEVEL_COLOR = [C.l0, C.l1, C.l2];

  function svgEl(tag, attrs, parent) {
    var e = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (var k in attrs) if (Object.prototype.hasOwnProperty.call(attrs, k)) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  /** Segmented control: one pressed button at a time. */
  function bindSeg(container, key, cb) {
    if (!container) return;
    container.addEventListener('click', function (ev) {
      var b = ev.target.closest('button');
      if (!b || b.dataset[key] === undefined) return;
      $$('button', container).forEach(function (o) { o.setAttribute('aria-pressed', o === b ? 'true' : 'false'); });
      cb(b.dataset[key]);
    });
  }

  /* ======================================================================
   * Hero demo: Structure Map -> quadtree -> Gaussians
   * ==================================================================== */
  function initDemo() {
    var FW = 64, FH = 64, PW = FW * 4, PH = FH * 4, SC = 3; // features, pointer-grid px, canvas scale
    var canvas = $('#demoCanvas'), ctx = canvas.getContext('2d');
    var chip = $('#demoChip');
    var src = document.createElement('canvas'); src.width = PW; src.height = PH;
    var sctx = src.getContext('2d', { willReadFrequently: true });

    var state = { theme: null, mode: 'quadtree', phi1: 0.1, phi2: 1.4, rgba: null, feat: null, levels: null, leaves: null, shownCount: 0 };
    var intro = null;

    /* ---------- the image: a hi-res copy for display, a 256x256 copy for the analysis ---------- */
    var photo = document.createElement('canvas'); photo.width = canvas.width; photo.height = canvas.height;
    var pctx = photo.getContext('2d');
    var THEMES = {
      light: { wash: '255,255,255', washQuad: 0.5, washMap: 0.45, stroke: ['rgba(81,84,106,0.7)', 'rgba(36,52,224,0.95)', 'rgba(240,32,107,0.95)'], fill: ['rgba(169,187,243,0.55)', 'rgba(36,52,224,0.86)', 'rgba(240,32,107,0.92)'] },
      dark:  { wash: '8,10,36', washQuad: 0.38, washMap: 0.55, stroke: ['rgba(169,187,243,0.3)', 'rgba(104,122,255,0.95)', 'rgba(255,64,128,0.95)'], fill: ['rgba(169,187,243,0.38)', 'rgba(60,84,255,0.78)', 'rgba(255,48,120,0.88)'] }
    };

    function analyse() {
      var id = sctx.getImageData(0, 0, PW, PH), d = id.data, sum = 0;
      for (var p = 0; p < d.length; p += 4) sum += 0.299 * d[p] + 0.587 * d[p + 1] + 0.114 * d[p + 2];
      state.theme = sum / (PW * PH * 255) < 0.42 ? THEMES.dark : THEMES.light;
      state.rgba = d;
      var score = Q.scoreMap(Q.toGray(d, PW, PH), PW, PH);
      state.feat = Q.poolScore(score, PW, PH, 4);
    }

    /** Cover-fit the image into the square demo, keep a sharp copy to show and a small copy to analyse. */
    function setImage(img) {
      var iw = img.naturalWidth || img.width, ih = img.naturalHeight || img.height, k = Math.max(1, Math.min(iw, ih) / PW);
      var side = Math.min(iw, ih), sx = (iw - side) / 2, sy = (ih - side) / 2;
      pctx.imageSmoothingQuality = 'high'; pctx.clearRect(0, 0, photo.width, photo.height);
      pctx.drawImage(img, sx, sy, side, side, 0, 0, photo.width, photo.height);
      sctx.setTransform(1, 0, 0, 1, 0, 0); sctx.imageSmoothingQuality = 'high'; sctx.clearRect(0, 0, PW, PH);
      // Two-step downscale keeps fine texture (scales, hair, grids) from aliasing into noise.
      var mid = document.createElement('canvas'); mid.width = mid.height = PW * 2;
      var mctx = mid.getContext('2d'); mctx.imageSmoothingQuality = 'high';
      mctx.drawImage(photo, 0, 0, mid.width, mid.height);
      sctx.drawImage(mid, 0, 0, PW, PH);
      analyse();
    }

    function loadUpload(file) {
      var url = URL.createObjectURL(file), img = new Image();
      img.onload = function () { setImage(img); URL.revokeObjectURL(url); render(true); };
      img.onerror = function () { URL.revokeObjectURL(url); };
      img.src = url;
    }

    /* ---------- drawing ---------- */
    function wash(alpha) { ctx.fillStyle = 'rgba(' + state.theme.wash + ',' + alpha + ')'; ctx.fillRect(0, 0, canvas.width, canvas.height); }

    function draw() {
      var W = canvas.width, H = canvas.height, i;
      ctx.imageSmoothingEnabled = true;
      if (state.mode === 'gaussians') {
        var out = Q.renderGaussians(state.rgba, PW, PH, state.leaves, SC, 0.5);
        ctx.putImageData(new ImageData(out.data, out.width, out.height), 0, 0);
        return;
      }
      ctx.drawImage(photo, 0, 0, W, H);
      if (state.mode === 'original') return;

      if (state.mode === 'structure') {
        wash(state.theme.washMap);
        var cw = W / FW, ch = H / FH, alphas = state.theme.fill;
        for (i = 0; i < state.levels.length; i++) {
          ctx.fillStyle = alphas[state.levels[i]];
          ctx.fillRect((i % FW) * cw, Math.floor(i / FW) * ch, cw + 0.5, ch + 0.5);
        }
        return;
      }

      // quadtree: one path per level so the canvas strokes three times, not tens of thousands
      wash(state.theme.washQuad);
      var L = state.leaves, strokes = state.theme.stroke;
      ctx.lineWidth = 1;
      for (var lv = 0; lv < 3; lv++) {
        ctx.beginPath();
        for (i = 0; i < L.count; i++) {
          if (L.level[i] !== lv) continue;
          var s = L.S[i] * SC;
          ctx.rect(L.x0[i] * SC + 0.5, L.y0[i] * SC + 0.5, s - 1, s - 1);
        }
        ctx.strokeStyle = strokes[lv]; ctx.stroke();
      }
    }

    function updateStats(immediate) {
      var c = state.leaves.counts, total = state.leaves.count, dense = Q.denseCount(FW, FH);
      var n = FW * FH;
      $('#statDense').textContent = fmt(dense);
      $('#statSaved').textContent = Math.round((1 - total / dense) * 100) + '%';
      $('#lv0').textContent = fmt(c.l0); $('#lv1').textContent = fmt(c.l1); $('#lv2').textContent = fmt(c.l2);
      var bars = $$('#levelBar i'), counts = [c.l0, c.l1, c.l2];
      bars.forEach(function (b, k) { b.style.flexGrow = String(counts[k] / n); });
      setCount(total, immediate);
    }

    var countRaf = 0;
    function setCount(target, immediate) {
      var el = $('#statCount');
      cancelAnimationFrame(countRaf);
      if (immediate || reduceMotion) { state.shownCount = target; el.textContent = fmt(target); return; }
      var from = state.shownCount, t0 = performance.now(), dur = 260;
      (function step(t) {
        var k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
        state.shownCount = lerp(from, target, e); el.textContent = fmt(state.shownCount);
        if (k < 1) countRaf = requestAnimationFrame(step);
      })(t0);
    }

    var PRESETS = { '0,0': 0, '0.1,1.4': 1, '2,2': 2 };
    function updatePresetUI() {
      var key = (+state.phi1).toFixed(2).replace(/\.?0+$/, '') + ',' + (+state.phi2).toFixed(2).replace(/\.?0+$/, '');
      key = key.replace(/^(-?\d+),/, '$1,');
      var norm = function (v) { return String(+(+v).toFixed(2)); };
      key = norm(state.phi1) + ',' + norm(state.phi2);
      $$('[data-preset]').forEach(function (b) { b.setAttribute('aria-pressed', b.dataset.preset === key ? 'true' : 'false'); });
      var out = $('#presetReadout');
      if (Object.prototype.hasOwnProperty.call(PRESETS, key)) {
        var r = D.ablation[0].rows[PRESETS[key]];
        out.innerHTML = 'Measured in the paper at ×4: <b>' + r.time + ' ms</b>, <b>' + fmt(+r.mem) + ' MB</b>, <b>' + r.psnr + ' dB</b> PSNR.';
      } else {
        out.textContent = 'Pick a preset to see what the paper measured for these settings.';
      }
    }

    function render(immediate) {
      if (!state.feat) return;
      state.levels = Q.quantize(state.feat, state.phi1, state.phi2);
      state.leaves = Q.buildLeaves(state.levels, FW, FH, 2);
      updateStats(immediate);
      updatePresetUI();
      draw();
    }

    function syncSliders() {
      $('#phi1').value = state.phi1; $('#phi2').value = state.phi2;
      $('#phi1Out').textContent = (+state.phi1).toFixed(2); $('#phi2Out').textContent = (+state.phi2).toFixed(2);
    }

    var pending = 0;
    function scheduleRender() { if (pending) return; pending = requestAnimationFrame(function () { pending = 0; render(); }); }

    function stopIntro() { if (intro) { cancelAnimationFrame(intro); intro = null; } }

    /* ---------- events ---------- */
    $('#phi1').addEventListener('input', function (e) {
      stopIntro(); state.phi1 = +e.target.value; if (state.phi1 > state.phi2) state.phi2 = state.phi1; syncSliders(); scheduleRender();
    });
    $('#phi2').addEventListener('input', function (e) {
      stopIntro(); state.phi2 = +e.target.value; if (state.phi2 < state.phi1) state.phi1 = state.phi2; syncSliders(); scheduleRender();
    });
    $$('[data-preset]').forEach(function (b) {
      b.addEventListener('click', function () {
        stopIntro(); var p = b.dataset.preset.split(',').map(Number);
        state.phi1 = p[0]; state.phi2 = p[1]; syncSliders(); render();
      });
    });
    bindSeg($('.demo-controls .ctl-row .seg'), 'mode', function (m) { state.mode = m; draw(); });
    $('#uploadBtn').addEventListener('click', function () { $('#demoFile').click(); });
    $('#demoFile').addEventListener('change', function (e) { var f = e.target.files && e.target.files[0]; if (f) { stopIntro(); loadUpload(f); } e.target.value = ''; });

    function hover(ev) {
      if (!state.levels) return;
      var r = canvas.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width, y = (ev.clientY - r.top) / r.height;
      if (x < 0 || y < 0 || x >= 1 || y >= 1) { chip.hidden = true; return; }
      var fx = Math.floor(x * FW), fy = Math.floor(y * FH), i = fy * FW + fx, lv = state.levels[i];
      var g = [1, 4, 16][lv];
      chip.innerHTML = '<i style="background:' + LEVEL_COLOR[lv] + '"></i>' + LEVEL_NAME[lv] + ' · ' + g + (g === 1 ? ' Gaussian' : ' Gaussians') + ' · score ' + state.feat[i].toFixed(2);
      chip.hidden = false;
    }
    canvas.addEventListener('pointermove', hover);
    canvas.addEventListener('pointerdown', hover);
    canvas.addEventListener('pointerleave', function (e) { if (e.pointerType === 'mouse') chip.hidden = true; });

    /* ---------- go ---------- */
    syncSliders();

    // One orchestrated moment: start dense, let the thresholds settle on the paper's defaults.
    function startIntro() {
      if (reduceMotion || window.__quadaNoIntro) return;
      var go = function () {
        var t0 = performance.now(), dur = 1700;
        state.phi1 = 0; state.phi2 = 0; syncSliders(); render(true);
        (function step(t) {
          var k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
          state.phi1 = Math.round(lerp(0, 0.1, e) * 100) / 100; state.phi2 = Math.round(lerp(0, 1.4, e) * 20) / 20;
          if (k >= 1) { state.phi1 = 0.1; state.phi2 = 1.4; }
          syncSliders(); render(k >= 1);
          intro = k < 1 ? requestAnimationFrame(step) : null;
        })(t0);
      };
      var ready = document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve();
      ready.then(function () { setTimeout(function () { if (!intro) intro = requestAnimationFrame(go); }, 350); });
    }

    var first = new Image();
    first.onload = function () { setImage(first); render(true); startIntro(); };
    first.src = window.QUADA_DEMO_IMAGE;
  }

  /* ======================================================================
   * Pointer Grid explorer
   * ==================================================================== */
  function initPointer() {
    var canvas = $('#pointerCanvas'); if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var ROOTS_W = 4, ROOTS_H = 4, CELL = 36, N = ROOTS_W * 4;           // 16x16 pointer grid, 1-cell margin
    var LEVELS = Uint8Array.from([0, 0, 1, 0,  0, 1, 2, 1,  0, 0, 1, 2,  1, 0, 0, 0]);
    var FILL = ['#E1E8FC', '#C9D2FA', '#FBC6D8'];
    var phase = 2, leaves, grid, cursor = { x: 8, y: 4 }, sel = -1;

    function build() { leaves = Q.buildLeaves(LEVELS, ROOTS_W, ROOTS_H, phase); grid = Q.pointerGrid(leaves); sel = grid.data[cursor.y * grid.pw + cursor.x]; }
    var px = function (v) { return (v + 1) * CELL; };

    function draw() {
      var W = canvas.width, m, i;
      ctx.clearRect(0, 0, W, W);
      ctx.fillStyle = '#F4F5FA'; ctx.fillRect(0, 0, W, W);
      ctx.fillStyle = '#fff'; ctx.fillRect(CELL, CELL, N * CELL, N * CELL);

      for (m = 0; m < leaves.count; m++) {
        var s = leaves.S[m] * CELL;
        ctx.fillStyle = FILL[leaves.level[m]]; ctx.fillRect(px(leaves.x0[m]), px(leaves.y0[m]), s, s);
      }
      // leaf outlines
      ctx.strokeStyle = 'rgba(81,84,106,0.55)'; ctx.lineWidth = 1;
      for (m = 0; m < leaves.count; m++) { var ss = leaves.S[m] * CELL; ctx.strokeRect(px(leaves.x0[m]) + 0.5, px(leaves.y0[m]) + 0.5, ss - 1, ss - 1); }
      // root boundaries
      ctx.strokeStyle = C.ink; ctx.lineWidth = 2;
      for (i = 0; i <= ROOTS_W; i++) { ctx.beginPath(); ctx.moveTo(px(i * 4), px(0)); ctx.lineTo(px(i * 4), px(N)); ctx.stroke(); ctx.beginPath(); ctx.moveTo(px(0), px(i * 4)); ctx.lineTo(px(N), px(i * 4)); ctx.stroke(); }
      // ids
      ctx.fillStyle = C.ink; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      for (m = 0; m < leaves.count; m++) {
        var sz = leaves.S[m]; ctx.font = '700 ' + (sz >= 2 ? 15 : 10) + 'px "Bricolage Grotesque", sans-serif';
        ctx.fillText(String(m), px(leaves.x0[m]) + sz * CELL / 2, px(leaves.y0[m]) + sz * CELL / 2 + 1);
      }

      if (sel < 0) return;
      var x0 = leaves.x0[sel], y0 = leaves.y0[sel], S = leaves.S[sel];
      var nb = Q.neighborIds(grid, x0, y0, S), coords = Q.lookupCoords(x0, y0, S);

      // neighbor leaves
      ctx.strokeStyle = C.l1; ctx.lineWidth = 3;
      var seen = {};
      nb.forEach(function (id, k) { if (id < 0 || id === sel || seen[id]) return; seen[id] = 1; var q = leaves.S[id] * CELL; ctx.strokeRect(px(leaves.x0[id]) + 1.5, px(leaves.y0[id]) + 1.5, q - 3, q - 3); });
      // selected leaf
      ctx.fillStyle = 'rgba(240,32,107,0.28)'; ctx.fillRect(px(x0), px(y0), S * CELL, S * CELL);
      ctx.strokeStyle = C.l2; ctx.lineWidth = 4; ctx.strokeRect(px(x0) + 2, px(y0) + 2, S * CELL - 4, S * CELL - 4);

      // the nine lookups
      coords.forEach(function (c, k) {
        var cx = px(c[0]) + CELL / 2, cy = px(c[1]) + CELL / 2, outside = nb[k] < 0;
        ctx.beginPath(); ctx.arc(cx, cy, 10, 0, Math.PI * 2);
        if (outside) { ctx.setLineDash([3, 3]); ctx.strokeStyle = C.ink2; ctx.lineWidth = 1.5; ctx.fillStyle = '#fff'; ctx.fill(); ctx.stroke(); ctx.setLineDash([]); }
        else { ctx.fillStyle = k === 4 ? C.l2 : C.l1; ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke(); }
        if (!outside) { ctx.fillStyle = '#fff'; ctx.font = '700 11px "Bricolage Grotesque", sans-serif'; ctx.fillText(String(k + 1), cx, cy + 0.5); }
      });
    }

    function panel() {
      var hint = $('#pointerHint'), facts = $('#pointerFacts'), gather = $('#gather');
      if (sel < 0) return;
      var x0 = leaves.x0[sel], y0 = leaves.y0[sel], S = leaves.S[sel], d = 2 - Math.log2(S);
      var o = Q.offsets(S);
      hint.textContent = 'Feature ' + sel + ' reads these nine neighbors from the Pointer Grid.';
      facts.innerHTML = '<b>Feature ' + sel + '</b> &middot; depth d = ' + d + ', side S = ' + S + ', top-left (x<sub>0</sub>, y<sub>0</sub>) = (' + x0 + ', ' + y0 + ')<br>' +
        'O(S) = {' + o.map(function (v) { return v < 0 ? '−' + (-v) : v; }).join(', ') + '}';
      var nb = Q.neighborIds(grid, x0, y0, S);
      gather.innerHTML = Array.prototype.map.call(nb, function (id, k) {
        if (id < 0) return '<div class="out" data-n="' + (k + 1) + '">outside<small>padding</small></div>';
        return '<div class="lv' + leaves.level[id] + (k === 4 ? ' self' : '') + '" data-n="' + (k + 1) + '">#' + id + '<small>S = ' + leaves.S[id] + (k === 4 ? ' (itself)' : '') + '</small></div>';
      }).join('');
    }

    function refresh() { build(); draw(); panel(); }

    function pick(ev) {
      var r = canvas.getBoundingClientRect(), s = canvas.width / r.width;
      var cx = Math.floor(((ev.clientX - r.left) * s) / CELL) - 1, cy = Math.floor(((ev.clientY - r.top) * s) / CELL) - 1;
      if (cx < 0 || cy < 0 || cx >= N || cy >= N) return;
      cursor.x = cx; cursor.y = cy; sel = grid.data[cy * grid.pw + cx]; draw(); panel();
    }
    canvas.addEventListener('pointermove', pick);
    canvas.addEventListener('pointerdown', pick);
    canvas.tabIndex = 0;
    canvas.addEventListener('keydown', function (e) {
      var d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key]; if (!d) return;
      e.preventDefault();
      cursor.x = Math.max(0, Math.min(N - 1, cursor.x + d[0])); cursor.y = Math.max(0, Math.min(N - 1, cursor.y + d[1]));
      sel = grid.data[cursor.y * grid.pw + cursor.x]; draw(); panel();
    });
    bindSeg($('#pointer .seg'), 'phase', function (p) { phase = +p; refresh(); });
    refresh();
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(draw);
  }

  /* ======================================================================
   * Charts
   * ==================================================================== */
  var STYLE = {
    'QuADA-GS': { color: C.l2, w: 3.6, label: 'QuADA-GS', z: 5 },
    'QuADA-GS ULTRA': { color: C.l2, w: 2.4, dash: '6 5', label: 'ULTRA', z: 4 },
    'GSASR': { color: C.l1, w: 3, label: 'GSASR', z: 3 },
    'GRAPE': { color: C.ink, w: 2, label: 'GRAPE', z: 2 },
    'GaussianSR': { color: '#7D88D8', w: 1.8, label: 'GaussianSR', z: 1 },
    'ContinuousSR*': { color: '#7D88D8', w: 1.8, label: 'ContinuousSR*', z: 1 },
    'CiaoSR': { color: C.gray, w: 1.6, label: 'CiaoSR', z: 1 }
  };
  function styleOf(name) { return STYLE[name] || { color: C.gray, w: 1.3, label: null, z: 0 }; }
  var COST_ORDER = Object.keys(D.cost).sort(function (a, b) { return styleOf(a).z - styleOf(b).z; });

  /** Chart width in CSS pixels, so the SVG is drawn 1:1 and its text stays readable on phones. */
  function chartWidth(host) { return Math.max(320, Math.min(900, Math.round(host.clientWidth || 880))); }
  function redrawOnResize(host, draw) {
    var last = chartWidth(host), t = 0;
    function check() { var w = chartWidth(host); if (Math.abs(w - last) > 8) { last = w; draw(); } }
    if ('ResizeObserver' in window) new ResizeObserver(function () { clearTimeout(t); t = setTimeout(check, 120); }).observe(host);
  }

  function tooltip(host) {
    var t = document.createElement('div'); t.className = 'tooltip'; t.hidden = true; host.appendChild(t);
    return {
      show: function (html, ev) { var r = host.getBoundingClientRect(); t.innerHTML = html; t.hidden = false; t.style.left = Math.max(60, Math.min(r.width - 60, ev.clientX - r.left)) + 'px'; t.style.top = (ev.clientY - r.top) + 'px'; },
      hide: function () { t.hidden = true; }
    };
  }

  function spread(items, minGap, lo, hi) { // push label y-positions apart
    items.sort(function (a, b) { return a.y - b.y; });
    for (var i = 1; i < items.length; i++) if (items[i].y - items[i - 1].y < minGap) items[i].y = items[i - 1].y + minGap;
    var over = items.length ? items[items.length - 1].y - hi : 0;
    if (over > 0) for (var j = items.length - 1; j >= 0; j--) { items[j].y -= over; if (j > 0 && items[j].y - items[j - 1].y < minGap) { items[j - 1].y = items[j].y - minGap; over = 0; } }
    items.forEach(function (it) { if (it.y < lo) it.y = lo; });
  }

  function initCostChart() {
    var host = $('#costChart'); if (!host) return;
    var metric = 'time', focus = null;
    var W = 880, H = 440, M = { l: 62, r: 150, t: 22, b: 44 }, scales = D.scales;
    function layout() { W = chartWidth(host); var n = W < 600; H = n ? 340 : 440; M = { l: n ? 44 : 62, r: n ? 98 : 150, t: 22, b: 44 }; }
    var tip = tooltip(host);

    function draw() {
      layout();
      var old = $('svg', host); if (old) old.remove();
      var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Line chart of ' + (metric === 'time' ? 'inference time' : 'GPU memory') + ' versus upscaling factor for each method' }, host);
      var dom = metric === 'time' ? [3, 30000] : [60, 60000], unit = metric === 'time' ? 'ms' : 'MB';
      var ly = function (v) { return M.t + (H - M.t - M.b) * (1 - (Math.log10(v) - Math.log10(dom[0])) / (Math.log10(dom[1]) - Math.log10(dom[0]))); };
      var lx = function (i) { return M.l + i * (W - M.l - M.r) / (scales.length - 1); };

      var ticks = metric === 'time' ? [10, 100, 1000, 10000] : [100, 1000, 10000];
      ticks.forEach(function (v) {
        svgEl('line', { x1: M.l, x2: W - M.r, y1: ly(v), y2: ly(v), 'class': 'grid' }, svg);
        svgEl('text', { x: M.l - 10, y: ly(v) + 4, 'text-anchor': 'end' }, svg).textContent = v >= 1000 ? (v / 1000) + 'k' : v;
      });
      svgEl('text', { x: M.l - 10, y: 10, 'text-anchor': 'end', 'class': 'axis-title' }, svg).textContent = unit;
      scales.forEach(function (s, i) { svgEl('text', { x: lx(i), y: H - M.b + 24, 'text-anchor': 'middle' }, svg).textContent = '×' + s; });
      svgEl('text', { x: (M.l + W - M.r) / 2, y: H - 4, 'text-anchor': 'middle', 'class': 'axis-title' }, svg).textContent = 'upscaling factor (log scale on the vertical axis)';

      // uncertainty bands for the adaptive methods
      ['QuADA-GS ULTRA', 'QuADA-GS'].forEach(function (name) {
        var d = D.cost[name], mean = d[metric], sd = d[metric + 'Std'], up = [], dn = [];
        scales.forEach(function (s, i) { up.push([lx(i), ly(mean[s] + sd[s])]); dn.unshift([lx(i), ly(Math.max(dom[0] * 1.02, mean[s] - sd[s]))]); });
        var pts = up.concat(dn).map(function (p) { return p[0].toFixed(1) + ',' + p[1].toFixed(1); }).join(' ');
        svgEl('polygon', { points: pts, fill: C.l2, 'fill-opacity': name === 'QuADA-GS' ? 0.2 : 0.09, 'data-m': name, 'class': 'band' }, svg);
      });

      var labels = [];
      COST_ORDER.forEach(function (name) {
        var st = styleOf(name), vals = D.cost[name][metric];
        var dpath = scales.map(function (s, i) { return (i ? 'L' : 'M') + lx(i).toFixed(1) + ',' + ly(vals[s]).toFixed(1); }).join(' ');
        var p = svgEl('path', { d: dpath, stroke: st.color, 'stroke-width': st.w, 'class': 'line', 'data-m': name }, svg);
        if (st.dash) p.setAttribute('stroke-dasharray', st.dash);
        var hit = svgEl('path', { d: dpath, 'class': 'hit', fill: 'none', stroke: 'transparent', 'stroke-width': 16, 'pointer-events': 'stroke', 'data-m': name }, svg);
        hit.addEventListener('pointerenter', function () { focus = name; applyFocus(svg); });
        hit.addEventListener('pointerleave', function () { focus = null; applyFocus(svg); tip.hide(); });
        hit.addEventListener('pointermove', function (ev) {
          var r = svg.getBoundingClientRect(), x = (ev.clientX - r.left) * (W / r.width);
          var i = Math.max(0, Math.min(scales.length - 1, Math.round((x - M.l) / ((W - M.l - M.r) / (scales.length - 1))))), s = scales[i];
          var sd = D.cost[name][metric + 'Std'][s];
          tip.show('<b>' + name + '</b><small>×' + s + ': ' + fmt(vals[s]) + ' ' + unit + (sd != null ? ' ± ' + fmt(sd) : '') + '</small>', ev);
        });
        if (st.label) labels.push({ name: name, y: ly(vals[scales[scales.length - 1]]), color: st.color, text: st.label });
      });
      spread(labels, 15, M.t + 6, H - M.b);
      labels.forEach(function (l) { var t = svgEl('text', { x: W - M.r + 10, y: l.y + 4, fill: l.color, 'class': 'lbl', 'data-m': l.name }, svg); t.textContent = l.text; t.style.fill = l.color; });
      applyFocus(svg);
    }

    function applyFocus(svg) {
      $$('.line, .lbl, .band', svg).forEach(function (e) { e.classList.toggle('dim', !!focus && e.getAttribute('data-m') !== focus); });
    }

    bindSeg($('#costMetric'), 'metric', function (m) { metric = m; draw(); });
    draw();
    redrawOnResize(host, draw);
  }

  var METRIC_INFO = {
    psnr: { label: 'PSNR (dB), higher is better', dec: 1, better: 1 },
    ssim: { label: 'SSIM, higher is better', dec: 3, better: 1 },
    lpips: { label: 'LPIPS, lower is better (axis flipped)', dec: 3, better: -1 },
    dists: { label: 'DISTS, lower is better (axis flipped)', dec: 3, better: -1 }
  };

  function niceStep(raw) { var p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p; return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p; }

  function initScatter() {
    var host = $('#scatter'); if (!host) return;
    var ds = 'Urban100', metric = 'psnr', scale = '4';
    var W = 880, H = 500, M = { l: 66, r: 28, t: 18, b: 56 };
    function layout() { W = chartWidth(host); var n = W < 600; H = n ? 400 : 500; M = { l: n ? 52 : 66, r: n ? 14 : 28, t: 18, b: 56 }; }
    var tip = tooltip(host);
    var PRIORITY = ['QuADA-GS', 'QuADA-GS ULTRA', 'GSASR', 'GRAPE', 'ContinuousSR*', 'GaussianSR', 'CiaoSR', 'LMF', 'SRNO', 'Meta-SR', 'LIIF', 'LTE', 'LINF'];

    function draw() {
      layout();
      var old = $('svg', host); if (old) old.remove();
      var info = METRIC_INFO[metric], better = info.better;
      var pts = Object.keys(D.quality[ds]).map(function (n) {
        return { name: n, t: D.cost[n].time[scale], mem: D.cost[n].mem[scale], v: D.quality[ds][n][metric][scale] };
      }).filter(function (p) { return p.v != null; });
      var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Scatter plot of ' + metric.toUpperCase() + ' against inference time at x' + scale + ' on ' + ds }, host);

      var tmin = Math.min.apply(null, pts.map(function (p) { return p.t; })), tmax = Math.max.apply(null, pts.map(function (p) { return p.t; }));
      var x0 = Math.pow(10, Math.floor(Math.log10(tmin * 0.8))), x1 = Math.pow(10, Math.ceil(Math.log10(tmax * 1.25)));
      var vs = pts.map(function (p) { return p.v; }), lo = Math.min.apply(null, vs), hi = Math.max.apply(null, vs), pad = (hi - lo) * 0.14 || 0.01;
      lo -= pad; hi += pad;
      var X = function (t) { return M.l + (Math.log10(t) - Math.log10(x0)) / (Math.log10(x1) - Math.log10(x0)) * (W - M.l - M.r); };
      var Y = function (v) { var u = (v - lo) / (hi - lo); if (better < 0) u = 1 - u; return M.t + (H - M.t - M.b) * (1 - u); };

      // y grid
      var step = niceStep((hi - lo) / 5), start = Math.ceil(lo / step) * step;
      for (var v = start; v <= hi; v += step) {
        svgEl('line', { x1: M.l, x2: W - M.r, y1: Y(v), y2: Y(v), 'class': 'grid' }, svg);
        svgEl('text', { x: M.l - 10, y: Y(v) + 4, 'text-anchor': 'end' }, svg).textContent = v.toFixed(info.dec);
      }
      // x grid (decades)
      for (var e = Math.log10(x0); e <= Math.log10(x1) + 1e-9; e++) {
        var t = Math.pow(10, e);
        svgEl('line', { x1: X(t), x2: X(t), y1: M.t, y2: H - M.b, 'class': 'grid' }, svg);
        svgEl('text', { x: X(t), y: H - M.b + 20, 'text-anchor': 'middle' }, svg).textContent = t >= 1000 ? (t / 1000) + 's' : t + ' ms';
      }
      svgEl('text', { x: (M.l + W - M.r) / 2, y: H - 8, 'text-anchor': 'middle', 'class': 'axis-title' }, svg).textContent = 'inference time (log scale), faster to the left';
      var yt = svgEl('text', { x: 0, y: 0, 'class': 'axis-title', transform: 'translate(16,' + (M.t + (H - M.t - M.b) / 2) + ') rotate(-90)', 'text-anchor': 'middle' }, svg); yt.textContent = info.label;

      // Pareto front
      var sorted = pts.slice().sort(function (a, b) { return a.t - b.t; }), best = -Infinity, front = [];
      sorted.forEach(function (p) { if (p.v * better > best) { front.push(p); best = p.v * better; } });
      svgEl('polyline', { points: front.map(function (p) { return X(p.t).toFixed(1) + ',' + Y(p.v).toFixed(1); }).join(' '), fill: 'none', stroke: C.ink, 'stroke-width': 1.5, 'stroke-dasharray': '5 5', opacity: 0.55 }, svg);

      var memMax = Math.max.apply(null, pts.map(function (p) { return p.mem; }));
      var rad = function (p) { return 5 + 11 * Math.sqrt(p.mem / memMax); };
      var order = pts.slice().sort(function (a, b) { return styleOf(a.name).z - styleOf(b.name).z; });
      var placed = [];
      order.forEach(function (p) {
        var st = styleOf(p.name), r = rad(p), cx = X(p.t), cy = Y(p.v);
        var ultra = p.name === 'QuADA-GS ULTRA';
        var c = svgEl('circle', { cx: cx, cy: cy, r: r, fill: ultra ? '#fff' : st.color, 'fill-opacity': ultra ? 1 : (st.z >= 2 ? 0.92 : 0.6), stroke: ultra ? C.l2 : '#fff', 'stroke-width': ultra ? 3 : 1.5, 'class': 'dot' }, svg);
        c.addEventListener('pointermove', function (ev) {
          tip.show('<b>' + p.name + '</b><small>' + p.t + ' ms · ' + fmt(p.mem) + ' MB · ' + info.label.split(',')[0].split(' (')[0] + ' ' + p.v.toFixed(info.dec + (metric === 'psnr' ? 1 : 1)) + '</small>', ev);
        });
        c.addEventListener('pointerleave', function () { tip.hide(); });
      });
      PRIORITY.forEach(function (name) {
        var p = pts.filter(function (q) { return q.name === name; })[0]; if (!p) return;
        var st = styleOf(name), r = rad(p), text = st.label || name, w = text.length * 6.6 + 8;
        var bx = X(p.t) + r + 5, by = Y(p.v) - 7, box = { x: bx, y: by, w: w, h: 15 };
        if (bx + w > W - 4) { box.x = X(p.t) - r - 5 - w; }
        var clash = placed.some(function (o) { return !(box.x + box.w < o.x || o.x + o.w < box.x || box.y + box.h < o.y || o.y + o.h < box.y); });
        if (clash && st.z < 2) return;
        placed.push(box);
        var tx = svgEl('text', { x: box.x < X(p.t) ? box.x + w - 4 : box.x, y: by + 11, 'text-anchor': box.x < X(p.t) ? 'end' : 'start', 'class': 'lbl' }, svg);
        tx.textContent = text; tx.style.fill = st.z >= 2 ? st.color : C.ink2;
      });
    }

    bindSeg($('#scatterDataset'), 'ds', function (v) { ds = v; draw(); });
    bindSeg($('#scatterMetric'), 'm', function (v) { metric = v; draw(); });
    bindSeg($('#scatterScale'), 's', function (v) { scale = v; draw(); });
    draw();
    redrawOnResize(host, draw);
  }

  /* ======================================================================
   * Comparison slider
   * ==================================================================== */
  function initCompare() {
    var stage = $('#cmpStage'); if (!stage) return;
    var PANELS = [
      ['lr-input', 'LR input'], ['meta-sr', 'Meta-SR'], ['liif', 'LIIF'], ['lte', 'LTE'], ['srno', 'SRNO'], ['lmf', 'LMF'],
      ['ciaosr', 'CiaoSR'], ['gaussiansr', 'GaussianSR'], ['gsasr', 'GSASR'], ['structure-map', 'Structure map'], ['quada-gs', 'QuADA-GS'], ['ground-truth', 'Ground truth']
    ];
    var L = $('#cmpLeft'), R = $('#cmpRight'), IL = $('#cmpImgLeft'), IR = $('#cmpImgRight'), range = $('#cmpRange');
    var opts = PANELS.map(function (p) { return '<option value="' + p[0] + '">' + p[1] + '</option>'; }).join('');
    L.innerHTML = opts; R.innerHTML = opts; L.value = 'gsasr'; R.value = 'quada-gs';
    function label(slug) { return PANELS.filter(function (p) { return p[0] === slug; })[0][1]; }
    function setImgs() {
      IL.src = 'static/images/compare/' + L.value + '.jpg'; IR.src = 'static/images/compare/' + R.value + '.jpg';
      IL.alt = label(L.value) + ' result'; IR.alt = label(R.value) + ' result';
      $('#cmpTagLeft').textContent = label(L.value); $('#cmpTagRight').textContent = label(R.value);
    }
    function setPos() { stage.style.setProperty('--pos', range.value + '%'); }
    L.addEventListener('change', setImgs); R.addEventListener('change', setImgs); range.addEventListener('input', setPos);
    setImgs(); setPos();
  }

  /* ======================================================================
   * Small things: copy buttons, lightbox, missing figures, scroll spy
   * ==================================================================== */
  function initMisc() {
    $$('.copy').forEach(function (b) {
      b.addEventListener('click', function () {
        var text = $(b.dataset.copy).textContent;
        var done = function () { var o = b.textContent; b.textContent = 'Copied'; b.classList.add('done'); setTimeout(function () { b.textContent = o; b.classList.remove('done'); }, 1400); };
        if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, fallback); else fallback();
        function fallback() { var ta = document.createElement('textarea'); ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0'; document.body.appendChild(ta); ta.select(); try { document.execCommand('copy'); done(); } catch (e) { /* ignore */ } ta.remove(); }
      });
    });

    // A figure that is not in static/images yet shows a clear placeholder instead of a broken icon.
    document.addEventListener('error', function (e) {
      var im = e.target;
      if (!im || im.tagName !== 'IMG' || !im.classList.contains('paper-fig')) return;
      var d = document.createElement('div'); d.className = 'fig-missing';
      d.innerHTML = '<strong>Figure not added yet</strong><span>' + im.getAttribute('src') + '</span>';
      im.replaceWith(d);
    }, true);

    var lb = $('#lightbox'), lbImg = $('img', lb);
    document.addEventListener('click', function (e) {
      var im = e.target.closest && e.target.closest('img.paper-fig');
      if (im && lb.showModal) { lbImg.src = im.currentSrc || im.src; lbImg.alt = im.alt; lb.showModal(); }
    });
    lb.addEventListener('click', function () { lb.close(); });

    if ('IntersectionObserver' in window) {
      var links = {}; $$('.topbar-nav a').forEach(function (a) { links[a.getAttribute('href').slice(1)] = a; });
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          if (!en.isIntersecting) return;
          var id = en.target.id, key = links[id] ? id : null;
          if (!key) return;
          Object.keys(links).forEach(function (k) { links[k].removeAttribute('aria-current'); });
          links[key].setAttribute('aria-current', 'true');
        });
      }, { rootMargin: '-35% 0px -55% 0px' });
      $$('main section[id]').forEach(function (s) { io.observe(s); });
    }
  }

  /* ---------- boot ---------- */
  function boot() {
    initDemo(); initPointer(); initCostChart(); initScatter(); initCompare(); initMisc();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
