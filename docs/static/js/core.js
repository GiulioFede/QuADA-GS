/*
 * QuADA-GS project page — core logic for the interactive demos.
 *
 * This is an *illustrative* re-implementation of three ideas from the paper so
 * that visitors can play with them in the browser:
 *   1. Structure Map   (structure tensor -> score map -> two thresholds -> levels 0/1/2)
 *   2. Quadtree        (level 0 = 1 leaf of side 4, level 1 = 4 leaves of side 2,
 *                       level 2 = 16 leaves of side 1)
 *   3. Pointer Grid    (dense index P of size 4H x 4W + 9 O(1) lookups per feature)
 * It is NOT the trained network: no learned predictor, no MLP decoder, no CUDA rasterizer.
 *
 * Pure functions only (no DOM) so the file can be unit-tested in Node.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.QuADA = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // Final score weights [w_texture, w_edge] = [3, 1] (paper ablation, bold row).
  var W_TEXTURE = 3;
  var W_EDGE = 1;
  // The demo's own calibration so the paper's default thresholds [0.1, 1.4] give a
  // sensible allocation on the demo scenes. The model works on its own score scale.
  var SCORE_GAIN = 260;

  /* ------------------------------------------------------------------ *
   * 1. Structure Map
   * ------------------------------------------------------------------ */

  function toGray(rgba, w, h) {
    var g = new Float32Array(w * h);
    for (var i = 0, p = 0; i < g.length; i++, p += 4) {
      g[i] = (0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2]) / 255;
    }
    return g;
  }

  function clampIdx(v, n) { return v < 0 ? 0 : v >= n ? n - 1 : v; }

  // 3-tap [1 2 1]/4 smoothing, separable, replicate border.
  function smooth121(src, w, h) {
    var tmp = new Float32Array(w * h), out = new Float32Array(w * h);
    var x, y;
    for (y = 0; y < h; y++) {
      for (x = 0; x < w; x++) {
        var a = src[y * w + clampIdx(x - 1, w)], b = src[y * w + x], c = src[y * w + clampIdx(x + 1, w)];
        tmp[y * w + x] = (a + 2 * b + c) * 0.25;
      }
    }
    for (y = 0; y < h; y++) {
      var ym = clampIdx(y - 1, h), yp = clampIdx(y + 1, h);
      for (x = 0; x < w; x++) {
        out[y * w + x] = (tmp[ym * w + x] + 2 * tmp[y * w + x] + tmp[yp * w + x]) * 0.25;
      }
    }
    return out;
  }

  /**
   * Per-pixel score = max(0, w_t * lambda2 + w_e * (lambda1 - lambda2)),
   * with lambda1 >= lambda2 the eigenvalues of the 2x2 structure tensor
   * (near zero: flat, one dominant: edge, two comparable: texture).
   */
  function scoreMap(gray, w, h, gain) {
    gain = gain || SCORE_GAIN;
    var ix = new Float32Array(w * h), iy = new Float32Array(w * h);
    var x, y, i;
    for (y = 0; y < h; y++) {
      var y0 = clampIdx(y - 1, h), y2 = clampIdx(y + 1, h);
      for (x = 0; x < w; x++) {
        var x0 = clampIdx(x - 1, w), x2 = clampIdx(x + 1, w);
        var a = gray[y0 * w + x0], b = gray[y0 * w + x], c = gray[y0 * w + x2];
        var d = gray[y * w + x0], f = gray[y * w + x2];
        var g = gray[y2 * w + x0], hh = gray[y2 * w + x], k = gray[y2 * w + x2];
        ix[y * w + x] = ((c + 2 * f + k) - (a + 2 * d + g)) / 8;
        iy[y * w + x] = ((g + 2 * hh + k) - (a + 2 * b + c)) / 8;
      }
    }
    var sxx = new Float32Array(w * h), syy = new Float32Array(w * h), sxy = new Float32Array(w * h);
    for (i = 0; i < w * h; i++) {
      sxx[i] = ix[i] * ix[i];
      syy[i] = iy[i] * iy[i];
      sxy[i] = ix[i] * iy[i];
    }
    sxx = smooth121(smooth121(sxx, w, h), w, h);
    syy = smooth121(smooth121(syy, w, h), w, h);
    sxy = smooth121(smooth121(sxy, w, h), w, h);

    var score = new Float32Array(w * h);
    for (i = 0; i < w * h; i++) {
      var tr = sxx[i] + syy[i];
      var det = sxx[i] * syy[i] - sxy[i] * sxy[i];
      var disc = Math.sqrt(Math.max(0, tr * tr * 0.25 - det));
      var l1 = tr * 0.5 + disc, l2 = Math.max(0, tr * 0.5 - disc);
      var s = W_TEXTURE * l2 + W_EDGE * (l1 - l2);
      // Soft-saturate into (0, 2): the paper's threshold sliders run from 0 to 2, so a threshold of 2
      // must mean "refine nothing", just as 0 means "refine everything".
      var raw = Math.max(0, s) * gain;
      score[i] = Math.min(2 - 1e-6, 2 * Math.tanh(raw / 2)) + 1e-6; // tiny floor so "score <= 0" never ties
    }
    return score;
  }

  /** Average-pool a (w x h) map by an integer factor f -> (w/f x h/f). */
  function poolScore(score, w, h, f) {
    var fw = Math.floor(w / f), fh = Math.floor(h / f);
    var out = new Float32Array(fw * fh);
    var inv = 1 / (f * f);
    for (var y = 0; y < fh; y++) {
      for (var x = 0; x < fw; x++) {
        var s = 0;
        for (var j = 0; j < f; j++) for (var i = 0; i < f; i++) s += score[(y * f + j) * w + x * f + i];
        out[y * fw + x] = s * inv;
      }
    }
    return out;
  }

  /** level 0: score <= phi1 (x1) | level 1: phi1 < score <= phi2 (x2) | level 2: score > phi2 (x4) */
  function quantize(featScore, phi1, phi2) {
    var lv = new Uint8Array(featScore.length);
    var p1 = Math.fround(phi1), p2 = Math.fround(phi2); // scores are float32: compare like with like
    for (var i = 0; i < lv.length; i++) {
      var s = featScore[i];
      lv[i] = s <= p1 ? 0 : (s <= p2 ? 1 : 2);
    }
    return lv;
  }

  /* ------------------------------------------------------------------ *
   * 2. Quadtree leaves
   * ------------------------------------------------------------------ */

  /**
   * Leaves of the sparse quadtree after `phase` expansions (0 = roots only,
   * 1 = after the first blind expansion, 2 = after the second = final).
   * Coordinates live on the Pointer Grid P (size 4W x 4H); a root owns a 4x4 block.
   */
  function buildLeaves(levels, W, H, phase) {
    if (phase === undefined) phase = 2;
    var n0 = 0, n1 = 0, n2 = 0, i;
    for (i = 0; i < levels.length; i++) { if (levels[i] === 0) n0++; else if (levels[i] === 1) n1++; else n2++; }
    var total;
    if (phase === 0) total = levels.length;
    else if (phase === 1) total = n0 + 4 * (n1 + n2);
    else total = n0 + 4 * n1 + 16 * n2;

    var x0 = new Uint16Array(total), y0 = new Uint16Array(total);
    var S = new Uint8Array(total), lvl = new Uint8Array(total), rootIdx = new Uint32Array(total);
    var m = 0;

    function push(x, y, s, level, r) { x0[m] = x; y0[m] = y; S[m] = s; lvl[m] = level; rootIdx[m] = r; m++; }

    for (var r = 0; r < levels.length; r++) {
      var rx = (r % W) * 4, ry = Math.floor(r / W) * 4, L = levels[r];
      if (phase === 0 || L === 0) { push(rx, ry, 4, L, r); continue; }
      for (var b = 0; b < 4; b++) {
        var cx = rx + (b & 1) * 2, cy = ry + (b >> 1) * 2;
        if (phase === 1 || L === 1) { push(cx, cy, 2, L, r); continue; }
        for (var q = 0; q < 4; q++) push(cx + (q & 1), cy + (q >> 1), 1, L, r);
      }
    }
    return { count: total, x0: x0, y0: y0, S: S, level: lvl, root: rootIdx, W: W, H: H, counts: { l0: n0, l1: n1, l2: n2 } };
  }

  /** Gaussians if every feature were expanded x4 (GSASR-style dense upsampling): 16 * H * W. */
  function denseCount(W, H) { return 16 * W * H; }

  /* ------------------------------------------------------------------ *
   * 3. Pointer Grid + Hierarchical Pointer Convolution lookups
   * ------------------------------------------------------------------ */

  /** Dense index P (4H x 4W): every cell stores the id of the leaf that owns it. */
  function pointerGrid(leaves) {
    var pw = leaves.W * 4, ph = leaves.H * 4;
    var P = new Int32Array(pw * ph).fill(-1);
    for (var m = 0; m < leaves.count; m++) {
      var s = leaves.S[m], x0 = leaves.x0[m], y0 = leaves.y0[m];
      for (var y = y0; y < y0 + s; y++) P.fill(m, y * pw + x0, y * pw + x0 + s);
    }
    return { data: P, pw: pw, ph: ph };
  }

  /** Offsets along each axis: O(S) = { -1, floor(S/2), S }. */
  function offsets(S) { return [-1, Math.floor(S / 2), S]; }

  /** The 9 lookup coordinates (row-major over dy then dx) for a leaf owning an SxS region at (x0, y0). */
  function lookupCoords(x0, y0, S) {
    var o = offsets(S), out = [];
    for (var a = 0; a < 3; a++) for (var b = 0; b < 3; b++) out.push([x0 + o[b], y0 + o[a]]);
    return out;
  }

  /** Neighbor ids from P for one leaf; -1 where the lookup falls outside the image. */
  function neighborIds(grid, x0, y0, S) {
    var c = lookupCoords(x0, y0, S), out = new Int32Array(9);
    for (var i = 0; i < 9; i++) {
      var x = c[i][0], y = c[i][1];
      out[i] = (x < 0 || y < 0 || x >= grid.pw || y >= grid.ph) ? -1 : grid.data[y * grid.pw + x];
    }
    return out;
  }

  /* ------------------------------------------------------------------ *
   * Gaussian "splat" preview (illustrative, normalized blending)
   * ------------------------------------------------------------------ */

  function integralImage(rgba, w, h) {
    var iw = w + 1, I = new Float64Array(iw * (h + 1) * 3);
    for (var y = 0; y < h; y++) {
      var rs0 = 0, rs1 = 0, rs2 = 0;
      for (var x = 0; x < w; x++) {
        var p = (y * w + x) * 4;
        rs0 += rgba[p]; rs1 += rgba[p + 1]; rs2 += rgba[p + 2];
        var o = ((y + 1) * iw + (x + 1)) * 3, up = (y * iw + (x + 1)) * 3;
        I[o] = I[up] + rs0; I[o + 1] = I[up + 1] + rs1; I[o + 2] = I[up + 2] + rs2;
      }
    }
    return { data: I, iw: iw };
  }

  function regionMean(ii, x0, y0, s, out) {
    var d = ii.data, iw = ii.iw;
    var a = (y0 * iw + x0) * 3, b = (y0 * iw + x0 + s) * 3, c = ((y0 + s) * iw + x0) * 3, e = ((y0 + s) * iw + x0 + s) * 3;
    var inv = 1 / (s * s);
    out[0] = (d[e] - d[b] - d[c] + d[a]) * inv;
    out[1] = (d[e + 1] - d[b + 1] - d[c + 1] + d[a + 1]) * inv;
    out[2] = (d[e + 2] - d[b + 2] - d[c + 2] + d[a + 2]) * inv;
  }

  /**
   * Render the leaves as 2D Gaussians (sigma proportional to the leaf side S) with
   * normalized blending: out = sum(w_i * c_i) / sum(w_i).  `src` is RGBA at Pointer Grid
   * resolution; the result is RGBA at `scale` times that resolution.
   */
  function renderGaussians(src, pw, ph, leaves, scale, sigmaK) {
    if (sigmaK === undefined) sigmaK = 0.5;
    var ow = pw * scale, oh = ph * scale;
    var acc = new Float32Array(ow * oh * 4); // r g b w
    var ii = integralImage(src, pw, ph);

    // One kernel per leaf size (centers share the same sub-pixel phase for a given S).
    var kernels = {}, sizes = [1, 2, 4];
    sizes.forEach(function (S) {
      var sigma = sigmaK * S * scale, r = Math.ceil(3 * sigma);
      var c = S * scale * 0.5 - 0.5; // center relative to the leaf's top-left pixel
      var ulo = Math.floor(c) - r, uhi = Math.floor(c) + r + 1, n = uhi - ulo + 1;
      var wts = new Float32Array(n * n);
      for (var v = 0; v < n; v++) for (var u = 0; u < n; u++) {
        var dx = (ulo + u) - c, dy = (ulo + v) - c;
        wts[v * n + u] = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
      }
      kernels[S] = { ulo: ulo, n: n, w: wts };
    });

    var col = [0, 0, 0];
    for (var m = 0; m < leaves.count; m++) {
      var S = leaves.S[m], k = kernels[S];
      regionMean(ii, leaves.x0[m], leaves.y0[m], S, col);
      var bx = leaves.x0[m] * scale + k.ulo, by = leaves.y0[m] * scale + k.ulo;
      for (var v = 0; v < k.n; v++) {
        var py = by + v; if (py < 0 || py >= oh) continue;
        for (var u = 0; u < k.n; u++) {
          var px = bx + u; if (px < 0 || px >= ow) continue;
          var wgt = k.w[v * k.n + u], o = (py * ow + px) * 4;
          acc[o] += wgt * col[0]; acc[o + 1] += wgt * col[1]; acc[o + 2] += wgt * col[2]; acc[o + 3] += wgt;
        }
      }
    }
    var out = new Uint8ClampedArray(ow * oh * 4);
    for (var i = 0; i < ow * oh; i++) {
      var w4 = acc[i * 4 + 3];
      if (w4 > 1e-9) {
        out[i * 4] = acc[i * 4] / w4; out[i * 4 + 1] = acc[i * 4 + 1] / w4; out[i * 4 + 2] = acc[i * 4 + 2] / w4;
      }
      out[i * 4 + 3] = 255;
    }
    return { data: out, width: ow, height: oh };
  }

  return {
    W_TEXTURE: W_TEXTURE, W_EDGE: W_EDGE, SCORE_GAIN: SCORE_GAIN,
    toGray: toGray, scoreMap: scoreMap, poolScore: poolScore, quantize: quantize,
    buildLeaves: buildLeaves, denseCount: denseCount,
    pointerGrid: pointerGrid, offsets: offsets, lookupCoords: lookupCoords, neighborIds: neighborIds,
    renderGaussians: renderGaussians
  };
});
