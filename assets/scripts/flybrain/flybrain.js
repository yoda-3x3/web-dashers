// Fly-style spiking network (sparse recurrent LIF; NOT the real Drosophila connectome). Ported from flyometrydash sim.js.
// Enabled only via ?flybrain=1. Hooked from GameScene.update through window.FlyBrain.tick(scene, deltaMs).
(function (root) {
  const UNIT = 60, BINS = 12, CH = 3, N_IN = BINS * CH + 4, N_H = 48;
  const TICK = 1 / 60, DECAY = Math.exp(-TICK / 0.05), HOLD = 4, POP = 8;
  const KEY = "flybrain_best_v1";

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const gauss = (r) => Math.sqrt(-2 * Math.log(r() || 1e-9)) * Math.cos(2 * Math.PI * r());

  const TOPO = (() => {
    const r = mulberry32(1337), inPre = [], inPost = [], recPre = [], recPost = [];
    for (let h = 0; h < N_H; h++) for (let i = 0; i < N_IN; i++) if (r() < 0.35) { inPre.push(i); inPost.push(h); }
    for (let h = 0; h < N_H; h++) for (let k = 0; k < N_H; k++) if (h !== k && r() < 0.12) { recPre.push(k); recPost.push(h); }
    return { inPre, inPost, recPre, recPost, nIn: inPre.length, nRec: recPre.length, size: inPre.length + recPre.length + 2 * N_H + 1 };
  })();

  class Brain {
    constructor(g) {
      const T = TOPO;
      this.wIn = g.subarray(0, T.nIn);
      this.wRec = g.subarray(T.nIn, T.nIn + T.nRec);
      this.wOut = g.subarray(T.nIn + T.nRec, T.nIn + T.nRec + N_H);
      this.bias = g.subarray(T.nIn + T.nRec + N_H, T.nIn + T.nRec + 2 * N_H);
      this.outBias = g[T.size - 1];
      this.v = new Float32Array(N_H); this.spk = new Uint8Array(N_H); this.cur = new Float32Array(N_H);
      this.vOut = 0;
    }
    step(inp) {
      const T = TOPO, cur = this.cur, spk = this.spk, v = this.v;
      for (let h = 0; h < N_H; h++) cur[h] = this.bias[h];
      for (let s = 0; s < T.nIn; s++) cur[T.inPost[s]] += this.wIn[s] * inp[T.inPre[s]];
      for (let s = 0; s < T.nRec; s++) if (spk[T.recPre[s]]) cur[T.recPost[s]] += this.wRec[s];
      let o = this.outBias;
      for (let h = 0; h < N_H; h++) {
        v[h] = v[h] * DECAY + cur[h];
        if (v[h] >= 1) { spk[h] = 1; v[h] = 0; o += this.wOut[h]; } else spk[h] = 0;
      }
      this.vOut = this.vOut * DECAY + o;
      if (this.vOut >= 1) { this.vOut = 0; return 1; }
      return 0;
    }
  }

  const randomGenome = (r) => { const g = new Float32Array(TOPO.size); for (let i = 0; i < g.length; i++) g[i] = gauss(r) * 0.4; return g; };
  const mutate = (g, r, sigma) => {
    const c = Float32Array.from(g), rate = 0.02 + r() * 0.1;
    for (let i = 0; i < c.length; i++) if (r() < rate) c[i] += gauss(r) * sigma;
    return c;
  };

  const inp = new Float32Array(N_IN);
  function sense(scene) {
    const st = scene._state, px = scene._playerWorldX, py = st.y;
    inp.fill(0);
    const objs = scene._level.getNearbySectionObjects(px);
    for (let i = 0; i < objs.length; i++) {
      const o = objs[i], hw = (o.w || UNIT) / 2;
      const left = o.x - hw - px;
      if (left + 2 * hw < 0 || left > BINS * UNIT) continue;
      const c = o.type === "hazard" ? 0 : o.type === "solid" ? 1 : -1;
      if (c < 0) continue;
      const b0 = Math.max(0, Math.floor(left / UNIT)), b1 = Math.min(BINS - 1, Math.floor((left + 2 * hw) / UNIT));
      for (let b = b0; b <= b1; b++) {
        inp[b * CH + c] = 1;
        // raised: top of object is above the player's centre
        if (o.y + (o.h || UNIT) / 2 > py + UNIT / 2) inp[b * CH + 2] = 1;
      }
    }
    const k = BINS * CH;
    inp[k] = st.onGround ? 1 : 0;
    inp[k + 1] = Math.max(-1, Math.min(1, (py - 30) / (UNIT * 6)));
    inp[k + 2] = Math.max(-1, Math.min(1, (st.yVelocity || 0) / 20));
    inp[k + 3] = st.isFlying ? 1 : 0;
    return inp;
  }

  const FB = {
    enabled: false, attempts: 0, best: 0, gen: 0, brain: null, hold: 0, acc: 0, held: false,
    pop: [], idx: 0, fits: [], parent: null, parentFit: -1, rnd: mulberry32(Date.now() & 0xffff),
    lastAttempt: -1, curPct: 0, raster: [], hud: null,
    init() {
      this.enabled = true;
      try {
        const s = JSON.parse(localStorage.getItem(KEY) || "null");
        if (s && s.genome && s.genome.length === TOPO.size) { this.parent = Float32Array.from(s.genome); this.parentFit = s.fit || 0; this.best = s.fit || 0; this.gen = s.gen || 0; }
      } catch (e) {}
      this._mkHud();
      this._nextGenome();
    },
    _nextGenome() {
      if (this.idx >= POP || !this.pop.length) {
        if (this.pop.length) this._endGen();
        this.pop = [];
        for (let i = 0; i < POP; i++) this.pop.push(this.parent ? (i === 0 ? this.parent : mutate(this.parent, this.rnd, 0.25)) : randomGenome(this.rnd));
        this.idx = 0; this.fits = [];
      }
      this.brain = new Brain(this.pop[this.idx]);
    },
    _endGen() {
      let bi = 0;
      for (let i = 1; i < this.fits.length; i++) if (this.fits[i] > this.fits[bi]) bi = i;
      this.gen++;
      if (this.fits[bi] > this.parentFit) {
        this.parentFit = this.fits[bi]; this.parent = Float32Array.from(this.pop[bi]);
        try { localStorage.setItem(KEY, JSON.stringify({ genome: Array.from(this.parent), fit: this.parentFit, gen: this.gen })); } catch (e) {}
      }
    },
    _endAttempt() {
      this.fits[this.idx] = this.curPct;
      if (this.curPct > this.best) this.best = this.curPct;
      this.idx++; this.curPct = 0; this.hold = 0; this.raster = [];
      this._nextGenome();
    },
    tick(scene, deltaMs) {
      if (!this.enabled || !scene._state || !scene._level || scene._menuActive || scene._paused) return;
      const att = scene._levelAttempts;
      if (this.lastAttempt !== -1 && att !== this.lastAttempt) { this._endAttempt(); }
      this.lastAttempt = att;
      const end = scene._level.endXPos || 1;
      this.curPct = Math.max(this.curPct, Math.min(100, (scene._playerWorldX / end) * 100));
      if (this.curPct > this.best) this.best = this.curPct;
      if (scene._state.isDead || scene._levelWon) { this._hud(); return; }
      this.acc += deltaMs / 1000;
      while (this.acc >= TICK) {
        this.acc -= TICK;
        if (this.brain.step(sense(scene))) this.hold = HOLD;
        this.raster.push(Uint8Array.from(this.brain.spk));
        if (this.raster.length > 120) this.raster.shift();
        const want = this.hold > 0;
        if (this.hold > 0) this.hold--;
        if (want && !this.held) { this.held = true; scene._pushButton(true); }
        else if (!want && this.held) { this.held = false; scene._releaseButton(true); }
      }
      this._hud();
    },
    _mkHud() {
      const d = document.createElement("div");
      d.style.cssText = "position:fixed;left:6px;top:6px;z-index:9999;pointer-events:none;font:12px monospace;color:#7f7;background:#000a;padding:4px";
      const c = document.createElement("canvas"); c.width = 240; c.height = 96;
      const t = document.createElement("div");
      d.appendChild(t); d.appendChild(c); document.body.appendChild(d);
      this.hud = { t, c: c.getContext("2d") };
    },
    _hud() {
      const h = this.hud; if (!h) return;
      h.t.textContent = "Fly-style net (not real connectome) gen " + this.gen + " cand " + (this.idx + 1) + "/" + POP +
        " attempt " + this.attempts() + " now " + this.curPct.toFixed(1) + "% best " + this.best.toFixed(1) + "%";
      const g = h.c; g.clearRect(0, 0, 240, 96);
      g.fillStyle = "#7f7";
      for (let x = 0; x < this.raster.length; x++) {
        const s = this.raster[x];
        for (let n = 0; n < N_H; n++) if (s[n]) g.fillRect(x * 2, n * 2, 2, 2);
      }
    },
    attempts() { return this._att || 0; },
  };
  const _tick = FB.tick;
  FB.tick = function (scene, dt) { this._att = scene._levelAttempts; return _tick.call(this, scene, dt); };
  root.FlyBrain = FB;
})(window);
