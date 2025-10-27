(function () {
    if (!window.GA || !("IntersectionObserver" in window)) return;

    const VIS_THRESHOLD_DEFAULT = 0.5;
    const HEARTBEAT_MS_DEFAULT = 15000;
    const MIN_VISIBLE_MS_DEFAULT = 250;

    const state = new Map(); // element_id -> {visible,since,acc,hb,seen,cfg}

    const now = () => (performance && performance.now) ? performance.now() : Date.now();

    function getCfgFor(el) {
      const t = parseFloat(el.getAttribute("data-ga-threshold") || "") || VIS_THRESHOLD_DEFAULT;
      const hb = parseInt(el.getAttribute("data-ga-heartbeat-ms") || "") || HEARTBEAT_MS_DEFAULT;
      const min = parseInt(el.getAttribute("data-ga-min-visible-ms") || "") || MIN_VISIBLE_MS_DEFAULT;
      return { threshold: Math.max(0, Math.min(1, t)), heartbeatMs: hb, minVisibleMs: min };
    }

    function ensureEntry(id) {
      if (!state.has(id)) state.set(id, { visible:false, since:null, acc:0, hb:null, seen:false, cfg:null });
      return state.get(id);
    }

    function send(name, params){ window.GA.event(name, params); }

    function onEnter(id, cfg) {
      const s = ensureEntry(id);
      if (s.visible) return;
      s.visible = true;
      s.since = now();
      s.cfg = cfg;
      if (!s.seen) { s.seen = true; send("element_impression", { element_id: id, threshold: cfg.threshold }); }
      if (cfg.heartbeatMs > 0) {
        s.hb = setInterval(() => {
          const dur = now() - s.since;
          if (dur >= cfg.heartbeatMs) {
            send("element_still_visible", { element_id: id, visible_ms: Math.round(dur), threshold: cfg.threshold });
          }
        }, cfg.heartbeatMs);
      }
      send("element_visible", { element_id: id, threshold: cfg.threshold });
    }

    function onExit(id) {
      const s = state.get(id);
      if (!s || !s.visible) return;
      s.visible = false;
      const dur = now() - (s.since || now());
      s.acc += dur;
      if (s.hb) { clearInterval(s.hb); s.hb = null; }
      if (dur >= (s.cfg?.minVisibleMs || MIN_VISIBLE_MS_DEFAULT)) {
        send("element_hidden", { element_id: id, visible_ms: Math.round(dur), threshold: s.cfg?.threshold });
      }
      s.since = null;
    }

    // observer per threshold bucket
    const observers = new Map();
    function getObserver(threshold) {
      if (observers.has(threshold)) return observers.get(threshold);
      const obs = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          const id = entry.target.getAttribute("data-ga-id");
          if (!id) return;
          const cfg = getCfgFor(entry.target);
          if (entry.intersectionRatio >= cfg.threshold) onEnter(id, cfg);
          else onExit(id);
        });
      }, { threshold: [threshold] });
      observers.set(threshold, obs);
      return obs;
    }

    function observe(el) {
      const id = el.getAttribute("data-ga-id");
      if (!id) return;
      const cfg = getCfgFor(el);
      getObserver(cfg.threshold).observe(el);
    }

    function observeExisting(){ document.querySelectorAll("[data-ga-id]").forEach(observe); }

    const mo = new MutationObserver((muts) => {
      muts.forEach((m) => {
        m.addedNodes && m.addedNodes.forEach((n) => {
          if (n.nodeType === 1) {
            if (n.hasAttribute && n.hasAttribute("data-ga-id")) observe(n);
            n.querySelectorAll && n.querySelectorAll("[data-ga-id]").forEach(observe);
          }
        });
        m.removedNodes && m.removedNodes.forEach((n) => {
          if (n.nodeType === 1) {
            const id = n.getAttribute && n.getAttribute("data-ga-id");
            if (id) onExit(id);
          }
        });
      });
    });

    document.addEventListener("visibilitychange", () => {
      if (document.hidden) {
        for (const [id, s] of state.entries()) if (s.visible) onExit(id);
      }
    });

    window.addEventListener("beforeunload", () => {
      for (const [id, s] of state.entries()) if (s.visible) onExit(id);
    });

    // small public API
    window.GAVisibility = {
      init(){ observeExisting(); mo.observe(document.documentElement, { childList: true, subtree: true }); },
      markVisible(el){ observe(el); }
    };
  })();
