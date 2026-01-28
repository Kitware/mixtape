(function(){
    if (!window.GA) return;

    const lastClick = new Map();     // id -> t_nav_ms
    const lastCheckbox = new Map();  // id -> {t, state}

    function sinceLast(map, key, now){
      const prev = map.get(key);
      map.set(key, now);
      if (!prev && prev !== 0) return null;
      return Math.max(0, now - prev);
    }

    window.GAButtons = {
      trackClick(selector, idOverride){
        document.addEventListener('click', (e) => {
          const el = e.target.closest(selector);
          if (!el) return;
          const id = idOverride || el.getAttribute('data-ga-id') || el.id || selector;
          const now = (performance && performance.now) ? Math.round(performance.now()) : null;
          const delta = sinceLast(lastClick, id, now ?? 0);
          window.GA.event('button_click', {
            button_id: id,
            since_last_click_ms: (delta === null ? null : Math.round(delta))
          });
        });
      },
      trackCheckbox(selector){
        document.addEventListener('change', (e) => {
          const el = e.target.closest(selector);
          if (!el || el.type !== 'checkbox') return;
          const id = el.getAttribute('data-ga-id') || el.id || selector;
          const now = (performance && performance.now) ? Math.round(performance.now()) : null;

          const prev = lastCheckbox.get(id);
          const delta = prev ? Math.max(0, (now ?? 0) - prev.t) : null;

          window.GA.event('checkbox_change', {
            checkbox_id: id,
            checked: !!el.checked,
            since_last_change_ms: (delta === null ? null : Math.round(delta))
          });
          lastCheckbox.set(id, { t: (now ?? 0), state: !!el.checked });
        });
      }
    };
  })();
