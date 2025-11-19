(() => {
  const factory = () => ({
    settingsOpen: false,
    collapsed: false,
    init() {
      try {
        if ($store.insights && typeof $store.insights.sidePanelCollapsed !== 'undefined') {
          this.collapsed = !!$store.insights.sidePanelCollapsed;
        }
      } catch (_) {}
    },
    toggleSettings() {
      this.settingsOpen = !this.settingsOpen;
    },
    toggleCollapsed() {
      this.collapsed = !this.collapsed;
      try {
        if ($store.insights) {
          $store.insights.sidePanelCollapsed = this.collapsed;
        } else {
          Alpine.store('insights', { sidePanelCollapsed: this.collapsed });
        }
      } catch (_) {}
    },
    updatePlaybackOnChange($event) {
      const raw = Number($event.target.value);
      const v = Math.min(10, Math.max(0.1, isNaN(raw) ? 1 : raw));
      const rounded = Math.round(v * 10) / 10;
      if ($store.insights) {
        $store.insights.playbackSpeed = rounded;
      }
      $event.target.value = rounded.toFixed(1);
    },
    updatePlaybackOnInput($event) {
      const raw = Number($event.target.value);
      if (!isNaN(raw) && $store.insights) {
        const v = Math.min(10, Math.max(0.1, raw));
        const rounded = Math.round(v * 10) / 10;
        $store.insights.playbackSpeed = rounded;
      }
    },
    setGlobalTimeline(v) {
      if ($store.insights) {
        $store.insights.useGlobalTimeline = v;
      }
    },
    setTimelineEpisodeIdx($event) {
      if ($store.insights) {
        $store.insights.timelineEpisodeIdx = +$event.target.value;
      }
    },
  });
  const register = () => {
    Alpine.data('insightsSidePanel', factory);
    try { window.insightsSidePanel = factory; } catch (_) {}
  };
  if (window.Alpine) register();
  else window.addEventListener('alpine:init', register);
})();
