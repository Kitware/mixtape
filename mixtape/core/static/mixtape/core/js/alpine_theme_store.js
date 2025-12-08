document.addEventListener('alpine:init', () => {
  Alpine.store('theme', {
    darkMode: window.matchMedia('(prefers-color-scheme: dark)').matches,
    darkbg: '#111827',
    lightbg: '#F1F2F6',
    lightline: '#4a5565',
    darkline: '#d1d5dc',
    get paper_bgcolor() { return this.darkMode ? this.darkbg : this.lightbg; },
    get plot_bgcolor() { return this.darkMode ? this.darkbg : this.lightbg; },
    lighttext: '#D1D5DB',
    darktext: '#000000',
    get font() {
      return {
        color: this.darkMode ? this.lighttext : this.darktext
      };
    },
    get axis() {
      return {
        gridcolor: this.darkMode ? this.lightline : this.darkline,
        zerolinecolor: this.darkMode ? this.lightbg : this.darkbg
      };
    }
  });
});
