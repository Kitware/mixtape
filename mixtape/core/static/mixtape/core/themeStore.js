const colorSchemeQuery = window.matchMedia('(prefers-color-scheme: dark)');
colorSchemeQuery.addEventListener('change', (e) => {
  syncDaisyTheme();
  Alpine.store('theme').syncFromCssProperties();
});

function syncDaisyTheme() {
  const darkMode = colorSchemeQuery.matches;
  document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light');
}
syncDaisyTheme();

document.addEventListener('alpine:init', () => {
  Alpine.store('theme', {
    backgroundColor: null,
    fontColor: null,
    gridColor: null,
    init() {
      this.syncFromCssProperties();
    },
    syncFromCssProperties() {
      this.backgroundColor = this.cssPropertyAsRgb('--color-base-100');
      this.fontColor = this.cssPropertyAsRgb('--color-base-content');
      this.gridColor = this.cssPropertyAsRgb('--color-base-300');
    },
    cssPropertyAsRgb(propertyName) {
      return new Color(window.getComputedStyle(document.body).getPropertyValue(propertyName)).to('srgb').toString();
    }
  });
});
