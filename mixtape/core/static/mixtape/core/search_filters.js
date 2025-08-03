document.addEventListener('alpine:init', () => {
  Alpine.data('homePageData', () => ({
    selectedEpisodes: [],
    allowedEnvironment: null,
    selectedAlgorithms: [],
    selectedEnvironments: [],
    searchQuery: '',
    sortOrder: 'oldest',
    sortOptions: ['newest', 'oldest', 'environment', 'algorithm'],
    algoOpen: false,
    envOpen: false,
    sortOpen: false,
    multiSelectMode: true,

    toggleMultiSelect() {
      this.multiSelectMode = !this.multiSelectMode;
      if (!this.multiSelectMode) {
        this.selectedEpisodes = [];
        this.allowedEnvironment = null;
      }
    },

    updateAllowedEnvironment(env) {
      if (this.selectedEpisodes.length === 0) {
        this.allowedEnvironment = null;
      } else {
        this.allowedEnvironment = env;
      }
    },

    viewSelectedEpisodes() {
      const insightsUrl = this.$root.dataset.insightsUrl;
      window.location.href =
        insightsUrl + '?episode_id=' + this.selectedEpisodes.join('&episode_id=');
    },

    matchesEnvironment(environment) {
      if (!this.selectedEnvironments.length) return true;
      return this.selectedEnvironments.includes(environment);
    },

    matchesAlgorithm(algorithm) {
      if (!this.selectedAlgorithms.length) return true;
      return this.selectedAlgorithms.includes(algorithm);
    },

    matchesSearch(environment, algorithm) {
      if (!this.searchQuery.trim()) return true;
      const query = this.searchQuery.toLowerCase();
      return (
        environment.toLowerCase().includes(query) ||
        algorithm.toLowerCase().includes(query)
      );
    },

    toggleSort() {
      const currentIndex = this.sortOptions.indexOf(this.sortOrder);
      const nextIndex = (currentIndex + 1) % this.sortOptions.length;
      this.sortOrder = this.sortOptions[nextIndex];
      this.applySorting();
    },

    getSortLabel() {
      const labels = {
        'newest': 'Newest First',
        'oldest': 'Oldest First',
        'environment': 'Environment A-Z',
        'algorithm': 'Algorithm A-Z',
      };
      return labels[this.sortOrder];
    },

    applySorting() {
      const container = document.getElementById('episodes-list');
      if (!container) return;
      const episodes = Array.from(
        container.querySelectorAll('li[data-episode-id]')
      );

      episodes.sort((a, b) => {
        const aId = parseInt(a.dataset.episodeId || '0', 10) || 0;
        const bId = parseInt(b.dataset.episodeId || '0', 10) || 0;
        const aEnv = a.dataset.environment || '';
        const bEnv = b.dataset.environment || '';
        const aAlg = a.dataset.algorithm || '';
        const bAlg = b.dataset.algorithm || '';

        switch (this.sortOrder) {
          case 'newest':
            return bId - aId;
          case 'oldest':
            return aId - bId;
          case 'environment':
            return aEnv.localeCompare(bEnv);
          case 'algorithm':
            return aAlg.localeCompare(bAlg);
          default:
            return 0;
        }
      });

      episodes.forEach((episode) => container.appendChild(episode));
    },
  }));
});
