document.addEventListener('alpine:init', () => {
  Alpine.data('clusteringTimeline', (key, title) => ({
    data: [],
    layout: {
      title: {text: `Agent ${title} Timeline`},
      height: 300,
      paper_bgcolor: Alpine.store('theme').backgroundColor,
      plot_bgcolor: Alpine.store('theme').backgroundColor,
      font: {
        color: Alpine.store('theme').fontColor
      },
    },
    plot: null,
    init() {
      if (this.$store.insights && this.$store.insights.clustering && this.$store.insights.clustering[key]) {
        this.buildPlot();
      }
      this.$watch(
        '$store.settings.timelineEpisodeIdx',
        () => {
          if (this.$store.insights && this.$store.insights.clustering && this.$store.insights.clustering[key]) {
            this.buildPlot();
          }
        }
      );
      this.$watch(
        () => this.$store.insights && this.$store.insights.clustering && this.$store.insights.clustering[key],
        (v) => {
          if (v) {
            this.buildPlot();
          }
        }
      );
      Alpine.effect(() => {
        if (
          !this.plot
          && this.$store.insights
          && this.$store.insights.clustering && this.$store.insights.clustering[key]
        ) { this.buildPlot(); }
        if (this.plot) {
          const plotElement = this.$el;
          const traces = plotElement && plotElement.data ? plotElement.data : null;
          if (!traces || traces.length < 2) return;
          let clustering = null;
          if (this.$store.insights && this.$store.insights.clustering) {
            clustering = this.$store.insights.clustering;
          }
          if (!clustering) return;
          const epClusters = clustering[key]?.episode_clusters;
          if (!epClusters) return;
          const epIdx = this.$store.settings.timelineEpisodeIdx;
          const zNow = Array.isArray(epClusters?.[0]?.[0])
            ? (epClusters?.[epIdx] || epClusters?.[0] || [])
            : epClusters;
          if (!Array.isArray(zNow) || zNow.length === 0) { return; }
          const agentsNow = zNow.length;
          const yMin = 0;
          const yMax = Math.max(0, (agentsNow || 1) - 1);
          const step = this.$store.insights.currentStep;
          const xLine = [step, step];
          const yLine = [yMin - 0.5, yMax + 0.5];
          // Use restyle: arguments must be arrays-of-arrays per selected trace
          Plotly.restyle(this.$el, { x: [xLine], y: [yLine] }, [1]);
        }
      });
    },
    resizePlot: _.debounce(function() {
      if (!this.$el.querySelector('.plotly')) return;
      Plotly.Plots.resize(this.$el);
    }, 200, {leading: true}),
    buildPlot() {
      let clustering = null;
      if (this.$store.insights && this.$store.insights.clustering) {
        clustering = this.$store.insights.clustering;
      }
      if (!clustering || !clustering[key]) return;
      const epClusters = clustering[key].episode_clusters;
      const epIdx = this.$store.settings.timelineEpisodeIdx;
      // If multi-episode shape [episodes][agents][steps], pick selected episode; else assume [agents][steps]
      let agentStepClusters = Array.isArray(epClusters?.[0]?.[0])
        ? (epClusters?.[epIdx] || epClusters?.[0] || [])
        : epClusters;
      // Pad each agent row to equal length [agents][steps]
      const maxLen = Array.isArray(agentStepClusters)
        ? Math.max(
            ...agentStepClusters.map(r => (Array.isArray(r) ? r.length : 0))
          )
        : 0;
      agentStepClusters = Array.isArray(agentStepClusters)
        ? agentStepClusters.map(r => {
            const row = Array.isArray(r) ? r.slice() : [];
            while (row.length < maxLen) row.push(null);
            return row;
          })
        : agentStepClusters;
      const currentStepVal = this.$store.insights.currentStep;
      const yVal = Array.isArray(agentStepClusters)
        ? agentStepClusters.length - 1
        : 1;
      this.data = [
        {
          agentStepClusters: agentStepClusters,
          type: 'heatmap',
          showscale: false,
          colorscale: 'YlGnBu',
        },
        {
          x: [currentStepVal, currentStepVal],
          y: [0, yVal],
          type: 'scatter',
          mode: 'lines',
          showlegend: false,
          line: { color: 'red', width: 4, opacity: 0.5 },
        }
      ];
      this.plot = Plotly.newPlot(
        this.$el,
        this.data,
        this.layout,
        {
          displayModeBar: false,
          responsive: true
        }
      );
      this.$watch('$store.settings.showPlotLegends', () => {
        this.$nextTick(() => {
          Plotly.relayout(
            this.$el,
            {
              autosize: true,
              showlegend: this.$store.settings.showPlotLegends
            }
          );
        });
      });
    },

  }));
});
