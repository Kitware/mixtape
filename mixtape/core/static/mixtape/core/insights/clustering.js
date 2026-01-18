document.addEventListener('alpine:init', () => {
  Alpine.data('clustering', (key, title) => ({
    data: [],
    layout: {
      title: {text: `Agent ${title}`},
      height: 450,
      xaxis: {
        gridcolor: Alpine.store('theme').gridColor,
        linecolor: Alpine.store('theme').gridColor,
        zeroline: false,
      },
      yaxis: {
        gridcolor: Alpine.store('theme').gridColor,
        linecolor: Alpine.store('theme').gridColor,
        zeroline: false,
      },
      paper_bgcolor: Alpine.store('theme').backgroundColor,
      plot_bgcolor: Alpine.store('theme').backgroundColor,
      font: {
        color: Alpine.store('theme').fontColor
      },
    },
    plot: null,
    init() {
      if (this.$store.insights.clustering && this.$store.insights.clustering[key]) {
        this.buildPlot();
      }
      this.$watch(
        () => this.$store.insights.clustering && this.$store.insights.clustering[key],
        (v) => { if (v) { this.buildPlot(); } }
      );
      Alpine.effect(() => {
        if (!this.plot && this.$store.insights.clustering && this.$store.insights.clustering[key]) {
          this.buildPlot();
        }
        if (this.plot) {
          const numAgents = this.$store.insights.uniqueAgents.length;
          const numEpisodes = this.$store.insights.episodeIds.length;
          const currentStep = this.$store.insights.currentStep;

          // Get the current plot data
          const plotElement = this.$refs.allManifolds;
          const currentData = plotElement.data;

          // Remove the highlight trace if it exists (it will be last, at numEpisodes * numAgents + 1)
          if (currentData && currentData.length > (numEpisodes * numAgents + 1)) {
            Plotly.deleteTraces(this.$refs.allManifolds, [numEpisodes * numAgents + 1]);
          }

          // Create arrays for the highlighted points
          const highlightX = [];
          const highlightY = [];
          const highlightText = [];

          for (let agentIdx = 0; agentIdx < numAgents; agentIdx++) {
            for (let episodeIdx = 0; episodeIdx < numEpisodes; episodeIdx++) {
              const idx = episodeIdx + agentIdx * numEpisodes;
              const agentTrace = currentData && currentData[idx];
              if (agentTrace
                  && Array.isArray(agentTrace.x)
                  && currentStep < agentTrace.x.length
                  && (!agentTrace.visible || agentTrace.visible === true)
              ) {
                highlightX.push(agentTrace.x[currentStep]);
                highlightY.push(agentTrace.y[currentStep]);
                const agents = this.$store.insights.uniqueAgents;
                const episodes = this.$store.insights.episodeIds;
                highlightText.push(`${agents[agentIdx]} Episode ${episodes[episodeIdx]}`);
              }
            }
          }

          // Only add the highlight trace if there are visible points
          if (highlightX.length > 0) {
            // Add new trace for highlighted points
            const textPosition = [
              'top left',
              'middle left',
              'bottom left',
              'top center',
              'middle center',
              'bottom center',
              'top right',
              'middle right',
              'bottom right'
            ];
            Plotly.addTraces(this.$refs.allManifolds, [{
              x: highlightX,
              y: highlightY,
              text: highlightText,
              type: 'scatter',
              mode: 'markers+text',
              showlegend: false,
              textposition: textPosition,
              textfont: {
                size: 12,
                color: this.$store.theme.fontColor
              },
              marker: {
                color: 'transparent',
                size: 20,
                line: {
                  width: 3,
                  color: 'red'
                }
              }
            }]);
          }
        }
      });
    },
    resizePlot: _.debounce(function() {
      if (!this.$refs.allManifolds.querySelector('.plotly')) return;
      Plotly.Plots.resize(this.$refs.allManifolds);
    }, 200, {leading: true}),
    buildPlot() {
      this.data = [];
      const agents = this.$store.insights.uniqueAgents;
      const episodes = this.$store.insights.episodeIds;
      let clustering = null;
      if (this.$store.insights.clustering) {
        clustering = this.$store.insights.clustering
      }
      if (!clustering || !clustering[key]) return;
      for (let agentIdx = 0; agentIdx < agents.length; agentIdx++) {
        clustering[key].episode_manifolds.forEach((episodeManifold, episodeIdx) => {
          const xCoords = [];
          const yCoords = [];
          episodeManifold.forEach(row => {
            const embedding = Array.isArray(row) ? row[agentIdx] : null;
            if (embedding && Array.isArray(embedding) && embedding.length >= 2) {
              xCoords.push(embedding[0]); yCoords.push(embedding[1]);
            }
          });
          this.data.push({
            x: xCoords,
            y: yCoords,
            type: 'scatter',
            mode: 'lines',
            colorscale: 'YlOrRd',
            line: { opacity: 1, width: 2 },
            name: `Episode ${episodes[episodeIdx]} ${agents[agentIdx]}`
          });
        });
      }
      this.data.push({
        x: clustering[key].all_manifolds_x,
        y: clustering[key].all_manifolds_y,
        type: 'scatter',
        mode: 'markers',
        marker: {
          color: clustering[key].all_clusters,
          cmin: 0,
          cmax: 9,
          colorscale: 'Viridis',
          size: 20,
          opacity: Array(clustering[key].all_manifolds_x.length).fill(1)
        },
        name: 'All Episodes'
      });
      this.plot = Plotly.newPlot(
        this.$refs.allManifolds,
        this.data,
        this.layout,
        { displayModeBar: false, responsive: true }
      );
      this.$watch('$store.settings.showPlotLegends', () => {
        this.$nextTick(() => {
          Plotly.relayout(this.$refs.allManifolds, {
            autosize: true,
            showlegend: this.$store.settings.showPlotLegends,
          })
        });
      });
    },
  }));
});
