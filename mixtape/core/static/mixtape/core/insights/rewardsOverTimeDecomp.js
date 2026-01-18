document.addEventListener('alpine:init', () => {
  Alpine.data('rewardsOverTimeDecomp', () => ({
    data: [],
    layout: {
      title: {text: 'Rewards Over Time (Decomposed)'},
      xaxis: {
        title: {text: 'Time Step'},
        gridcolor: Alpine.store('theme').gridColor,
        linecolor: Alpine.store('theme').gridColor,
        zeroline: false,
      },
      yaxis: {
        title: {text: 'Cumulative Reward'},
        gridcolor: Alpine.store('theme').gridColor,
        linecolor: Alpine.store('theme').gridColor,
        zeroline: false,
      },
      autosize: true,
      showlegend: Alpine.store('settings').showPlotLegends,
      paper_bgcolor: Alpine.store('theme').backgroundColor,
      plot_bgcolor: Alpine.store('theme').backgroundColor,
      font: {
        color: Alpine.store('theme').fontColor
      },
      annotations: [],
      shapes: [],
    },
    config: {
      displayModeBar: false,
      responsive: true
    },
    plot: null,
    init() {
      this.initPlot();
      this.$watch('$store.settings.showPlotLegends', () => {
        this.$nextTick(() => {
          Plotly.relayout(
            this.$el,
            {
              autosize: true,
              showlegend: this.$store.settings.showPlotLegends
            })
        });
      });
      Alpine.effect(this.updateCurrentStep);
    },
    initPlot() {
      // Plot decomposed rewards for each episode
      this.$store.insights.decomposedRewards.forEach((episodeRewards, episodeIdx) => {
        Object.keys(episodeRewards).forEach((rewardType, rewardIdx) => {
          const rewardData = episodeRewards[rewardType];
          this.data.push({
            x: Array.from({length: rewardData.length}, (_, i) => i),
            y: rewardData,
            type: 'scatter',
            mode: 'lines',
            name: `Episode ${this.$store.insights.episodeIds[episodeIdx]} - ${rewardType}`,
          });
        });
      });

      // Calculate overall min and max Y values from all data
      const allYValues = this.$store.insights.decomposedRewards.flatMap(episodeRewards =>
        Object.values(episodeRewards).flatMap(rewardData => rewardData)
      );

      // Add current step indicator as a shape
      this.layout.shapes = this.createShapes(allYValues);

      // Add annotations for current step values
      this.layout.annotations = this.createAnnotations();

      this.plot = Plotly.newPlot(this.$el, this.data, this.layout, {displayModeBar: false});
    },
    resizePlot: _.debounce(function() {
      if (!this.$el.querySelector('.plotly')) return;
      Plotly.Plots.resize(this.$el);
    }, 200, {leading: true}),
    createShapes(allYValues) {
      const maxY = Math.max(...allYValues);
      const minY = Math.min(...allYValues);

      return [{
        type: 'line',
        x0: this.$store.insights.currentStep,
        x1: this.$store.insights.currentStep,
        y0: minY,
        y1: maxY,
        line: {
          color: 'red',
          width: 2,
          dash: 'dash'
        }
      }];
    },
    createAnnotations() {
      const episodeCount = Object.keys(this.$store.insights.decomposedRewards).length;
      const annotations = this.$store.insights.decomposedRewards.flatMap((episodeRewards, episodeIdx) => {
        const componentCount = Object.keys(episodeRewards).length;
        const offset = 40 / (componentCount * episodeCount);
        return Object.entries(episodeRewards).map(([rewardType, rewardData]) => {
          const currentValue = rewardData[this.$store.insights.currentStep] || rewardData[rewardData.length - 1];

          return {
            x: this.$store.insights.currentStep,
            y: currentValue,
            text: currentValue.toFixed(2),
            showarrow: true,
            arrowhead: 2,
            arrowsize: 1,
            arrowwidth: 2,
            ax: 20,
            ay: -20,
            font: {
              size: 10
            }
          };
        })
      });
      return annotations;
    },
    updateCurrentStep() {
      if (this.plot) {
        // Calculate overall min and max Y values from all data
        let allYValues = [];
        // Get all Y values from decomposed rewards
        this.$store.insights.decomposedRewards.forEach(episodeRewards => {
          Object.values(episodeRewards).forEach(rewardData => {
            allYValues = allYValues.concat(rewardData);
          });
        });

        Plotly.relayout(this.$el, {
          shapes: this.createShapes(allYValues),
          annotations: this.createAnnotations()
        });
      }
    }
  }));
});
