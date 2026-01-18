document.addEventListener('alpine:init', () => {
  Alpine.data('rewardsOverTime', () => ({
    data: [],
    layout: {
      title: {text: 'Rewards Over Time'},
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
      annotations: [
        {
          x: Alpine.store('insights').currentStep,
          y: Alpine.store('insights').rewardsOverTime[0][Alpine.store('insights').currentStep],
          text: Alpine.store('insights').rewardsOverTime[0][Alpine.store('insights').currentStep].toFixed(2),
        }
      ],
    },
    config: {
      displayModeBar: false,
      responsive: true
    },
    plot: null,
    init() {
      this.$store.insights.rewardsOverTime.forEach((episode, idx) => {
        this.data.push({
          x: Array.from({length: episode.length}, (_, i) => i),
          y: episode,
          type: 'scatter',
          name: `Episode ${this.$store.insights.episodeIds[idx]}`,
          marker: {
            line: {
              width: 2
            }
          }
        });
      });
      this.data.push({
        x: [this.$store.insights.currentStep, this.$store.insights.currentStep],
        y: [0, Math.max(...this.$store.insights.rewardsOverTime.flat())],
        type: 'scatter',
        mode: 'lines',
        line: {
          color: 'red',
          width: 2,
          dash: 'dash'
        },
        name: 'Current Step'
      });
      this.plot = Plotly.newPlot(this.$refs.rewardsOverTime, this.data, this.layout, this.config);
      this.$watch('$store.settings.showPlotLegends', () => {
        this.$nextTick(() => {
          Plotly.relayout(
            this.$refs.rewardsOverTime,
            {
              autosize: true,
              showlegend: this.$store.settings.showPlotLegends
            })
        });
      });
      Alpine.effect(() => {
        if (this.plot) {
          Plotly.update(this.$refs.rewardsOverTime, {
            x: [[this.$store.insights.currentStep, this.$store.insights.currentStep]],
            y: [[Math.min(...this.$store.insights.rewardsOverTime.flat()), Math.max(...this.$store.insights.rewardsOverTime.flat())]]
          }, {
            annotations: this.$store.insights.rewardsOverTime.map((episode, idx) => {
              return {
                x: this.getStep(episode),
                y: episode[this.getStep(episode)],
                text: episode[this.getStep(episode)].toFixed(2),
              }
            })
          }, [this.$store.insights.rewardsOverTime.length]);
        }
      });
    },
    resizePlot: _.debounce(function() {
      if (!this.$refs.rewardsOverTime.querySelector('.plotly')) return;
      Plotly.Plots.resize(this.$refs.rewardsOverTime);
    }, 200, {leading: true}),
    getStep(episode) {
      if (this.$store.insights.currentStep < Object.keys(episode).length) {
        return this.$store.insights.currentStep;
      }
      return Object.keys(episode).length - 1;
    },


  }));
});
