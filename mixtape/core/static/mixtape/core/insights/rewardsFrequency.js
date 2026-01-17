document.addEventListener('alpine:init', () => {
  Alpine.data('rewardsFrequency', () => ({
    data: [],
    layout: {
      title: {text: 'Rewards VS Frequency'},
      xaxis: {
        title: {text: 'Reward'},
        gridcolor: Alpine.store('theme').axis.gridcolor,
        zerolinecolor: Alpine.store('theme').axis.zerolinecolor
      },
      yaxis: {
        title: {text: 'Frequency'},
        gridcolor: Alpine.store('theme').axis.gridcolor,
        zerolinecolor: Alpine.store('theme').axis.zerolinecolor
      },
      autosize: true,
      barmode: 'overlay',
      showlegend: Alpine.store('settings').showPlotLegends,
      paper_bgcolor: Alpine.store('theme').paper_bgcolor,
      plot_bgcolor: Alpine.store('theme').plot_bgcolor,
      font: Alpine.store('theme').font,
    },
    config: {
      displayModeBar: false,
    },
    plot: null,
    init() {
      this.rewardHistogram.forEach((episode, idx) => {
        this.data.push({
          x: episode,
          type: 'histogram',
          name: `Episode ${this.episodeIds[idx]}`,
          marker: {
            line: {
              width: 2
            }
          }
        });
      });
      this.plot = Plotly.newPlot(this.$refs.rewardsFrequency, this.data, this.layout, this.config);
      this.$watch('$store.settings.showPlotLegends', () => {
        this.$nextTick(() => {
          Plotly.relayout(
            this.$refs.rewardsFrequency,
            {
              autosize: true,
              showlegend: this.$store.settings.showPlotLegends
            })
        });
      });
    },
    resizePlot: _.debounce(function() {
      if (!this.$refs.rewardsFrequency.querySelector('.plotly')) return;
      Plotly.Plots.resize(this.$refs.rewardsFrequency);
    }, 200, {leading: true}),
  }));
});
