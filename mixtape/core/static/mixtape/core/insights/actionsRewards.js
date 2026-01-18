document.addEventListener('alpine:init', () => {
  Alpine.data('actionsRewards', () => ({
    data: [],
    layout: {
      title: {text: 'Actions VS Reward Frequency'},
      xaxis: {
        title: {text: 'Action'},
        gridcolor: Alpine.store('theme').gridColor,
        linecolor: Alpine.store('theme').gridColor,
        zeroline: false,
      },
      yaxis: {
        title: {text: 'Reward Frequency'},
        gridcolor: Alpine.store('theme').gridColor,
        linecolor: Alpine.store('theme').gridColor,
        zeroline: false,
      },
      autosize: true,
      barmode: Object.keys(Alpine.store('insights').actionVReward).length > 1 ? 'group' : 'stack',
      showlegend: Alpine.store('settings').showPlotLegends,
      paper_bgcolor: Alpine.store('theme').backgroundColor,
      plot_bgcolor: Alpine.store('theme').backgroundColor,
      font: {
        color: Alpine.store('theme').fontColor
      },
    },
    config: {
      displayModeBar: false,
    },
    plot: null,
    actionVReward: Object.keys(Alpine.store('insights').actionVReward).length > 1 ? Alpine.store('insights').actionVReward : Object.values(Alpine.store('insights').actionVReward)[0],
    init() {
      this.data = Object.entries(this.$store.insights.actionVReward).map(([grouping, values]) => ({
        x: Object.keys(values),
        y: Object.values(values),
        type: 'bar',
        name: grouping,
        marker: {
          line: {
            width: 2
          }
        }
      }));
      this.plot = Plotly.react(this.$refs.actionsRewards, this.data, this.layout, this.config);
      this.$watch('$store.settings.showPlotLegends', () => {
        this.$nextTick(() => {
          Plotly.relayout(
            this.$refs.actionsRewards, {
              autosize: true,
              showlegend: this.$store.settings.showPlotLegends
            })
        });
      });
    },
    resizePlot: _.debounce(function() {
      if (!this.$refs.actionsRewards.querySelector('.plotly')) return;
      Plotly.Plots.resize(this.$refs.actionsRewards);
    }, 200, {leading: true}),
  }));
});
