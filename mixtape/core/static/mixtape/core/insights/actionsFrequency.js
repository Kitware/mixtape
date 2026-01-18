document.addEventListener('alpine:init', () => {
  Alpine.data('actionsFrequency', () => ({
    data: [],
    layout: {
      title: {text: 'Actions VS Frequency'},
      autosize: true,
      barmode: Object.keys(Alpine.store('insights').actionVFrequency).length > 1 ? 'group' : 'stack',
      showlegend: Alpine.store('settings').showPlotLegends,
      paper_bgcolor: Alpine.store('theme').backgroundColor,
      plot_bgcolor: Alpine.store('theme').backgroundColor,
      font: {
        color: Alpine.store('theme').fontColor
      },
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
    },
    config: {
      displayModeBar: false,
    },
    plot: null,
    actionVFrequency: Object.keys(Alpine.store('insights').actionVFrequency).length > 1 ? Alpine.store('insights').actionVFrequency : Object.values(Alpine.store('insights').actionVFrequency)[0],
    init() {
      Object.entries(this.$store.insights.actionVFrequency).forEach(([grouping, values]) => {
        this.data.push({
          x: Object.keys(values),
          y: Object.values(values),
          type: 'bar',
          name: grouping,
          marker: {
            line: {
              width: 2
            }
          }
        });
      });
      this.plot = Plotly.react(this.$refs.actionsFrequency, this.data, this.layout, this.config);
      this.$watch('$store.settings.showPlotLegends', () => {
        this.$nextTick(() => {
          Plotly.relayout(
            this.$refs.actionsFrequency,
            {
              autosize: true,
              showlegend: this.$store.settings.showPlotLegends
            })
        });
      });
    },
    resizePlot: _.debounce(function () {
      if (!this.$refs.actionsFrequency.querySelector('.plotly')) return;
      Plotly.Plots.resize(this.$refs.actionsFrequency);
    }, 200, {leading: true}),
  }));
});
