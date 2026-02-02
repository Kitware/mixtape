document.addEventListener('alpine:init', () => {
  Alpine.data('replay', () => ({
    playing: false,
    stepInput: 1,
    maxSteps: Alpine.store('insights').parsedData.max_steps - 1,
    init() {
      this.stepInput = this.$store.insights.currentStep + 1;
      this.$watch('$store.insights.currentStep', (val) => {
        this.stepInput = val + 1;
      });
    },
    stepForward() {
      if (!this.playing) {
        return;
      }
      if (this.$store.insights.currentStep < this.maxSteps) {
        const delay = Math.max(50, Math.round(500 / (this.$store.settings.playbackSpeed || 1)));
        setTimeout(() => {
          this.$store.insights.currentStep++;
          this.stepForward();
        }, delay);
      } else {
        this.playing = false;
      }
    },
    jumpToStep() {
      const val = Math.max(1, Math.min(this.maxSteps + 1, this.stepInput || 1));
      this.stepInput = val;
      this.$store.insights.currentStep = val - 1;
    },
  }));
});
