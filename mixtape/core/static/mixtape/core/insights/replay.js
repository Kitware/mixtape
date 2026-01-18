document.addEventListener('alpine:init', () => {
  Alpine.data('replay', () => ({
    playing: false,
    maxSteps: Alpine.store('insights').parsedData.max_steps - 1,
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
        this.$store.insights.currentStep = 0;
      }
    },
  }));
});
