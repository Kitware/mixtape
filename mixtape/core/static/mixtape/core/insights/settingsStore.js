document.addEventListener('alpine:init', () => {
  Alpine.store('settings', {
    // Info toggles
    showPlayback: true,
    showTimeline: true,
    showEpisodeInfo: true,
    showMediaViewer: true,
    // Plot toggles
    showPlotRewardFrequency: true,
    showPlotActionFrequency: true,
    showPlotActionRewardFrequency: true,
    showPlotRewardsTime: true,
    showPlotObservationClustering: true,
    showPlotActionClustering: true,
    // Timeline settings
    playbackSpeed: 1.0,
    useGlobalTimeline: true,
    timelineEpisodeIdx: 0,
    // Plot settings
    showPlotLegends: true,
  });
});
