document.addEventListener('alpine:init', () => {
  Alpine.store('insights', {
    // Immutable data
    parsedData: null,
    clustering: null,
    multiKey: '',
    clusteringAvailable: false,

    episodeIds: [],
    uniqueAgents: [],
    rewardsOverTime: [],
    decomposedRewards: [],
    actionVReward: {},
    rewardHistogram: [],
    actionVFrequency: {},
    stepData: [],
    timelineKeySteps: [],

    // User-configurable flags
    episodeSummaries: [],
    // Runtime values
    currentStep: 0,

    // Get active timeline key steps based on global/episode mode
    getActiveTimelineKeySteps() {
      const settings = Alpine.store('settings');
      if (!settings) return [];
      const useGlobal = settings.useGlobalTimeline;
      const epIdx = settings.timelineEpisodeIdx;

      if (useGlobal && Array.isArray(this.timelineKeySteps)) {
        // Global mode: merge all episodes' key steps, sort by reward, take top 40
        const allSteps = [];
        this.timelineKeySteps.forEach((episodeSteps, episodeIdx) => {
          if (Array.isArray(episodeSteps)) {
            episodeSteps.forEach(step => {
              allSteps.push({
                ...step,
                episodeIdx: episodeIdx,
                episodeId: this.episodeIds?.[episodeIdx] || episodeIdx,
              });
            });
          }
        });
        // Sort by total_rewards descending, take top 40, then sort by step number
        const topSteps = allSteps
          .sort((a, b) => b.total_rewards - a.total_rewards)
          .slice(0, 40)
          .sort((a, b) => {
            // Sort by episode first, then by step number within episode
            if (a.episodeIdx !== b.episodeIdx) return a.episodeIdx - b.episodeIdx;
            return a.number - b.number;
          });
        return topSteps;
      } else {
        // Episode mode: return selected episode's key steps
        return this.timelineKeySteps?.[epIdx] || [];
      }
    },

    init() {
      // Parse JSON data
      this.parsedData = JSON.parse(
        document.getElementById('parsed_data_json').textContent
      );
      this.clustering = JSON.parse(
        document.getElementById('clustering_json').textContent
      );
      this.multiKey = JSON.parse(
        document.getElementById('clustering_multi_key').textContent
      );
      this.clusteringAvailable = JSON.parse(
        document.getElementById('clustering_available_json').textContent
      );

      this.episodeIds = this.parsedData.episode_ids;
      this.uniqueAgents = this.parsedData.unique_agents;
      this.rewardsOverTime = this.parsedData.rewards_over_time;
      this.decomposedRewards = this.parsedData.decomposed_rewards;
      this.actionVReward = this.parsedData.action_v_reward;
      this.rewardHistogram = this.parsedData.reward_histogram;
      this.actionVFrequency = this.parsedData.action_v_frequency;
      this.stepData = this.parsedData.step_data;
      this.timelineKeySteps = this.parsedData.timeline_key_steps;
    },
  });
});
