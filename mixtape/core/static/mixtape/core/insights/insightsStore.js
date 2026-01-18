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
    timelineKeyStepsGlobal: [],

    // User-configurable flags
    episodeSummaries: [],
    // Runtime values
    currentStep: 0,

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
      this.decomposedRewards = this.parsedData?.decomposed_rewards ?? [];
      this.actionVReward = this.parsedData?.action_v_reward ?? {};
      this.rewardHistogram = this.parsedData?.reward_histogram ?? [];
      this.actionVFrequency = this.parsedData?.action_v_frequency ?? {};
      this.stepData = this.parsedData?.step_data ?? [];
      this.timelineKeySteps = this.parsedData?.timeline_key_steps ?? [];
      this.timelineKeyStepsGlobal = this.parsedData?.timeline_key_steps_global ?? [];
    },
  });
});
