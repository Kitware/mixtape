document.addEventListener('alpine:init', () => {
  Alpine.data('insightsTabs', () => ({
    // initial state
    active: 'Overview',
    parsedData: null,
    clustering: null,
    episodesCount: null,
    agentsCount: null,
    maxStepsCount: null,
    avgRewardDisplay: null,
    actionVReward: {},
    rewardHistogram: [],
    actionVFrequency: {},
    rewardsOverTime: [],
    decomposedRewards: [],
    uniqueAgents: [],
    episodeIds: [],
    stepData: [],
    timelineKeySteps: [],
    timelineKeyStepsGlobal: [],
    currentStep: 0,
    maxSteps: 0,
    maxStepsGlobal: 0,
    episodeSummaries: [],
    mediaEpisodeIdx: 0,
    linkMediaToEpisode: true,
    limitToEpisode: false,
    debug: false,
    playing: false,
    multiKey: '',
    clusteringAvailable: false,
    pollTimer: null,
    showToast: false,
    toastText: '',
    toastTimer: null,
    pollStartTs: 0,
    pollTimeoutMs: 120000,
    pollAttempts: 0,
    pollMaxAttempts: 0,
    pollVisibilityHandler: null,
    pollPageHideHandler: null,
    pollBeforeUnloadHandler: null,

    init() {
      // Parse JSON data
      this.parsedData = JSON.parse(
        document.getElementById('parsed_data_json')?.textContent || 'null'
      );
      this.clustering = JSON.parse(
        document.getElementById('clustering_json')?.textContent || 'null'
      );
      this.multiKey = JSON.parse(
        document.getElementById('clustering_multi_key')?.textContent || '""'
      );
      this.clusteringAvailable = JSON.parse(
        document.getElementById('clustering_available_json')?.textContent || 'false'
      );
      this.episodeIds = this.parsedData?.episode_ids ?? [];
      this.uniqueAgents = this.parsedData?.unique_agents ?? [];
      this.rewardsOverTime = this.parsedData?.rewards_over_time ?? [];
      this.decomposedRewards = this.parsedData?.decomposed_rewards ?? [];
      this.actionVReward = this.parsedData?.action_v_reward ?? {};
      this.rewardHistogram = this.parsedData?.reward_histogram ?? [];
      this.actionVFrequency = this.parsedData?.action_v_frequency ?? {};
      this.stepData = this.parsedData?.step_data ?? [];
      this.timelineKeySteps = this.parsedData?.timeline_key_steps ?? [];
      this.timelineKeyStepsGlobal = this.parsedData?.timeline_key_steps_global ?? [];
      this.episodesCount = Array.isArray(this.episodeIds) ? this.episodeIds.length : null;
      this.agentsCount = Array.isArray(this.uniqueAgents) ? this.uniqueAgents.length : null;
      this.maxStepsGlobal = (this.parsedData?.max_steps ?? 1) - 1;
      this.maxSteps = this.maxStepsGlobal;

      // Average reward
      const avgReward = this.computeAverageReward(this.parsedData);
      this.avgRewardDisplay = (avgReward ?? null) !== null ? Number(avgReward).toFixed(2) : null;

      // Episode summary
      this.episodeSummaries = (Array.isArray(this.episodeIds) ? this.episodeIds : [])
        .map((id, idx) => this.summarizeEpisode(idx, id));

      // Max steps count (for multi-episode mode)
      this.maxStepsCount = (() => {
        const nums = Array.isArray(this.episodeSummaries)
          ? this.episodeSummaries.map(e => (typeof e?.steps === 'number' ? e.steps : 0))
          : [];
        const m = nums.length ? Math.max(...nums) : 0;
        return m || null;
      })();

      // Build watchers
      this.updateMaxSteps();
      this.$watch('limitToEpisode', () => this.updateMaxSteps());
      this.$watch('$store.settings.timelineEpisodeIdx', () => {
        this.updateMaxSteps();
        if (this.linkMediaToEpisode) this.mediaEpisodeIdx = this.$store.settings.timelineEpisodeIdx;
      });
      this.$watch('episodeSummaries', () => {
        if (!Array.isArray(this.episodeSummaries) || this.episodeSummaries.length === 0) {
          this.mediaEpisodeIdx = 0;
        } else if (this.mediaEpisodeIdx >= this.episodeSummaries.length) {
          this.mediaEpisodeIdx = 0;
        }
      });
      this.$watch('linkMediaToEpisode', (v) => { if (v) { this.mediaEpisodeIdx = this.$store.settings.timelineEpisodeIdx; } });
      // Ensure timeline mode changes recompute limits
      this.$watch('$store.settings.useGlobalTimeline', this.updateMaxSteps);

      // Store
      const store = (window.Alpine && Alpine.store('insights')) || null;
      if (store) {
        // Push derived values
        store.currentStep = this.currentStep;
        store.episodeSummaries = this.episodeSummaries;
        store.multiKey = this.multiKey;
        store.clusteringAvailable = this.clusteringAvailable;
        store.clustering = this.clustering;
      }
      // push changes
      this.$watch('episodeSummaries', v => { Alpine.store('insights').episodeSummaries = v; });
      this.$watch('currentStep', v => { Alpine.store('insights').currentStep = v; });
      this.$watch('clustering', (v) => {
        const store = (window.Alpine && Alpine.store('insights')) || null;
        if (store) store.clustering = v;
      });

      // Polling: multi-episode artifact or single-episode partials
      const isMulti = Array.isArray(this.episodeIds) && this.episodeIds.length > 1;
      const isSingle = Array.isArray(this.episodeIds) && this.episodeIds.length === 1;
      if ((isMulti && !this.clustering && this.multiKey)
          || (isSingle && (!this.clustering || !this.clustering?.obs || !this.clustering?.agent_outs))) {
        this.startClusteringPolling();
      }
    },

    stepForward() {
      if (!this.playing) return;
      if (this.currentStep < this.maxSteps) {
        const delay = Math.max(50, Math.round(500 / (this.$store.settings.playbackSpeed || 1)));
        setTimeout(() => { this.currentStep++; this.stepForward(); }, delay);
      } else { this.playing = false; this.currentStep = 0; }
    },
    updateMaxSteps() {
      if (this.limitToEpisode) {
        const steps = this.episodeSummaries[this.$store.settings.timelineEpisodeIdx]?.steps ?? null;
        this.maxSteps = (typeof steps === 'number' && steps > 0) ? steps - 1 : this.maxStepsGlobal;
      } else { this.maxSteps = this.maxStepsGlobal; }
    },
    selectTab(tab) {
      this.active = tab;
    },
    overviewVisible() {
      return (
        this.$store.settings.showPlotRewardFrequency ||
        this.$store.settings.showPlotActionFrequency ||
        this.$store.settings.showPlotActionRewardFrequency
      );
    },
    overviewGridColsClass() {
      const c = (this.$store.settings.showPlotRewardFrequency ? 1 : 0)
        + (this.$store.settings.showPlotActionFrequency ? 1 : 0)
        + (this.$store.settings.showPlotActionRewardFrequency ? 1 : 0);
      return c >= 3 ? 'lg:grid-cols-3' : (c === 2 ? 'lg:grid-cols-2' : 'lg:grid-cols-1');
    },
    maxStepsTooltip() {
      const numEpSummaries = Array.isArray(this.episodeSummaries) ? this.episodeSummaries.length : 0;
      if (numEpSummaries <= 1) {
        const steps = this.episodeSummaries?.[0]?.steps ?? null;
        return `Total Steps: ${steps ?? '—'}`;
      }
      const lines = [`Max Steps: ${this.maxStepsCount ?? '—'}`];
      for (let i = 0; i < numEpSummaries; i++) {
        const steps = this.episodeSummaries?.[i]?.steps ?? '—';
        lines.push(`Ep ${i + 1}: ${steps}`);
      }
      return lines.join('\n');
    },
    timelineTitle(ts) {
      if (Array.isArray(ts?.episodes)) {
        const parts = ts.episodes.map(e => `Episode ${e.episode_id}: ${Number(e.total_rewards ?? 0).toFixed(2)}`);
        return `Step ${ts.number}\n${parts.join('\n')}`;
      }
      return `Step ${ts?.number} — Reward: ${Number((ts && ts.total_rewards) ?? 0).toFixed(2)}`;
    },
    timelineAria(ts) {
      if (Array.isArray(ts?.episodes)) {
        const parts = ts.episodes.map(e => `Episode ${e.episode_id}: ${Number(e.total_rewards ?? 0).toFixed(2)}`);
        return `Step ${ts.number} ${parts.join(', ')}`;
      }
      return `Step ${ts?.number} Reward ${Number((ts && ts.total_rewards) ?? 0).toFixed(2)}`;
    },

    // helpers for templates
    getEpisodeStepIndex(epIdx) {
      const steps = this.stepData?.[epIdx] || {};
      const keys = Object.keys(steps);
      if (!keys.length) return null;
      const nums = keys.map(k => +k).sort((a,b) => a - b);
      let chosen = nums[0];
      for (let i = 0; i < nums.length; i++) { if (nums[i] <= this.currentStep) chosen = nums[i]; else break; }
      return chosen;
    },
    getEpisodeStepData(epIdx) {
      const idx = this.getEpisodeStepIndex(epIdx);
      if (idx === null) return null;
      const steps = this.stepData?.[epIdx] || {};
      return steps[idx] ?? steps[String(idx)] ?? null;
    },
    // Compatibility alias used by some templates
    getStep(step) {
      return this.getEpisodeStepData(this.$store.settings.timelineEpisodeIdx);
    },
    getAgentStep(epIdx, agent) {
      const epStepData = this.getEpisodeStepData(epIdx);
      const agentSteps = epStepData && Array.isArray(epStepData.agent_steps) ? epStepData.agent_steps : [];
      return agentSteps.find(agentStep => agentStep.agent === agent) || null;
    },
    getAgentAction(epIdx, agent) {
      const agentStep = this.getAgentStep(epIdx, agent);
      return (agentStep && (agentStep.action ?? agentStep.action_string ?? agentStep.action_name ?? agentStep.action_id)) ?? '—';
    },
    getAgentRewardStr(epIdx, agent) {
      const agentStep = this.getAgentStep(epIdx, agent);
      const totalReward = agentStep ? (agentStep.total_reward ?? agentStep.reward ?? null) : null;
      return (typeof totalReward === 'number') ? totalReward.toFixed(3) : (totalReward ?? '—');
    },
    getEpisodeImageUrl(epIdx) {
      const epStepData = this.getEpisodeStepData(epIdx);
      return epStepData && epStepData.image_url ? epStepData.image_url : null;
    },
    get episodePairs() {
      const pairs = [];
      const numEpSummaries = Array.isArray(this.episodeSummaries) ? this.episodeSummaries.length : 0;
      for (let i = 0; i < numEpSummaries; i++) {
        pairs.push({ epIdx: i, kind: 'action' }); pairs.push({ epIdx: i, kind: 'reward' });
      }
      return pairs;
    },
    totalFromSeries(yTrace) {
      const rewards = Array.isArray(yTrace) ? yTrace.filter(reward => typeof reward === 'number') : [];
      if (!rewards.length) return null;
      const last = rewards[rewards.length - 1];
      return (typeof last === 'number' && rewards.some(reward => reward < last))
        ? last
        : rewards.reduce((a, b) => a + b, 0);
    },
    computeAverageReward(parsedData) {
      if (!parsedData) return null;
      if (parsedData.rewards_totals && typeof parsedData.rewards_totals.average_per_episode === 'number') {
        return parsedData.rewards_totals.average_per_episode;
      }
      if (Array.isArray(parsedData.episode_total_rewards)) {
        const rewards = parsedData.episode_total_rewards.filter(reward => typeof reward === 'number');
        return rewards.length ? rewards.reduce((a,b)=>a+b,0) / rewards.length : null;
      }
      if (Array.isArray(parsedData.episode_rewards)) {
        const rewards = parsedData.episode_rewards.filter(reward => typeof reward === 'number');
        return rewards.length ? rewards.reduce((a,b)=>a+b,0) / rewards.length : null;
      }
      if (Array.isArray(parsedData.rewards_over_time)) {
        const perEpTotals = parsedData.rewards_over_time.map(trace => {
          if (!trace) return null;
          const yTrace = Array.isArray(trace.y) ? trace.y : [];
          return this.totalFromSeries(yTrace);
        }).filter(reward => typeof reward === 'number');
        return perEpTotals.length ? perEpTotals.reduce((a,b)=>a+b,0) / perEpTotals.length : null;
      }
      return null;
    },
    summarizeEpisode(idx, id) {
      const allRewards = Array.isArray(this.rewardsOverTime[idx]) ? this.rewardsOverTime[idx] : [];
      const steps = allRewards.length || null;
      const perStep = allRewards.filter(rewards => typeof rewards === 'number');
      let totalReward = null;
      if (Array.isArray(this.parsedData?.episode_total_rewards)) {
        totalReward = typeof this.parsedData.episode_total_rewards[idx] === 'number' ? this.parsedData.episode_total_rewards[idx] : null;
      }
      if (totalReward === null && allRewards.length) {
        totalReward = this.totalFromSeries(allRewards);
      }
      const minPerStep = perStep.length ? Math.min(...perStep) : null;
      const maxPerStep = perStep.length ? Math.max(...perStep) : null;
      const avgPerStep = perStep.length ? (perStep.reduce((a,b)=> a + b, 0) / perStep.length) : null;
      return { id: id ?? idx, steps, totalReward, minPerStep, maxPerStep, avgPerStep };
    },
    startClusteringPolling() {
      if (this.pollTimer) return;
      this.pollStartTs = Date.now();
      this.pollAttempts = 0;
      this.pollMaxAttempts = Math.ceil((this.pollTimeoutMs || 120000) / 2000) + 1;
      const fn = async () => {
        this.pollAttempts += 1;
        const elapsed = Date.now() - this.pollStartTs;
        if (elapsed > this.pollTimeoutMs || this.pollAttempts > this.pollMaxAttempts) {
          this.stopClusteringPolling();
          return;
        }
        await this.checkClusteringStatus();
      };
      fn();
      this.pollTimer = setInterval(fn, 2000);
      // Stop polling on navigation away or when tab is hidden
      this.pollVisibilityHandler = () => { if (document.hidden) this.stopClusteringPolling(); };
      this.pollPageHideHandler = () => { this.stopClusteringPolling(); };
      this.pollBeforeUnloadHandler = () => { this.stopClusteringPolling(); };
      document.addEventListener('visibilitychange', this.pollVisibilityHandler);
      window.addEventListener('pagehide', this.pollPageHideHandler);
      window.addEventListener('beforeunload', this.pollBeforeUnloadHandler);
    },
    stopClusteringPolling() {
      if (this.pollTimer) { clearInterval(this.pollTimer); this.pollTimer = null; }
      if (this.pollVisibilityHandler) { document.removeEventListener('visibilitychange', this.pollVisibilityHandler); this.pollVisibilityHandler = null; }
      if (this.pollPageHideHandler) { window.removeEventListener('pagehide', this.pollPageHideHandler); this.pollPageHideHandler = null; }
      if (this.pollBeforeUnloadHandler) { window.removeEventListener('beforeunload', this.pollBeforeUnloadHandler); this.pollBeforeUnloadHandler = null; }
    },
    async checkClusteringStatus() {
      const isMulti = !!this.multiKey && Array.isArray(this.episodeIds) && this.episodeIds.length > 1;
      if (isMulti) {
        const resp = await fetch(
          `/api/v1/clustering/status/?multi_key=${encodeURIComponent(this.multiKey)}`,
          { credentials: 'same-origin' }
        );
        if (!resp.ok) return;
        const data = await resp.json();
        if (data && data.available) {
          this.clusteringAvailable = true;
          await this.fetchClusteringResult();
          this.stopClusteringPolling();
        }
        return;
      }
      const isSingle = Array.isArray(this.episodeIds) && this.episodeIds.length === 1;
      if (isSingle) {
        const eid = this.episodeIds[0];
        const resp = await fetch(
          `/api/v1/clustering/status/?episode_id=${encodeURIComponent(eid)}`,
          { credentials: 'same-origin' }
        );
        if (!resp.ok) return;
        const data = await resp.json();
        if (!data) return;
        if (data.obs_available || data.agent_outs_available) {
          await this.fetchClusteringResult();
        }
        if (data.available) {
          this.clusteringAvailable = true;
          this.stopClusteringPolling();
        }
      }
    },
    async fetchClusteringResult() {
      let resp = null;
      const isMulti = !!this.multiKey && Array.isArray(this.episodeIds) && this.episodeIds.length > 1;
      if (isMulti) {
        resp = await fetch(
          `/api/v1/clustering/result/?multi_key=${encodeURIComponent(this.multiKey)}`,
          { credentials: 'same-origin' }
        );
      } else if (Array.isArray(this.episodeIds) && this.episodeIds.length === 1) {
        const eid = this.episodeIds[0];
        resp = await fetch(
          `/api/v1/clustering/result/?episode_id=${encodeURIComponent(eid)}`,
          { credentials: 'same-origin' }
        );
      }
      if (!resp || !resp.ok) return;
      const obj = await resp.json();
      this.clustering = obj;
      const store = (window.Alpine && Alpine.store('insights')) || null;
      if (store) store.clustering = obj;
      if (this.clustering && ((this.clustering.obs && this.clustering.agent_outs) || isMulti)) {
        this.toastText = 'Clustering computation completed';
        this.showToast = true;
        if (this.toastTimer) { clearTimeout(this.toastTimer); }
        this.toastTimer = setTimeout(
          () => { this.showToast = false; this.toastTimer = null; },
          4000
        );
      }
    },
  }));
});
