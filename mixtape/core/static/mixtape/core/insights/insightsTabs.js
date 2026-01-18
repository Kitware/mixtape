document.addEventListener('alpine:init', () => {
  Alpine.data('insightsTabs', () => ({
    // initial state
    active: 'Overview',
    maxStepsCount: null,
    avgRewardDisplay: null,
    episodeSummaries: [],
    mediaEpisodeIdx: 0,
    linkMediaToEpisode: true,
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
      // Average reward
      const avgReward = this.computeAverageReward(this.$store.insights.parsedData);
      this.avgRewardDisplay = (avgReward ?? null) !== null ? Number(avgReward).toFixed(2) : null;

      // Episode summary
      this.episodeSummaries = this.$store.insights.episodeIds.map((id, idx) => this.summarizeEpisode(idx, id));

      // Max steps count (for multi-episode mode)
      this.maxStepsCount = (() => {
        const nums = Array.isArray(this.episodeSummaries)
          ? this.episodeSummaries.map(e => (typeof e?.steps === 'number' ? e.steps : 0))
          : [];
        const m = nums.length ? Math.max(...nums) : 0;
        return m || null;
      })();

      // Build watchers
      this.$watch('$store.settings.timelineEpisodeIdx', () => {
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

      // Store
      const store = (window.Alpine && Alpine.store('insights')) || null;
      if (store) {
        // Push derived values
        store.episodeSummaries = this.episodeSummaries;
      }
      // push changes
      this.$watch('episodeSummaries', v => { Alpine.store('insights').episodeSummaries = v; });

      // Polling: multi-episode artifact or single-episode partials
      const isMulti = this.$store.insights.episodeIds.length > 1;
      const isSingle = this.$store.insights.episodeIds.length === 1;
      if ((isMulti && !this.$store.insights.clustering && this.$store.insights.multiKey)
          || (isSingle && (!this.$store.insights.clustering || !this.$store.insights.clustering?.obs || !this.$store.insights.clustering?.agent_outs))) {
        this.startClusteringPolling();
      }
    },
    selectTab(tab) {
      this.active = tab;
    },
    maxStepsTooltip() {
      if (this.episodeSummaries.length <= 1) {
        return `Total Steps: ${this.episodeSummaries[0].steps ?? '—'}`;
      }
      return [
        `Max Steps: ${this.maxStepsCount ?? '—'}`,
        ...this.episodeSummaries
          .map((episodeSummary) => `Ep ${episodeSummary.id}: ${episodeSummary.steps ?? '-'}`)
      ].join('\n');
    },
    // helpers for templates
    getEpisodeStepIndex(epIdx) {
      const steps = this.$store.insights.stepData?.[epIdx] || {};
      const keys = Object.keys(steps);
      if (!keys.length) return null;
      const nums = keys.map(k => +k).sort((a,b) => a - b);
      let chosen = nums[0];
      for (let i = 0; i < nums.length; i++) { if (nums[i] <= this.$store.insights.currentStep) chosen = nums[i]; else break; }
      return chosen;
    },
    getEpisodeStepData(epIdx) {
      const idx = this.getEpisodeStepIndex(epIdx);
      if (idx === null) return null;
      const steps = this.$store.insights.stepData?.[epIdx] || {};
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
      const allRewards = this.$store.insights.rewardsOverTime[idx];
      const steps = allRewards.length || null;
      const perStep = allRewards.filter(rewards => typeof rewards === 'number');
      let totalReward = this.$store.insights.parsedData.episode_total_rewards[idx];
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
      const isMulti = !!this.$store.insights.multiKey && this.$store.insights.episodeIds.length > 1;
      if (isMulti) {
        const resp = await fetch(
          `/api/v1/clustering/status/?multi_key=${encodeURIComponent(this.$store.insights.multiKey)}`,
          { credentials: 'same-origin' }
        );
        if (!resp.ok) return;
        const data = await resp.json();
        if (data && data.available) {
          this.$store.insights.clusteringAvailable = true;
          await this.fetchClusteringResult();
          this.stopClusteringPolling();
        }
        return;
      }
      const isSingle = this.$store.insights.episodeIds.length === 1;
      if (isSingle) {
        const eid = this.$store.insights.episodeIds[0];
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
          this.$store.insights.clusteringAvailable = true;
          this.stopClusteringPolling();
        }
      }
    },
    async fetchClusteringResult() {
      let resp = null;
      const isMulti = !!this.$store.insights.multiKey && this.$store.insights.episodeIds.length > 1;
      if (isMulti) {
        resp = await fetch(
          `/api/v1/clustering/result/?multi_key=${encodeURIComponent(this.$store.insights.multiKey)}`,
          { credentials: 'same-origin' }
        );
      } else if (this.$store.insights.episodeIds.length === 1) {
        const eid = this.$store.insights.episodeIds[0];
        resp = await fetch(
          `/api/v1/clustering/result/?episode_id=${encodeURIComponent(eid)}`,
          { credentials: 'same-origin' }
        );
      }
      if (!resp || !resp.ok) return;
      const obj = await resp.json();
      this.$store.insights.clustering = obj;
      const store = (window.Alpine && Alpine.store('insights')) || null;
      if (store) store.clustering = obj;
      if (this.$store.insights.clustering && ((this.$store.insights.clustering.obs && this.$store.insights.clustering.agent_outs) || isMulti)) {
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
