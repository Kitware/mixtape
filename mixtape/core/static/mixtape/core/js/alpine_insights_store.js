document.addEventListener('alpine:init', () => {
  Alpine.store('insights',
    {
      // User-configurable flags
      episodeSummaries: [],
      // Runtime values
      currentStep: 0,
    }
  );
});
