document.addEventListener('DOMContentLoaded', function () {
  const dialog = document.getElementById('feedbackModal');
  const feedbackCancelBtn = document.getElementById('feedbackCancelBtn');
  const doneKey = 'mixtape_feedback_done';

  function preventScrollBehindModal(lock) {
    if (!document || !document.body || !document.body.style) {
      return;
    }
    document.body.style.overflow = lock ? 'hidden' : '';
  }

  if (dialog && dialog.showModal && localStorage.getItem(doneKey) !== '1') {
    dialog.showModal();
    preventScrollBehindModal(true);
    dialog.addEventListener('close', function() {
      preventScrollBehindModal(false);
    }, { once: true });
  }

  if (feedbackCancelBtn && dialog) {
    const iframe = dialog.querySelector('iframe');
    // Detect form submission by counting iframe load events.
    // - First: initial render; Second: Redirect to “Thanks for submitting” page
    // FIXME: This is a bit of a hack and could break if Google changes its redirect
    // behavior or even with some browsers/ad-blockers. We should create our own form and
    // use Google Form hooks if we want to keep the feedback dialog.
    let loadCount = 0;
    let submitted = false;

    function updateButton() {
      if (!feedbackCancelBtn) {
        return;
      }
      // Differentiate between completion and rejection
      feedbackCancelBtn.textContent = submitted ? 'Done' : 'Cancel';
    }
    updateButton();

    if (iframe) {
      iframe.addEventListener('load', function () {
        loadCount += 1;
        if (loadCount >= 2) {
          submitted = true;
          updateButton();
        }
      });
    }

    feedbackCancelBtn.addEventListener('click', function () {
      if (submitted) {
        localStorage.setItem(doneKey, '1');
      }
      dialog.close();
    });
  }
});
