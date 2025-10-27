// Optional SPA route helper: call GA.pageView() when your router changes.
(function(){
    if (!window.GA) return;
    window.GA.pageView = function(extra){
      window.GA.event('page_view', Object.assign({
        page_location: location.href,
        page_title: document.title
      }, extra || {}));
    };
  })();
