// Enhanced GA tracking with state information
(function(){
    if (!window.GA) return;

    // Event sequence counter for guaranteed ordering
    let eventSequence = 0;
    let sessionStartTime = Date.now();
    let lastActivityTime = Date.now();
    let sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);

    // Helper to safely get Alpine store data
    function getAlpineStore(storeName) {
      try {
        return window.Alpine?.store?.(storeName);
      } catch(e) {
        return null;
      }
    }

    // Helper to get data-ga-id from element or its attributes
    function getElementId(el) {
      return el.getAttribute('data-ga-id') || el.id || el.name || 'unknown';
    }

    // Helper to add common metadata to all events
    function enrichEvent(eventName, params) {
      eventSequence++;
      lastActivityTime = Date.now();

      const enrichedParams = Object.assign({
        event_sequence: eventSequence,
        session_id: sessionId,
        session_duration_ms: Date.now() - sessionStartTime,
        page_path: location.pathname,
        page_url: location.href
      }, params);

      window.GA.event(eventName, enrichedParams);
    }

    window.GAState = {
      /**
       * Track a menu/dropdown interaction with current state
       * @param {string} menuId - Identifier for the menu
       * @param {string} action - Action performed (open, close, item_click)
       * @param {Object} extraParams - Additional parameters to include
       */
      trackMenu(menuId, action, extraParams = {}) {
        window.GA.event('menu_interaction', Object.assign({
          menu_id: menuId,
          action: action
        }, extraParams));
      },

      /**
       * Track visualization option changes with full state
       * Captures which plots are visible/hidden after the change
       */
      trackVisibilityToggle(optionName, newState) {
        const visOptions = getAlpineStore('visOptions');
        if (!visOptions) {
          enrichEvent('visibility_toggle', {
            option_name: optionName,
            new_state: newState
          });
          return;
        }

        const visiblePlots = (visOptions.visible || []).join(',');
        const totalOptions = (visOptions.allOptions || []).length;
        const visibleCount = (visOptions.visible || []).length;

        enrichEvent('visibility_toggle', {
          option_name: optionName,
          new_state: newState,
          visible_plots: visiblePlots,
          visible_count: visibleCount,
          total_count: totalOptions,
          all_visible: visibleCount === totalOptions
        });
      },

      /**
       * Track playback speed changes
       */
      trackPlaybackSpeed(newSpeed, currentStep, maxSteps) {
        enrichEvent('playback_speed_change', {
          new_speed: newSpeed,
          current_step: currentStep,
          max_steps: maxSteps,
          progress_percent: maxSteps > 0 ? Math.round((currentStep / maxSteps) * 100) : 0
        });
      },

      /**
       * Track playback state changes (play/pause)
       */
      trackPlaybackState(isPlaying, currentStep, maxSteps, playbackSpeed) {
        enrichEvent('playback_state_change', {
          action: isPlaying ? 'play' : 'pause',
          current_step: currentStep,
          max_steps: maxSteps,
          playback_speed: playbackSpeed,
          progress_percent: maxSteps > 0 ? Math.round((currentStep / maxSteps) * 100) : 0
        });
      },

      /**
       * Track theme/layout changes
       */
      trackThemeChange(isDarkMode) {
        enrichEvent('theme_change', {
          theme: isDarkMode ? 'dark' : 'light'
        });
      },

      trackLayoutChange(isDense) {
        enrichEvent('layout_change', {
          layout: isDense ? 'dense' : 'normal'
        });
      },

      /**
       * Track legend position/overlap changes
       */
      trackLegendSettings(settingType, newValue) {
        const visOptions = getAlpineStore('visOptions');
        enrichEvent('legend_settings_change', {
          setting_type: settingType,
          new_value: newValue,
          legend_position: visOptions?.legendPosition || 'unknown',
          legend_overlap: visOptions?.legendOverlap || 'unknown',
          legends_visible: visOptions?.visible?.includes('plotLegends') || false
        });
      },

      /**
       * Track filter interactions with current filter state
       */
      trackFilterChange(filterType, selectedValues) {
        const valuesList = Array.isArray(selectedValues) ? selectedValues.join(',') : selectedValues;
        enrichEvent('filter_change', {
          filter_type: filterType,
          selected_values: valuesList,
          selection_count: Array.isArray(selectedValues) ? selectedValues.length : 1
        });
      },

      /**
       * Track episode selection with context
       */
      trackEpisodeSelection(episodeId, isSelected, totalSelected, allowedEnvironment) {
        enrichEvent('episode_selection', {
          episode_id: episodeId,
          action: isSelected ? 'select' : 'deselect',
          total_selected: totalSelected,
          allowed_environment: allowedEnvironment || 'none'
        });
      },

      /**
       * Track timeline scrubber interactions
       */
      trackTimelineSeek(newStep, maxSteps, isDragging) {
        enrichEvent('timeline_seek', {
          new_step: newStep,
          max_steps: maxSteps,
          progress_percent: maxSteps > 0 ? Math.round((newStep / maxSteps) * 100) : 0,
          interaction_type: isDragging ? 'drag' : 'click'
        });
      },

      /**
       * Track manual step changes (buttons, not playback)
       */
      trackStepChange(newStep, maxSteps, changeType) {
        enrichEvent('step_change', {
          new_step: newStep,
          max_steps: maxSteps,
          progress_percent: maxSteps > 0 ? Math.round((newStep / maxSteps) * 100) : 0,
          change_type: changeType // 'forward', 'back', 'start', 'end', 'scrubber'
        });
      },

      /**
       * Track page navigation
       */
      trackNavigation(fromPath, toPath, navigationMethod) {
        enrichEvent('page_navigation', {
          from_path: fromPath,
          to_path: toPath,
          navigation_method: navigationMethod // 'link', 'button', 'browser_back', 'browser_forward'
        });

        // Send page_view event after navigation to update GA4 page context
        // This ensures subsequent events are associated with the new page
        if (window.GA && window.GA.pageView) {
          window.GA.pageView({
            page_path: toPath
          });
        }
      },

      /**
       * Track viewport/scroll changes - which plots are visible
       */
      trackViewportChange(visiblePlots, scrollY, scrollX) {
        enrichEvent('viewport_change', {
          visible_plots: visiblePlots.join(','),
          visible_plot_count: visiblePlots.length,
          scroll_y: Math.round(scrollY),
          scroll_x: Math.round(scrollX),
          viewport_height: window.innerHeight,
          viewport_width: window.innerWidth
        });
      },

      /**
       * Track session start
       */
      trackSessionStart() {
        sessionStartTime = Date.now();
        sessionId = 'session_' + sessionStartTime + '_' + Math.random().toString(36).substr(2, 9);
        eventSequence = 0;

        enrichEvent('session_start', {
          user_agent: navigator.userAgent,
          screen_width: window.screen.width,
          screen_height: window.screen.height,
          viewport_width: window.innerWidth,
          viewport_height: window.innerHeight,
          timezone_offset: new Date().getTimezoneOffset(),
          language: navigator.language
        });
      },

      /**
       * Track session end
       */
      trackSessionEnd(reason) {
        enrichEvent('session_end', {
          reason: reason, // 'page_unload', 'navigation', 'inactivity_timeout'
          session_duration_ms: Date.now() - sessionStartTime,
          total_events: eventSequence,
          last_activity_ms_ago: Date.now() - lastActivityTime
        });
      },

      /**
       * Get current session info
       */
      getSessionInfo() {
        return {
          session_id: sessionId,
          session_duration_ms: Date.now() - sessionStartTime,
          event_count: eventSequence,
          last_activity_ms_ago: Date.now() - lastActivityTime
        };
      }
    };

    // Auto-track session start on load
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', function() {
        window.GAState.trackSessionStart();
      });
    } else {
      window.GAState.trackSessionStart();
    }

    // Auto-track session end on page unload
    window.addEventListener('beforeunload', function() {
      window.GAState.trackSessionEnd('page_unload');
    });

    // Track navigation events
    let currentPath = location.pathname;
    window.addEventListener('popstate', function() {
      const newPath = location.pathname;
      if (newPath !== currentPath) {
        window.GAState.trackNavigation(currentPath, newPath, 'browser_back_forward');
        currentPath = newPath;
      }
    });

    // Track viewport changes (debounced scroll tracking)
    let scrollTimeout;
    let lastScrollY = window.scrollY;
    let lastScrollX = window.scrollX;

    function checkViewportChanges() {
      const scrollY = window.scrollY;
      const scrollX = window.scrollX;

      // Only track if scroll changed significantly (>100px)
      if (Math.abs(scrollY - lastScrollY) > 100 || Math.abs(scrollX - lastScrollX) > 100) {
        // Find visible plots
        const visiblePlots = [];
        document.querySelectorAll('[data-ga-id*="plot"], [data-ga-id*="chart"]').forEach(function(el) {
          const rect = el.getBoundingClientRect();
          const isVisible = rect.top < window.innerHeight && rect.bottom > 0;
          if (isVisible) {
            visiblePlots.push(el.getAttribute('data-ga-id'));
          }
        });

        if (visiblePlots.length > 0) {
          window.GAState.trackViewportChange(visiblePlots, scrollY, scrollX);
        }

        lastScrollY = scrollY;
        lastScrollX = scrollX;
      }
    }

    window.addEventListener('scroll', function() {
      clearTimeout(scrollTimeout);
      scrollTimeout = setTimeout(checkViewportChanges, 300);
    }, { passive: true });

  })();
