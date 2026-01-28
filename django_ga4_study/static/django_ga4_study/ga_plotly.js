(function(){
    if (!window.GA) return;

    const interactionCooldown = new Map();
    const COOLDOWN_MS = 500;

    function shouldTrack(plotId, eventType) {
        const key = `${plotId}:${eventType}`;
        const now = Date.now();
        const lastTime = interactionCooldown.get(key) || 0;

        if (now - lastTime < COOLDOWN_MS) {
            return false;
        }
        interactionCooldown.set(key, now);
        return true;
    }

    function getPlotId(graphDiv) {
        if (graphDiv.id) return graphDiv.id;
        const gaId = graphDiv.closest('[data-ga-id]');
        if (gaId) return gaId.dataset.gaId;
        const xRef = graphDiv.getAttribute('x-ref');
        if (xRef) return xRef;
        return 'unknown-plot';
    }

    function getPlotTitle(graphDiv) {
        try {
            if (graphDiv._fullLayout && graphDiv._fullLayout.title) {
                return graphDiv._fullLayout.title.text || null;
            }
        } catch (e) {}
        return null;
    }

    function attachPlotlyListeners(graphDiv) {
        const plotId = getPlotId(graphDiv);

        graphDiv.on('plotly_click', function(data) {
            if (!shouldTrack(plotId, 'click')) return;

            const point = data.points[0];
            window.GA.event('plotly_click', {
                plot_id: plotId,
                plot_title: getPlotTitle(graphDiv),
                trace_name: point.data.name || null,
                point_x: point.x,
                point_y: point.y,
                point_index: point.pointIndex
            });
        });

        graphDiv.on('plotly_hover', function(data) {
            if (!shouldTrack(plotId, 'hover')) return;

            const point = data.points[0];
            window.GA.event('plotly_hover', {
                plot_id: plotId,
                plot_title: getPlotTitle(graphDiv),
                trace_name: point.data.name || null,
                point_x: point.x,
                point_y: point.y
            });
        });

        graphDiv.on('plotly_unhover', function(data) {
            if (!shouldTrack(plotId, 'unhover')) return;

            window.GA.event('plotly_unhover', {
                plot_id: plotId,
                plot_title: getPlotTitle(graphDiv)
            });
        });

        graphDiv.on('plotly_doubleclick', function() {
            if (!shouldTrack(plotId, 'doubleclick')) return;

            window.GA.event('plotly_doubleclick', {
                plot_id: plotId,
                plot_title: getPlotTitle(graphDiv)
            });
        });

        graphDiv.on('plotly_relayout', function(eventData) {
            if (!shouldTrack(plotId, 'relayout')) return;
            if (!eventData) return;

            const isZoom = eventData['xaxis.range[0]'] !== undefined ||
                           eventData['yaxis.range[0]'] !== undefined ||
                           eventData['xaxis.autorange'] !== undefined;
            const isPan = eventData.dragmode === 'pan';
            const isReset = eventData['xaxis.autorange'] === true &&
                            eventData['yaxis.autorange'] === true;

            let action = 'other';
            if (isReset) action = 'reset';
            else if (isZoom) action = 'zoom';
            else if (isPan) action = 'pan';

            if (action === 'other' && Object.keys(eventData).length === 0) return;

            window.GA.event('plotly_relayout', {
                plot_id: plotId,
                plot_title: getPlotTitle(graphDiv),
                action: action,
                has_x_change: eventData['xaxis.range[0]'] !== undefined,
                has_y_change: eventData['yaxis.range[0]'] !== undefined
            });
        });

        graphDiv.on('plotly_legendclick', function(data) {
            if (!shouldTrack(plotId, 'legendclick')) return;

            window.GA.event('plotly_legendclick', {
                plot_id: plotId,
                plot_title: getPlotTitle(graphDiv),
                trace_name: data.data[data.curveNumber].name,
                trace_index: data.curveNumber,
                new_visibility: data.data[data.curveNumber].visible === true ? 'hidden' : 'visible'
            });
        });

        graphDiv.on('plotly_legenddoubleclick', function(data) {
            if (!shouldTrack(plotId, 'legenddoubleclick')) return;

            window.GA.event('plotly_legenddoubleclick', {
                plot_id: plotId,
                plot_title: getPlotTitle(graphDiv),
                trace_name: data.data[data.curveNumber].name,
                trace_index: data.curveNumber
            });
        });

        graphDiv.on('plotly_selected', function(data) {
            if (!shouldTrack(plotId, 'selected')) return;
            if (!data || !data.points) return;

            window.GA.event('plotly_selected', {
                plot_id: plotId,
                plot_title: getPlotTitle(graphDiv),
                points_selected: data.points.length,
                traces_involved: [...new Set(data.points.map(p => p.data.name))].join(',')
            });
        });

        graphDiv.on('plotly_deselect', function() {
            if (!shouldTrack(plotId, 'deselect')) return;

            window.GA.event('plotly_deselect', {
                plot_id: plotId,
                plot_title: getPlotTitle(graphDiv)
            });
        });

        graphDiv._gaTracked = true;
    }

    function scanForPlots() {
        if (typeof Plotly === 'undefined') return;

        const plotDivs = document.querySelectorAll('.js-plotly-plot, [class*="plotly"]');
        plotDivs.forEach(function(div) {
            if (div._fullLayout && !div._gaTracked) {
                attachPlotlyListeners(div);
            }
        });

        document.querySelectorAll('[x-ref]').forEach(function(div) {
            if (div._fullLayout && !div._gaTracked) {
                attachPlotlyListeners(div);
            }
        });
    }

    const originalReact = window.Plotly && window.Plotly.react;
    const originalNewPlot = window.Plotly && window.Plotly.newPlot;

    function wrapPlotlyMethod(original, methodName) {
        return function(gd) {
            const result = original.apply(this, arguments);

            if (result && result.then) {
                result.then(function(graphDiv) {
                    if (graphDiv && !graphDiv._gaTracked) {
                        attachPlotlyListeners(graphDiv);
                    }
                });
            }

            return result;
        };
    }

    function initPlotlyTracking() {
        if (typeof Plotly !== 'undefined') {
            if (Plotly.react) {
                Plotly.react = wrapPlotlyMethod(originalReact || Plotly.react, 'react');
            }
            if (Plotly.newPlot) {
                Plotly.newPlot = wrapPlotlyMethod(originalNewPlot || Plotly.newPlot, 'newPlot');
            }
        }

        scanForPlots();

        const observer = new MutationObserver(function(mutations) {
            let shouldScan = false;
            mutations.forEach(function(mutation) {
                if (mutation.addedNodes.length > 0) {
                    shouldScan = true;
                }
            });
            if (shouldScan) {
                setTimeout(scanForPlots, 100);
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    window.GAPlotly = {
        init: initPlotlyTracking,
        attachListeners: attachPlotlyListeners,
        scan: scanForPlots
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(initPlotlyTracking, 500);
        });
    } else {
        setTimeout(initPlotlyTracking, 500);
    }
})();
