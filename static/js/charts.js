// Plotly chart helpers for dashboard
function renderWERChart(elementId, data) {
    const trace = {
        x: ['Substitutions', 'Insertions', 'Deletions', 'Hits'],
        y: [data.substitutions, data.insertions, data.deletions, data.hits],
        type: 'bar',
        marker: { color: ['#ef4444', '#f59e0b', '#6366f1', '#22c55e'] },
    };
    Plotly.newPlot(elementId, [trace], {
        margin: { t: 20, b: 40, l: 40, r: 20 },
        height: 250,
    });
}
