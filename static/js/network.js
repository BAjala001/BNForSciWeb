class NetworkVisualizer {
    constructor(containerId, nodes, options = {}) {
        this.container = document.getElementById(containerId);
        this.nodes = nodes;
        this.options = {
            width: this.container.offsetWidth,
            height: 400,
            nodeRadius: 20,
            ...options
        };
        
        this.svg = d3.select(this.container)
            .append('svg')
            .attr('width', this.options.width)
            .attr('height', this.options.height);
            
        this.initializeSimulation();
        this.drawNetwork();
    }
    
    initializeSimulation() {
        const nodeData = Object.entries(this.nodes).map(([name, node]) => ({
            id: name,
            label: node.label,
            x: Math.random() * this.options.width,
            y: Math.random() * this.options.height
        }));
        
        const links = [];
        Object.entries(this.nodes).forEach(([name, node]) => {
            if (node.parents) {
                node.parents.forEach(parent => {
                    links.push({
                        source: parent,
                        target: name
                    });
                });
            }
        });
        
        this.simulation = d3.forceSimulation(nodeData)
            .force('link', d3.forceLink(links).id(d => d.id))
            .force('charge', d3.forceManyBody().strength(-100))
            .force('center', d3.forceCenter(this.options.width / 2, this.options.height / 2));
            
        this.nodeData = nodeData;
        this.links = links;
    }
    
    drawNetwork() {
        // Draw links
        this.linkElements = this.svg.append('g')
            .selectAll('line')
            .data(this.links)
            .join('line')
            .attr('class', 'link')
            .attr('marker-end', 'url(#arrow)');
            
        // Define arrow marker
        this.svg.append('defs').append('marker')
            .attr('id', 'arrow')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', this.options.nodeRadius + 10)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('class', 'arrow-head');
            
        // Draw nodes
        this.nodeElements = this.svg.append('g')
            .selectAll('circle')
            .data(this.nodeData)
            .join('circle')
            .attr('r', this.options.nodeRadius)
            .attr('class', 'node-circle')
            .call(this.drag());
            
        // Add node labels
        this.labelElements = this.svg.append('g')
            .selectAll('text')
            .data(this.nodeData)
            .join('text')
            .text(d => d.label)
            .attr('class', 'node-label')
            .attr('dy', '.35em');
            
        // Update positions on simulation tick
        this.simulation.on('tick', () => this.updatePositions());
    }
    
    updatePositions() {
        this.linkElements
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
            
        this.nodeElements
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);
            
        this.labelElements
            .attr('x', d => d.x)
            .attr('y', d => d.y);
    }
    
    drag() {
        return d3.drag()
            .on('start', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            })
            .on('drag', (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on('end', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            });
    }
    
    resize() {
        this.options.width = this.container.offsetWidth;
        this.svg.attr('width', this.options.width);
        this.simulation.force('center', d3.forceCenter(this.options.width / 2, this.options.height / 2));
        this.simulation.alpha(0.3).restart();
    }
}

// Marginals display handler
class MarginalsDisplay {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
    }
    
    update(marginals) {
        Object.entries(marginals).forEach(([nodeName, marginal]) => {
            const nodeContainer = document.getElementById(`marginals-${nodeName}`);
            if (!nodeContainer) return;
            
            nodeContainer.innerHTML = '';
            marginal.forEach(row => {
                const value = Object.keys(row)[0];
                const prob = row[value];
                const width = prob * 100;
                
                const barHtml = `
                    <div class="mb-2">
                        <div class="d-flex justify-content-between">
                            <span>${value}</span>
                            <span>${(prob * 100).toFixed(2)}%</span>
                        </div>
                        <div class="probability-bar" style="width: ${width}%"></div>
                    </div>
                `;
                nodeContainer.innerHTML += barHtml;
            });
        });
    }
}

// Initialize on document load
document.addEventListener('DOMContentLoaded', function() {
    // Network visualization
    if (typeof networkData !== 'undefined') {
        const visualizer = new NetworkVisualizer('networkVisualization', networkData);
        window.addEventListener('resize', () => visualizer.resize());
    }
    
    // Marginals display
    const marginalsDisplay = new MarginalsDisplay('marginalsContainer');
    
    // Calculate marginals button
    const calculateButton = document.getElementById('calculateMarginals');
    if (calculateButton) {
        calculateButton.addEventListener('click', async () => {
            try {
                const response = await fetch('/api/calculate-marginals/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
                    },
                    body: JSON.stringify({
                        network_id: networkId
                    })
                });
                
                const data = await response.json();
                marginalsDisplay.update(data.marginals);
            } catch (error) {
                console.error('Error calculating marginals:', error);
            }
        });
    }
    
    // Finding selects
    document.querySelectorAll('.finding-select').forEach(select => {
        select.addEventListener('change', async function() {
            const nodeName = this.dataset.node;
            const value = this.value;
            
            if (value) {
                try {
                    const response = await fetch('/api/set-finding/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
                        },
                        body: JSON.stringify({
                            network_id: networkId,
                            node: nodeName,
                            value: value
                        })
                    });
                    
                    const data = await response.json();
                    marginalsDisplay.update(data.marginals);
                } catch (error) {
                    console.error('Error setting finding:', error);
                }
            }
        });
    });
}); 