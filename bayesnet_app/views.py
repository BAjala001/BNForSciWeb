# Configure matplotlib to use Agg backend
import matplotlib
matplotlib.use('Agg')

from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .models import BayesNetModel
from BayesNet import BayesNet
from Node import Node  # Import Node directly from Node.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import itertools
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.models import User

def index(request):
    """Home page view"""
    return render(request, 'bayesnet_app/index.html')

def user_login(request):
    if request.user.is_authenticated:
        return redirect('bayesnet_app:index')
        
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            messages.success(request, 'Successfully logged in!')
            return redirect('bayesnet_app:index')
        else:
            messages.error(request, 'Invalid username or password.')
            
    return render(request, 'bayesnet_app/login.html')

def user_register(request):
    if request.user.is_authenticated:
        return redirect('bayesnet_app:index')
        
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')
        
        if password1 != password2:
            messages.error(request, 'Passwords do not match!')
            return render(request, 'bayesnet_app/register.html')
            
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists!')
            return render(request, 'bayesnet_app/register.html')
            
        if User.objects.filter(email=email).exists():
            messages.error(request, 'Email already registered!')
            return render(request, 'bayesnet_app/register.html')
            
        user = User.objects.create_user(username=username, email=email, password=password1)
        login(request, user)
        messages.success(request, 'Registration successful!')
        return redirect('bayesnet_app:index')
        
    return render(request, 'bayesnet_app/register.html')

def user_logout(request):
    logout(request)
    messages.success(request, 'Successfully logged out!')
    return redirect('bayesnet_app:index')

@login_required(login_url='bayesnet_app:login')
def create_network(request):
    """View for creating a new Bayesian Network"""
    if request.method == 'POST':
        data = json.loads(request.body)
        network = BayesNet(label=data['name'])
        
        # First pass: Add nodes
        for node_data in data['nodes']:
            node = Node(
                name=node_data['name'],
                label=node_data['label'],
                values=node_data['values'],
                parents=node_data['parents']
            )
            network.AddNode(node)
        
        # Second pass: Set CPTs
        for node_data in data['nodes']:
            # Set CPT if provided
            if 'cpt' in node_data:
                cpt_values = node_data['cpt']
                if isinstance(cpt_values, list):
                    node = network.nodes[node_data['name']]
                    cpt = create_cpt_dataframe(node, cpt_values, network)
                    network.SetCpt(node.name, cpt)
        
        # Save network
        network.Save()
        return JsonResponse({'success': True, 'network_id': network.label})
    
    return render(request, 'bayesnet_app/create.html')

@login_required(login_url='bayesnet_app:login')
def analyze_network(request, network_id):
    """View for analyzing an existing network"""
    try:
        network = BayesNet.LoadBN(network_id)
        context = {
            'network': network,
            'nodes': network.nodes
        }
        return render(request, 'bayesnet_app/analyze.html', context)
    except:
        return redirect('bayesnet_app:index')

def create_cpt_dataframe(node, cpt_values, network):
    """Helper function to create CPT DataFrame with consistent format"""
    try:
        if node.parents:
            # For nodes with parents, reshape the CPT
            parent_nodes = [network.nodes[p] for p in node.parents]
            total_parent_states = np.prod([len(p.values) for p in parent_nodes])
            num_node_values = len(node.values)
            cpt_matrix = np.array(cpt_values).reshape(total_parent_states, num_node_values)
            
            # Generate all parent value combinations
            parent_values = [network.nodes[p].values for p in node.parents]
            parent_combinations = list(itertools.product(*parent_values))
            
            # Create rows for each parent combination
            rows = []
            for i, combo in enumerate(parent_combinations):
                # Create a row for each value of the node
                for j, node_value in enumerate(node.values):
                    row = {}
                    # Add parent values
                    for k, parent in enumerate(node.parents):
                        row[parent] = str(combo[k])  # Convert to string
                    # Add node value and probability
                    row[node.name] = str(node_value)  # Node's own value
                    row['Prob'] = float(cpt_matrix[i, j])  # Probability for this combination
                    rows.append(row)
            
            cpt = pd.DataFrame(rows)
            
            # Ensure correct column order
            cols = list(node.parents) + [node.name, 'Prob']
            cpt = cpt[cols]
            
        else:
            # For root nodes
            rows = []
            for i, value in enumerate(node.values):
                rows.append({
                    node.name: str(value),
                    'Prob': float(cpt_values[i])
                })
            cpt = pd.DataFrame(rows)
            cpt = cpt[[node.name, 'Prob']]  # Ensure correct column order
        
        print(f"Created CPT for {node.name}:")  # Debug log
        print(cpt)  # Debug log
        return cpt
        
    except Exception as e:
        print(f"Error creating CPT for {node.name}: {str(e)}")
        raise

def set_findings_on_network(network, findings):
    """Helper function to set findings on a network"""
    has_findings = False
    if findings:
        print("Setting findings:", findings)
        for node_name, value in findings.items():
            if value:  # Only set finding if value is not empty
                node = network.nodes[node_name]
                node.finding = {node_name: str(value)}  # Set finding directly on node
                has_findings = True
                print(f"Set finding for {node_name}: {value}")  # Debug log
    return has_findings

@csrf_exempt
def calculate_marginals(request):
    """API endpoint for calculating marginals"""
    if request.method == 'POST':
        try:
            # Parse request data
            data = json.loads(request.body)
            
            # Debug logging
            print("Received data:", data)
            
            # Check if network data exists
            if not data or 'network' not in data:
                return JsonResponse({'error': 'No network data provided'}, status=400)
            
            network_data = data['network']
            
            # Validate network data
            if 'nodes' not in network_data or not network_data['nodes']:
                return JsonResponse({'error': 'No nodes found in network data'}, status=400)
            
            # Create a new BayesNet instance with a default label
            network = BayesNet(label="user_network")
            
            try:
                # First pass: Add all nodes to the network
                for node_data in network_data['nodes']:
                    # Create a Node object with proper string values
                    node = Node(
                        name=str(node_data['name']),
                        label=str(node_data['label']),
                        values=[str(v) for v in node_data['values']],
                        parents=[str(p) for p in node_data['parents']]
                    )
                    network.AddNode(node)
                    print(f"Added node: {node.name}, type: {type(node)}")  # Debug log
                
                # Second pass: Set CPTs after all nodes are added
                for node_data in network_data['nodes']:
                    node_name = node_data['name']
                    
                    # Set CPT if provided
                    if 'cpt' in node_data and node_data['cpt']:
                        cpt_values = node_data['cpt']
                        if isinstance(cpt_values, list):
                            node = network.nodes[node_name]
                            cpt = create_cpt_dataframe(node, cpt_values, network)
                            network.SetCpt(node_name, cpt)
                
                # Check if all CPTs sum to exactly 1
                problematic_nodes = []
                for node_name, node in network.nodes.items():
                    cpt = node.cpt
                    if node.parents:
                        # For nodes with parents, group by parent values
                        parent_columns = node.parents
                        grouped = cpt.groupby(parent_columns)['Prob'].sum()
                        # Check if any group's sum is not exactly 1
                        for parent_values, prob_sum in grouped.items():
                            if prob_sum != 1.0:
                                problematic_nodes.append(f"{node.label} ({node_name}) for parent values {parent_values}")
                    else:
                        # For root nodes, simply sum all probabilities
                        prob_sum = cpt['Prob'].sum()
                        if prob_sum != 1.0:
                            problematic_nodes.append(f"{node.label} ({node_name})")
                
                if problematic_nodes:
                    error_msg = "CPT probabilities must sum to exactly 1.0 for the following nodes:\n"
                    error_msg += "\n".join(problematic_nodes)
                    return JsonResponse({'error': error_msg}, status=400)
                
                # Calculate joint distribution first
                print("Calculating joint distribution...")
                network.CalcJointDist()
                
                # Set findings if provided
                has_findings = False
                if 'findings' in network_data and network_data['findings']:
                    findings = network_data['findings']
                    print("Setting findings:", findings)
                    has_findings = network.SetFindings(findings)
                
                # Calculate marginals
                if has_findings:
                    print("Calculating marginals with findings...")
                    network.CalcMargGivenFindings()
                    marginals = network.margWithFindings
                else:
                    print("Calculating marginals without findings...")
                    network.CalcMarginals()
                    marginals = network.marginals
                
                # Convert marginals to JSON-serializable format
                marginals_json = {}
                for node_name, marginal in marginals.items():
                    # Convert DataFrame to list of dicts with exact format
                    node_marginals = []
                    for _, row in marginal.iterrows():
                        value = row[node_name] if node_name in row else row.index[0]
                        # Convert probability to percentage (already in decimal form, just multiply by 100)
                        prob = float(row['Prob']) * 100
                        node_marginals.append({str(value): str(prob)})
                    marginals_json[node_name] = node_marginals
                
                print("Sending marginals to frontend:", marginals_json)  # Debug log
                return JsonResponse({'marginals': marginals_json})
                
            except Exception as e:
                print(f"Error in network operations: {str(e)}")
                raise
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            print(f"Error in calculate_marginals: {str(e)}")
            return JsonResponse({'error': str(e)}, status=400)
            
    return JsonResponse({'error': 'Invalid request method'}, status=400)

@csrf_exempt
def set_finding(request):
    """API endpoint for setting findings"""
    if request.method == 'POST':
        data = json.loads(request.body)
        network = BayesNet.LoadBN(data['network_id'])
        network.SetFinding(data['node'], data['value'])
        network.CalcMargGivenFindings()
        
        # Convert marginals to JSON-serializable format
        marginals = {}
        for node_name, marginal in network.margWithFindings.items():
            marginals[node_name] = marginal.to_dict('records')
        
        return JsonResponse({'marginals': marginals})
    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def plot_marginals(request):
    """API endpoint for plotting marginals"""
    if request.method == 'POST':
        try:
            # Parse request data
            data = json.loads(request.body)
            
            # Debug logging
            print("Received plot data:", data)
            
            # Check if network data exists
            if not data or 'network' not in data:
                print("Error: Missing network data")
                return JsonResponse({'error': 'No network data provided'}, status=400)
            
            network_data = data['network']
            
            # Validate network data
            if 'nodes' not in network_data or not network_data['nodes']:
                print("Error: Missing nodes in network data")
                return JsonResponse({'error': 'No nodes found in network data'}, status=400)
            
            try:
                import matplotlib
                matplotlib.use('Agg')  # Ensure we're using Agg backend
                import matplotlib.pyplot as plt
                plt.ioff()  # Turn off interactive mode
                
                # Create a new BayesNet instance with a default label
                network = BayesNet(label="user_network")
                
                # First pass: Add all nodes to the network
                for node_data in network_data['nodes']:
                    # Create a Node object with proper string values and visual properties
                    node = Node(
                        name=str(node_data['name']),
                        label=str(node_data['label']),
                        values=[str(v) for v in node_data['values']],
                        parents=[str(p) for p in node_data['parents']]
                    )
                    # Set visual properties
                    node.xPos = float(node_data.get('xPos', 0))
                    node.yPos = float(node_data.get('yPos', 0))
                    node.width = float(node_data.get('width', 0.9))
                    node.height = float(node_data.get('height', 0.4))
                    node.edgecolor = str(node_data.get('edgecolor', '#0000ff'))
                    node.facecolor = str(node_data.get('facecolor', '#add8e6'))
                    node.linewidth = float(node_data.get('linewidth', 1))
                    node.fontsize = int(node_data.get('fontsize', 12))
                    node.textcolor = str(node_data.get('textcolor', '#000000'))
                    node.ha = 'center'
                    node.va = 'center'
                    
                    network.AddNode(node)
                    print(f"Added node: {node.name}, type: {type(node)}")  # Debug log
                
                # Second pass: Set CPTs after all nodes are added
                for node_data in network_data['nodes']:
                    node_name = node_data['name']
                    
                    # Set CPT if provided
                    if 'cpt' in node_data and node_data['cpt']:
                        cpt_values = node_data['cpt']
                        if isinstance(cpt_values, list):
                            node = network.nodes[node_name]
                            cpt = create_cpt_dataframe(node, cpt_values, network)
                            network.SetCpt(node_name, cpt)
                            print(f"Set CPT for node: {node.name}")  # Debug log

                # Check if all CPTs sum to exactly 1
                problematic_nodes = []
                for node_name, node in network.nodes.items():
                    cpt = node.cpt
                    if node.parents:
                        # For nodes with parents, group by parent values
                        parent_columns = node.parents
                        grouped = cpt.groupby(parent_columns)['Prob'].sum()
                        # Check if any group's sum is not exactly 1
                        for parent_values, prob_sum in grouped.items():
                            if prob_sum != 1.0:
                                problematic_nodes.append(f"{node.label} ({node_name}) for parent values {parent_values}")
                    else:
                        # For root nodes, simply sum all probabilities
                        prob_sum = cpt['Prob'].sum()
                        if prob_sum != 1.0:
                            problematic_nodes.append(f"{node.label} ({node_name})")
                
                if problematic_nodes:
                    error_msg = "CPT probabilities must sum to exactly 1.0 for the following nodes:\n"
                    error_msg += "\n".join(problematic_nodes)
                    return JsonResponse({'error': error_msg}, status=400)
                
                # Calculate joint distribution first
                print("Calculating joint distribution...")
                network.CalcJointDist()
                
                # Set findings if provided
                has_findings = False
                if 'findings' in network_data and network_data['findings']:
                    findings = network_data['findings']
                    print("Setting findings for plot:", findings)
                    has_findings = network.SetFindings(findings)
                
                # Calculate marginals with findings if needed
                if has_findings:
                    print("Calculating marginals with findings...")
                    network.CalcMargGivenFindings()
                else:
                    print("Calculating marginals without findings...")
                    network.CalcMarginals()
                
                # Get the figure from PlotMarginals
                print("Creating plot with findings:", has_findings)  # Debug log
                fig = network.PlotMarginals(with_findings=has_findings)
                
                # Save plot to bytes buffer
                buffer = io.BytesIO()
                fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                buffer.seek(0)
                image_png = buffer.getvalue()
                buffer.close()
                
                # Clean up
                plt.close(fig)
                plt.close('all')
                
                # Encode the image in base64
                graphic = base64.b64encode(image_png).decode('utf-8')
                
                # Collect node positions
                node_positions = [
                    {
                        'name': node_name,
                        'xPos': float(node.xPos),
                        'yPos': float(node.yPos)
                    }
                    for node_name, node in network.nodes.items()
                ]
                
                return JsonResponse({
                    'success': True,
                    'plot': graphic,
                    'node_positions': node_positions
                })
                
            except ImportError as e:
                print(f"Error importing matplotlib: {str(e)}")
                return JsonResponse({
                    'error': 'Unable to generate plot due to missing dependencies. Please contact support.'
                }, status=500)
            except Exception as e:
                print(f"Error in network operations: {str(e)}")
                return JsonResponse({'error': str(e)}, status=400)
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            print(f"Error in plot_marginals: {str(e)}")
            return JsonResponse({'error': str(e)}, status=400)
            
    return JsonResponse({'error': 'Invalid request method'}, status=400)

@csrf_exempt
def save_network(request):
    """API endpoint for saving a network"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            network = BayesNet(label="user_network")
            
            # First pass: Add nodes
            for node_data in data['nodes']:
                node = Node(
                    name=node_data['name'],
                    label=node_data['label'],
                    values=node_data['values'],
                    parents=node_data['parents']
                )
                network.AddNode(node)
            
            # Second pass: Set CPTs
            for node_data in data['nodes']:
                node_name = node_data['name']
                node = network.nodes[node_name]
                
                # Set CPT if provided
                if 'cpt' in node_data:
                    cpt_values = node_data['cpt']
                    if isinstance(cpt_values, list):
                        cpt = create_cpt_dataframe(node, cpt_values, network)
                        network.SetCpt(node.name, cpt)
            
            # Save network
            network.Save()
            return JsonResponse({'success': True, 'network_id': network.label})
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def plot_network(request):
    """API endpoint for plotting network structure"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Create a new BayesNet instance
            bn = BayesNet(label="network_plot")
            
            # First pass: Add all nodes to the network
            for node_data in data['nodes']:
                # Create a Node object with proper string values
                node = Node(
                    name=str(node_data['name']),
                    label=str(node_data['label']),
                    values=[str(v) for v in node_data['values']],
                    parents=[str(p) for p in node_data['parents']]
                )
                
                # Set visual properties
                node.xPos = float(node_data.get('xPos', 0))
                node.yPos = float(node_data.get('yPos', 0))
                node.width = float(node_data.get('width', 0.8))
                node.height = float(node_data.get('height', 0.4))
                node.edgecolor = str(node_data.get('edgecolor', '#0000ff'))
                node.facecolor = str(node_data.get('facecolor', '#add8e6'))
                node.linewidth = float(node_data.get('linewidth', 1))
                node.fontsize = int(node_data.get('fontsize', 12))
                node.textcolor = str(node_data.get('textcolor', '#000000'))
                node.ha = 'center'
                node.va = 'center'
                
                bn.AddNode(node)
                
            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            
            # Draw arrows between nodes first
            for node_name, node in bn.nodes.items():
                for parent_name in node.parents:
                    parent = bn.nodes[parent_name]
                    
                    # Calculate edge points
                    dx = node.xPos - parent.xPos
                    dy = node.yPos - parent.yPos
                    angle = np.arctan2(dy, dx)
                    
                    # Calculate start and end points
                    start_x = parent.xPos + (parent.width/2) * np.cos(angle)
                    start_y = parent.yPos + (parent.height/2) * np.sin(angle)
                    end_x = node.xPos - (node.width/2) * np.cos(angle)
                    end_y = node.yPos - (node.height/2) * np.sin(angle)
                    
                    # Draw arrow
                    arrow = plt.matplotlib.patches.FancyArrowPatch(
                        (start_x, start_y),
                        (end_x, end_y),
                        arrowstyle='-|>',
                        connectionstyle='arc3,rad=0',
                        mutation_scale=15,
                        linewidth=1.5,
                        color='gray',
                        zorder=1
                    )
                    ax.add_patch(arrow)
            
            # Draw nodes
            for node_name, node in bn.nodes.items():
                # Draw node ellipse with visual properties
                ellipse = plt.matplotlib.patches.Ellipse(
                    (node.xPos, node.yPos),
                    node.width,
                    node.height,
                    facecolor=node.facecolor,
                    edgecolor=node.edgecolor,
                    linewidth=node.linewidth,
                    alpha=0.7,
                    zorder=2
                )
                ax.add_patch(ellipse)
                
                # Add node label with visual properties
                ax.text(
                    node.xPos,
                    node.yPos,
                    node.name,  # Only display the node name
                    ha=node.ha,
                    va=node.va,
                    fontsize=node.fontsize,
                    color=node.textcolor,
                    zorder=3
                )
            
            # Set plot bounds
            all_x = [node.xPos for node in bn.nodes.values()]
            all_y = [node.yPos for node in bn.nodes.values()]
            margin = 0.5
            
            if all_x and all_y:
                plt.xlim(min(all_x) - margin, max(all_x) + margin)
                plt.ylim(max(all_y) + margin, min(all_y) - margin)  # Invert Y-axis by swapping min and max
            
            # Show axis with grid
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            plt.axis('on')
            
            # Add title
            plt.title("Network Structure", pad=20, fontsize=12, fontweight='bold')
            
            # Save plot to bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100,
                       facecolor='white', edgecolor='none', pad_inches=0.1,
                       transparent=False)
            plt.close(fig)
            plt.close('all')
            
            # Encode plot data
            buf.seek(0)
            plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()
            
            # Collect node positions
            node_positions = [
                {
                    'name': node_name,
                    'xPos': float(node.xPos),
                    'yPos': float(node.yPos)
                }
                for node_name, node in bn.nodes.items()
            ]
            
            return JsonResponse({
                'success': True,
                'plot': plot_data,
                'node_positions': node_positions
            })

        except Exception as e:
            print(f"Error in plot_network: {str(e)}")
            import traceback
            traceback.print_exc()
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=400)

    return JsonResponse({
        'success': False,
        'error': 'Invalid request method'
    }, status=400)

@csrf_exempt
def save_network_db(request):
    """API endpoint for saving a network to the database"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            network_name = data.get('name')
            
            if not network_name:
                return JsonResponse({'success': False, 'error': 'Network name is required'})
            
            # Create BayesNet instance
            network = BayesNet(label=network_name)
            
            # First pass: Add nodes
            for node_data in data['nodes']:
                node = Node(
                    name=node_data['name'],
                    label=node_data['label'],
                    values=node_data['values'],
                    parents=node_data['parents']
                )
                network.AddNode(node)
            
            # Second pass: Set CPTs
            for node_data in data['nodes']:
                node_name = node_data['name']
                node = network.nodes[node_name]
                
                # Set CPT if provided
                if 'cpt' in node_data:
                    cpt_values = node_data['cpt']
                    if isinstance(cpt_values, list):
                        cpt = create_cpt_dataframe(node, cpt_values, network)
                        network.SetCpt(node.name, cpt)
            
            # Try to find existing network with same name
            try:
                network_model = BayesNetModel.objects.get(name=network_name)
                # Update existing network
                network_model.save_network(network)
                return JsonResponse({'success': True, 'message': 'Network updated successfully'})
            except BayesNetModel.DoesNotExist:
                # Create new network
                network_model = BayesNetModel(name=network_name)
                network_model.save_network(network)
                return JsonResponse({'success': True, 'message': 'Network saved successfully'})
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'error': 'Invalid request'}, status=400)
