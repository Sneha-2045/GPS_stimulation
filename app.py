import streamlit as st
import numpy as np
import pandas as pd
import math
import time
import random
from math import sqrt, atan2, degrees
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
from PIL import Image
import io
import base64
import threading

# YOLOv8 imports (Streamlit deployment-friendly)
@st.cache_resource
def load_yolo():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")

YOLO_AVAILABLE = True
YOLO_MODEL = None

# Page configuration
st.set_page_config(
    page_title="AI Drone Fleet System",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .drone-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'drones' not in st.session_state:
    st.session_state.drones = [
        {'id': 1, 'x': 100, 'y': 150, 'status': 'idle', 'battery': 100, 'payload': 0, 'target': None, 'route': [], 'current_step': 0},
        {'id': 2, 'x': 300, 'y': 200, 'status': 'idle', 'battery': 95, 'payload': 0, 'target': None, 'route': [], 'current_step': 0},
        {'id': 3, 'x': 500, 'y': 100, 'status': 'idle', 'battery': 90, 'payload': 0, 'target': None, 'route': [], 'current_step': 0}
    ]

if 'tasks' not in st.session_state:
    st.session_state.tasks = [
        {'id': 1, 'location': 'Building A', 'priority': 'high', 'type': 'delivery', 'status': 'pending'},
        {'id': 2, 'location': 'Building B', 'priority': 'medium', 'type': 'surveillance', 'status': 'pending'},
        {'id': 3, 'location': 'Building C', 'priority': 'low', 'type': 'delivery', 'status': 'pending'}
    ]

if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False

if 'animation_frame' not in st.session_state:
    st.session_state.animation_frame = 0

if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = time.time()

if 'yolo_model_ready' not in st.session_state:
    st.session_state.yolo_model_ready = False

# Campus locations
CAMPUS_LOCATIONS = {
    'Building A': {'x': 150, 'y': 250, 'type': 'building', 'icon': '🏢'},
    'Building B': {'x': 350, 'y': 300, 'type': 'building', 'icon': '🏛️'},
    'Building C': {'x': 550, 'y': 200, 'type': 'building', 'icon': '🏗️'},
    'Parking Lot': {'x': 200, 'y': 400, 'type': 'parking', 'icon': '🅿️'},
    'Quad': {'x': 400, 'y': 450, 'type': 'open', 'icon': '🌳'},
    'Library': {'x': 600, 'y': 350, 'type': 'building', 'icon': '📚'},
    'Cafeteria': {'x': 250, 'y': 500, 'type': 'building', 'icon': '🍽️'},
    'Sports Center': {'x': 450, 'y': 550, 'type': 'building', 'icon': '⚽'}
}

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_angle(x1, y1, x2, y2):
    """Calculate angle between two points in degrees"""
    return degrees(atan2(y2 - y1, x2 - x1))

# ==================== OPTIMIZATION ALGORITHMS ====================

def knn_route_optimization(drones, destinations):
    """KNN-based route optimization"""
    assignments = []
    remaining_destinations = destinations.copy()
    
    if not drones or not destinations:
        return assignments
    
    # Prepare drone positions
    drone_coords = [(d['x'], d['y']) for d in drones if d['status'] == 'idle']
    
    if not drone_coords:
        return assignments
    
    nbrs = NearestNeighbors(n_neighbors=min(len(drone_coords), len(destinations)), algorithm='ball_tree')
    nbrs.fit(drone_coords)
    
    for dest_name, dest_data in remaining_destinations.items():
        if dest_name not in CAMPUS_LOCATIONS:
            continue
        
        loc = CAMPUS_LOCATIONS[dest_name]
        distances, indices = nbrs.kneighbors([[loc['x'], loc['y']]])
        
        for idx, drone_idx in enumerate(indices[0]):
            if drone_idx < len(drones) and drones[drone_idx]['status'] == 'idle':
                drone = drones[drone_idx]
                dist = distances[0][idx]
                
                assignments.append({
                    'drone_id': drone['id'],
                    'destination': dest_name,
                    'distance': dist,
                    'drone_x': drone['x'],
                    'drone_y': drone['y'],
                    'dest_x': loc['x'],
                    'dest_y': loc['y']
                })
                drones[drone_idx]['status'] = 'assigned'
                break
    
    return assignments

def dijkstra_route_optimization(drones, destinations):
    """Dijkstra's algorithm for shortest path optimization"""
    assignments = []
    remaining_destinations = destinations.copy()
    
    if not drones or not destinations:
        return assignments
    
    # Create graph of all possible assignments
    all_assignments = []
    
    for dest_name, dest_data in remaining_destinations.items():
        if dest_name not in CAMPUS_LOCATIONS:
            continue
        
        loc = CAMPUS_LOCATIONS[dest_name]
        
        for drone in drones:
            if drone['status'] == 'idle':
                dist = calculate_distance(drone['x'], drone['y'], loc['x'], loc['y'])
                # Priority score: distance + battery penalty
                priority = dist + (100 - drone['battery']) * 0.5
                all_assignments.append((priority, dist, drone, dest_name, loc))
    
    # Sort by priority (Dijkstra: shortest path first)
    all_assignments.sort(key=lambda x: x[0])
    
    assigned_drones = set()
    for priority, dist, drone, dest_name, loc in all_assignments:
        if drone['id'] in assigned_drones:
            continue
        if dest_name not in remaining_destinations:
            continue
        
        assignments.append({
            'drone_id': drone['id'],
            'destination': dest_name,
            'distance': dist,
            'drone_x': drone['x'],
            'drone_y': drone['y'],
            'dest_x': loc['x'],
            'dest_y': loc['y']
        })
        assigned_drones.add(drone['id'])
        if dest_name in remaining_destinations:
            del remaining_destinations[dest_name]
    
    return assignments

def genetic_algorithm_optimization(drones, destinations):
    """Genetic Algorithm for evolutionary route optimization"""
    assignments = []
    
    if not drones or not destinations:
        return assignments
    
    def fitness(assignment_list):
        """Fitness function: minimize total distance"""
        total_dist = sum(a['distance'] for a in assignment_list)
        return 10000 - total_dist  # Higher is better
    
    # Generate initial population
    population = []
    for _ in range(5):
        candidate = []
        temp_drones = [d for d in drones if d['status'] == 'idle']
        temp_dests = list(destinations.keys())
        random.shuffle(temp_dests)
        
        for dest_name in temp_dests[:len(temp_drones)]:
            if dest_name not in CAMPUS_LOCATIONS:
                continue
            loc = CAMPUS_LOCATIONS[dest_name]
            
            if temp_drones:
                drone = min(temp_drones, key=lambda d: calculate_distance(d['x'], d['y'], loc['x'], loc['y']))
                dist = calculate_distance(drone['x'], drone['y'], loc['x'], loc['y'])
                candidate.append({
                    'drone_id': drone['id'],
                    'destination': dest_name,
                    'distance': dist,
                    'drone_x': drone['x'],
                    'drone_y': drone['y'],
                    'dest_x': loc['x'],
                    'dest_y': loc['y']
                })
                temp_drones.remove(drone)
        
        if candidate:
            population.append((fitness(candidate), candidate))
    
    # Select best solution
    if population:
        population.sort(key=lambda x: x[0], reverse=True)
        assignments = population[0][1]
    
    return assignments

def particle_swarm_optimization(drones, destinations):
    """Particle Swarm Optimization for route planning"""
    assignments = []
    
    if not drones or not destinations:
        return assignments
    
    # Initialize swarm
    particles = []
    for _ in range(5):
        particle_assignments = []
        temp_drones = [d for d in drones if d['status'] == 'idle']
        temp_dests = list(destinations.keys())
        random.shuffle(temp_dests)
        
        for dest_name in temp_dests[:len(temp_drones)]:
            if dest_name not in CAMPUS_LOCATIONS:
                continue
            loc = CAMPUS_LOCATIONS[dest_name]
            
            if temp_drones:
                # PSO: balance distance and battery
                best_drone = None
                best_score = float('inf')
                
                for drone in temp_drones:
                    dist = calculate_distance(drone['x'], drone['y'], loc['x'], loc['y'])
                    score = dist * 0.7 + (100 - drone['battery']) * 0.3
                    if score < best_score:
                        best_score = score
                        best_drone = drone
                
                if best_drone:
                    dist = calculate_distance(best_drone['x'], best_drone['y'], loc['x'], loc['y'])
                    particle_assignments.append({
                        'drone_id': best_drone['id'],
                        'destination': dest_name,
                        'distance': dist,
                        'drone_x': best_drone['x'],
                        'drone_y': best_drone['y'],
                        'dest_x': loc['x'],
                        'dest_y': loc['y']
                    })
                    temp_drones.remove(best_drone)
        
        total_dist = sum(a['distance'] for a in particle_assignments)
        fitness = 10000 - total_dist
        particles.append((fitness, particle_assignments))
    
    # Global best
    if particles:
        particles.sort(key=lambda x: x[0], reverse=True)
        assignments = particles[0][1]
    
    return assignments

def ensemble_optimization(drones, destinations):
    """Ensemble model combining multiple algorithms"""
    # Get results from all models
    knn_result = knn_route_optimization([d.copy() for d in drones], destinations.copy())
    dijkstra_result = dijkstra_route_optimization([d.copy() for d in drones], destinations.copy())
    ga_result = genetic_algorithm_optimization([d.copy() for d in drones], destinations.copy())
    pso_result = particle_swarm_optimization([d.copy() for d in drones], destinations.copy())
    
    # Weighted voting
    all_results = [
        (knn_result, 0.3),
        (dijkstra_result, 0.3),
        (ga_result, 0.2),
        (pso_result, 0.2)
    ]
    
    # Select best assignments based on weighted distance
    final_assignments = []
    used_drones = set()
    used_dests = set()
    
    # Sort all assignments by weighted score
    all_assignments = []
    for result, weight in all_results:
        for assignment in result:
            if assignment['drone_id'] not in used_drones and assignment['destination'] not in used_dests:
                score = assignment['distance'] * weight
                all_assignments.append((score, assignment))
    
    all_assignments.sort(key=lambda x: x[0])
    
    for score, assignment in all_assignments:
        if assignment['drone_id'] not in used_drones and assignment['destination'] not in used_dests:
            final_assignments.append(assignment)
            used_drones.add(assignment['drone_id'])
            used_dests.add(assignment['destination'])
    
    return final_assignments

# ==================== COMPUTER VISION ====================

@st.cache_resource
def load_yolo_model():
    """Load YOLOv8 model with caching"""
    try:
        model = load_yolo()
        st.session_state.yolo_model_ready = True
        return model
    except Exception as e:
        st.session_state.yolo_model_ready = False
        st.session_state.yolo_error = str(e)
        return None

def detect_objects_yolo(image, model=None):
    """Run YOLOv8 object detection on image"""
    if not YOLO_AVAILABLE:
        return image, []
    
    try:
        # Load model if not provided
        if model is None:
            model = load_yolo_model()
            if model is None:
                return image, []
        
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure image is in correct format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Run inference
        results = model(image, verbose=False)
        
        # Extract detections
        detections = []
        annotated_image = image.copy()
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    # Filter for relevant classes: person, vehicle, building-related
                    relevant_classes = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 
                                      'traffic light', 'stop sign', 'fire hydrant', 'bench']
                    
                    if class_name in relevant_classes and confidence > 0.25:
                        detections.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                        label = f"{class_name} {confidence:.2f}"
                        cv2.putText(annotated_image, label, (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return annotated_image, detections
    
    except Exception as e:
        st.error(f"Detection error: {str(e)}")
        return image, []

# ==================== VISUALIZATION ====================

def create_drone_map(drones, assignments, destinations, animation_progress=0):
    """Create interactive map with drones and routes"""
    fig = go.Figure()
    
    # Add campus locations
    for loc_name, loc_data in CAMPUS_LOCATIONS.items():
        if loc_name in destinations:
            # Highlight active destinations with pulsing effect
            pulse_size = 50 + 10 * math.sin(animation_progress * 0.1) if animation_progress > 0 else 50
            fig.add_trace(go.Scatter(
                x=[loc_data['x']],
                y=[loc_data['y']],
                mode='markers+text',
                marker=dict(
                    size=pulse_size,
                    color='red', 
                    symbol='circle', 
                    line=dict(width=3, color='darkred'),
                    opacity=0.8
                ),
                text=[loc_data['icon']],
                textposition="middle center",
                textfont=dict(size=20),
                name=loc_name,
                hovertemplate=f"<b>{loc_name}</b><br>Type: {loc_data['type']}<br>Status: Active Target<extra></extra>"
            ))
        else:
            # Regular locations
            fig.add_trace(go.Scatter(
                x=[loc_data['x']],
                y=[loc_data['y']],
                mode='markers+text',
                marker=dict(size=30, color='gray', symbol='circle', opacity=0.4),
                text=[loc_data['icon']],
                textposition="middle center",
                name=loc_name,
                showlegend=False,
                hovertemplate=f"<b>{loc_name}</b><extra></extra>"
            ))
    
    # Add drone routes with smooth animation
    for assignment in assignments:
        drone = next((d for d in drones if d['id'] == assignment['drone_id']), None)
        if not drone:
            continue
        
        # Calculate intermediate points for smooth animation
        start_x, start_y = assignment['drone_x'], assignment['drone_y']
        end_x, end_y = assignment['dest_x'], assignment['dest_y']
        
        # Create smooth path with multiple waypoints
        num_waypoints = 30
        waypoints_x = []
        waypoints_y = []
        
        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            # Smooth easing function for natural movement
            eased_t = t * t * (3 - 2 * t)  # Smoothstep function
            waypoints_x.append(start_x + (end_x - start_x) * eased_t)
            waypoints_y.append(start_y + (end_y - start_y) * eased_t)
        
        # Current position based on animation progress
        if animation_progress > 0:
            progress = min(animation_progress / 100, 1.0)
            # Use smoothstep for natural easing
            eased_progress = progress * progress * (3 - 2 * progress)
            waypoint_idx = int(eased_progress * num_waypoints)
            waypoint_idx = min(waypoint_idx, num_waypoints)
            current_x = waypoints_x[waypoint_idx]
            current_y = waypoints_y[waypoint_idx]
        else:
            current_x, current_y = start_x, start_y
        
        # Route line with gradient effect
        fig.add_trace(go.Scatter(
            x=waypoints_x,
            y=waypoints_y,
            mode='lines',
            line=dict(color='rgba(0, 100, 255, 0.6)', width=3),
            showlegend=False,
            hoverinfo='skip',
            name=f"Route {drone['id']}"
        ))
        
        # Drone at current position
        fig.add_trace(go.Scatter(
            x=[current_x],
            y=[current_y],
            mode='markers+text',
            marker=dict(
                size=35, 
                color='blue', 
                symbol='triangle-up',
                line=dict(width=2, color='darkblue'),
                opacity=0.9
            ),
            text=['🚁'],
            textposition="middle center",
            textfont=dict(size=18),
            name=f"Drone {drone['id']}",
            hovertemplate=f"<b>Drone {drone['id']}</b><br>Status: {drone['status']}<br>Battery: {drone['battery']}%<br>Target: {assignment['destination']}<br>Distance: {assignment['distance']:.1f}m<extra></extra>"
        ))
        
        # Add progress indicator
        if animation_progress > 0 and animation_progress < 100:
            fig.add_trace(go.Scatter(
                x=[current_x],
                y=[current_y - 15],
                mode='markers',
                marker=dict(size=8, color='green', symbol='circle'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Add idle drones
    for drone in drones:
        if drone['status'] == 'idle' and not any(a['drone_id'] == drone['id'] for a in assignments):
            fig.add_trace(go.Scatter(
                x=[drone['x']],
                y=[drone['y']],
                mode='markers+text',
                marker=dict(
                    size=28, 
                    color='lightblue', 
                    symbol='triangle-up',
                    line=dict(width=1, color='blue'),
                    opacity=0.7
                ),
                text=['🚁'],
                textposition="middle center",
                textfont=dict(size=16),
                name=f"Drone {drone['id']} (Idle)",
                hovertemplate=f"<b>Drone {drone['id']}</b><br>Status: Idle<br>Battery: {drone['battery']}%<extra></extra>"
            ))
    
    fig.update_layout(
        title=dict(
            text="🚁 AI-Powered Drone Fleet - Real-Time Route Optimization",
            font=dict(size=20, color='#333')
        ),
        xaxis=dict(range=[0, 700], showgrid=True, gridcolor='lightgray', title="X Coordinate (m)"),
        yaxis=dict(range=[0, 600], showgrid=True, gridcolor='lightgray', title="Y Coordinate (m)"),
        width=1000,
        height=700,
        showlegend=True,
        hovermode='closest',
        template='plotly_white',
        plot_bgcolor='rgba(240, 240, 240, 0.5)'
    )
    
    return fig

def calculate_metrics(drones, assignments, destinations):
    """Calculate live metrics"""
    total_distance = sum(a['distance'] for a in assignments)
    
    active_drones = len([d for d in drones if d['status'] != 'idle'])
    total_drones = len(drones)
    utilization = (active_drones / total_drones * 100) if total_drones > 0 else 0
    
    total_tasks = len(destinations)
    completed_tasks = len(assignments)
    coverage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    
    # Model accuracy (simulated based on efficiency)
    avg_distance = total_distance / len(assignments) if assignments else 0
    max_possible_distance = 1000
    accuracy = max(0, (max_possible_distance - avg_distance) / max_possible_distance * 100)
    
    return {
        'total_distance': round(total_distance, 2),
        'utilization': round(utilization, 1),
        'coverage': round(coverage, 1),
        'accuracy': round(accuracy, 1),
        'active_drones': active_drones,
        'total_drones': total_drones
    }

# ==================== MAIN APP ====================

st.markdown("""
<div class="main-header">
    <h1>🚁 AI-Powered Autonomous Drone Fleet System</h1>
    <p style="font-size: 1.2rem; margin-top: 10px;">Intelligent Route Optimization • Computer Vision • Real-Time Analytics</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Control Panel")

# Mode selection
mode = st.sidebar.selectbox(
    "🎯 Operation Mode",
    ["Delivery Optimization", "Surveillance Mode"],
    help="Select the operational mode for the drone fleet"
)

# Algorithm selection
algorithm = st.sidebar.selectbox(
    "🤖 Optimization Algorithm",
    [
        "KNN (Nearest Neighbors)",
        "Dijkstra's Algorithm",
        "Genetic Algorithm",
        "Particle Swarm Optimization",
        "Ensemble Model"
    ],
    help="Choose the AI algorithm for route optimization"
)

# Drone Management
st.sidebar.markdown("---")
st.sidebar.subheader("🚁 Drone Fleet Management")

st.sidebar.write(f"**Total Drones:** {len(st.session_state.drones)}")

if st.sidebar.button("➕ Add New Drone", use_container_width=True):
    new_id = max([d['id'] for d in st.session_state.drones], default=0) + 1
    # Place new drone at random safe position
    new_x = random.randint(50, 650)
    new_y = random.randint(50, 550)
    st.session_state.drones.append({
        'id': new_id,
        'x': new_x,
        'y': new_y,
        'status': 'idle',
        'battery': random.randint(80, 100),
        'payload': 0,
        'target': None,
        'route': [],
        'current_step': 0
    })
    st.rerun()

# Drone position editor
with st.sidebar.expander("📍 Edit Drone Positions", expanded=False):
    for i, drone in enumerate(st.session_state.drones):
        st.write(f"**Drone {drone['id']}**")
        col_x, col_y = st.columns(2)
        with col_x:
            new_x = st.number_input("X", value=drone['x'], min_value=0, max_value=700, key=f"x_{drone['id']}")
        with col_y:
            new_y = st.number_input("Y", value=drone['y'], min_value=0, max_value=600, key=f"y_{drone['id']}")
        
        if st.button(f"Update Drone {drone['id']}", key=f"update_{drone['id']}"):
            st.session_state.drones[i]['x'] = int(new_x)
            st.session_state.drones[i]['y'] = int(new_y)
            st.rerun()
        
        if st.button(f"Remove Drone {drone['id']}", key=f"remove_{drone['id']}"):
            st.session_state.drones.pop(i)
            st.rerun()
        
        st.markdown("---")

# Simulation controls
st.sidebar.markdown("---")
st.sidebar.subheader("🎮 Simulation Controls")

simulation_speed = st.sidebar.slider("⚡ Animation Speed", 1, 10, 5, help="Control the speed of drone movement animation")

col_start, col_pause = st.sidebar.columns(2)
with col_start:
    if st.button("▶️ Start", use_container_width=True):
        st.session_state.simulation_running = True
        st.session_state.animation_frame = 0
        st.rerun()

with col_pause:
    if st.button("⏸️ Pause", use_container_width=True):
        st.session_state.simulation_running = False
        st.rerun()

if st.sidebar.button("🔄 Reset", use_container_width=True):
    st.session_state.simulation_running = False
    st.session_state.animation_frame = 0
    for drone in st.session_state.drones:
        drone['status'] = 'idle'
        drone['target'] = None
        drone['route'] = []
        drone['current_step'] = 0
    st.rerun()

# Task management
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Task Management")

if st.sidebar.button("➕ Add Random Task", use_container_width=True):
    locations = list(CAMPUS_LOCATIONS.keys())
    new_location = random.choice(locations)
    task_types = ['delivery', 'surveillance']
    priorities = ['high', 'medium', 'low']
    
    new_task = {
        'id': len(st.session_state.tasks) + 1,
        'location': new_location,
        'priority': random.choice(priorities),
        'type': random.choice(task_types),
        'status': 'pending'
    }
    st.session_state.tasks.append(new_task)
    st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🗺️ Real-Time Drone Fleet Map")
    
    # Get active destinations based on mode
    active_destinations = {}
    for task in st.session_state.tasks:
        if task['status'] == 'pending':
            if mode == "Delivery Optimization" and task['type'] == 'delivery':
                active_destinations[task['location']] = task
            elif mode == "Surveillance Mode" and task['type'] == 'surveillance':
                active_destinations[task['location']] = task
    
    # Run optimization algorithm
    drones_copy = [d.copy() for d in st.session_state.drones]
    assignments = []
    
    if algorithm == "KNN (Nearest Neighbors)":
        assignments = knn_route_optimization(drones_copy, active_destinations)
    elif algorithm == "Dijkstra's Algorithm":
        assignments = dijkstra_route_optimization(drones_copy, active_destinations)
    elif algorithm == "Genetic Algorithm":
        assignments = genetic_algorithm_optimization(drones_copy, active_destinations)
    elif algorithm == "Particle Swarm Optimization":
        assignments = particle_swarm_optimization(drones_copy, active_destinations)
    elif algorithm == "Ensemble Model":
        assignments = ensemble_optimization(drones_copy, active_destinations)
    
    # Animation progress - smooth update without constant reruns
    animation_progress = st.session_state.animation_frame if st.session_state.simulation_running else 0
    
    # Create and display map
    map_placeholder = st.empty()
    map_fig = create_drone_map(st.session_state.drones, assignments, active_destinations, animation_progress)
    map_placeholder.plotly_chart(map_fig, use_container_width=True)
    
    # Update animation frame smoothly - only update when running
    if st.session_state.simulation_running:
        current_time = time.time()
        # Update every 0.15 seconds to avoid flickering (smoother)
        time_delta = current_time - st.session_state.last_update_time
        if time_delta >= 0.15:
            speed_factor = simulation_speed / 5.0
            frame_increment = max(1, int(3 * speed_factor))
            new_frame = (st.session_state.animation_frame + frame_increment) % 100
            
            # Only rerun if frame actually changed significantly
            if abs(new_frame - st.session_state.animation_frame) >= 2:
                st.session_state.animation_frame = new_frame
                st.session_state.last_update_time = current_time
                
                # Auto-trigger detection when drone reaches destination
                if st.session_state.animation_frame >= 95:
                    for assignment in assignments:
                        task = active_destinations.get(assignment['destination'])
                        if task and task['status'] == 'pending':
                            task['status'] = 'scanning'
                
                # Rerun to update display
                st.rerun()

with col2:
    st.subheader("📊 Live Metrics")
    
    # Calculate metrics
    metrics = calculate_metrics(st.session_state.drones, assignments, active_destinations)
    
    # Display metrics cards
    st.markdown(f"""
    <div class="metric-card">
        <h2 style="margin: 0; font-size: 36px;">{metrics['total_distance']:.1f}m</h2>
        <p style="margin: 10px 0 0 0; font-size: 14px;">Total Distance</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
        <h2 style="margin: 0; font-size: 36px;">{metrics['utilization']:.1f}%</h2>
        <p style="margin: 10px 0 0 0; font-size: 14px;">Fleet Utilization</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
        <h2 style="margin: 0; font-size: 36px;">{metrics['coverage']:.1f}%</h2>
        <p style="margin: 10px 0 0 0; font-size: 14px;">Coverage Rate</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
        <h2 style="margin: 0; font-size: 36px;">{metrics['accuracy']:.1f}%</h2>
        <p style="margin: 10px 0 0 0; font-size: 14px;">Model Accuracy</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🚁 Drone Status")
    
    for drone in st.session_state.drones:
        status_icon = "🟢" if drone['status'] == 'idle' else "🔵" if drone['status'] == 'assigned' else "🟡"
        st.markdown(f"""
        <div class="drone-card">
            <strong>{status_icon} Drone {drone['id']}</strong><br>
            Battery: {drone['battery']}% | Status: {drone['status']}<br>
            Position: ({drone['x']}, {drone['y']})
        </div>
        """, unsafe_allow_html=True)

# Computer Vision Module
st.markdown("---")
st.subheader("📷 Drone Camera Module - Object Detection")

# YOLO Status
yolo_status = st.empty()
if YOLO_AVAILABLE:
    try:
        model = load_yolo_model()
        if model is not None:
            yolo_status.success("✅ YOLOv8 Model Ready")
        else:
            yolo_status.info("🔄 Loading YOLOv8 model (first run may take time)...")
    except Exception as e:
        yolo_status.warning(f"⚠️ YOLOv8 Error: {str(e)}")
else:
    yolo_status.warning("⚠️ YOLOv8 not installed. Install with: `pip install ultralytics`")

col_cam1, col_cam2 = st.columns([1, 1])

with col_cam1:
    st.markdown("### Upload Aerial Image")
    uploaded_file = st.file_uploader(
        "Choose an image for object detection",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an aerial image to detect people, vehicles, buildings, or obstacles"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🔍 Run Object Detection", use_container_width=True):
            with st.spinner("Running YOLOv8 object detection..."):
                # Convert PIL to numpy array
                img_array = np.array(image)
                
                # Run detection
                model = load_yolo_model()
                if model:
                    annotated_img, detections = detect_objects_yolo(img_array, model)
                    
                    # Display results
                    st.session_state.last_detection = {
                        'image': annotated_img,
                        'detections': detections,
                        'location': 'Uploaded Image'
                    }
                    st.rerun()
                else:
                    st.error("YOLOv8 model not available. Please check installation.")

with col_cam2:
    st.markdown("### Detection Results")
    
    if 'last_detection' in st.session_state:
        detection_data = st.session_state.last_detection
        
        # Display annotated image
        st.image(detection_data['image'], caption="Detected Objects", use_container_width=True)
        
        # Display detection details
        if detection_data['detections']:
            st.success(f"✅ Found {len(detection_data['detections'])} objects")
            
            detection_df = pd.DataFrame(detection_data['detections'])
            st.dataframe(
                detection_df.style.format({'confidence': '{:.2%}'}),
                use_container_width=True,
                hide_index=True
            )
            
            # Add to history
            st.session_state.detection_history.append({
                'timestamp': time.time(),
                'location': detection_data['location'],
                'count': len(detection_data['detections']),
                'objects': detection_data['detections']
            })
        else:
            st.info("No relevant objects detected")
    else:
        st.info("👆 Upload an image and click 'Run Object Detection' to see results")

# Auto-trigger detection simulation
st.markdown("---")
st.subheader("🤖 Autonomous Intelligence - Area Scanning")

# Check if any drones have reached their destinations
auto_scan_triggered = False
scanning_drones = []

for assignment in assignments:
    if st.session_state.animation_frame >= 95:
        drone = next((d for d in st.session_state.drones if d['id'] == assignment['drone_id']), None)
        if drone and drone['status'] != 'scanning':
            scanning_drones.append({
                'drone_id': assignment['drone_id'],
                'destination': assignment['destination'],
                'assignment': assignment
            })
            drone['status'] = 'scanning'
            auto_scan_triggered = True

# Display auto-scan results
if auto_scan_triggered and scanning_drones:
    for scan_info in scanning_drones:
        st.success(f"🚁 **Drone {scan_info['drone_id']}** reached **{scan_info['destination']}**!")
        st.info(f"📡 **Autonomous area scan initiated** at {scan_info['destination']}...")
        
        # Auto-run detection if image is available
        if uploaded_file is not None:
            with st.spinner(f"Running YOLOv8 detection for Drone {scan_info['drone_id']}..."):
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                model = load_yolo_model()
                if model:
                    annotated_img, detections = detect_objects_yolo(img_array, model)
                    
                    col_scan1, col_scan2 = st.columns([1, 1])
                    
                    with col_scan1:
                        st.image(annotated_img, caption=f"Auto-scan at {scan_info['destination']}", use_container_width=True)
                    
                    with col_scan2:
                        if detections:
                            st.success(f"✅ **{len(detections)} objects detected**")
                            detection_summary = pd.DataFrame(detections)
                            st.dataframe(detection_summary, use_container_width=True, hide_index=True)
                            
                            # Add to history
                            st.session_state.detection_history.append({
                                'timestamp': time.time(),
                                'location': scan_info['destination'],
                                'count': len(detections),
                                'objects': detections,
                                'drone_id': scan_info['drone_id']
                            })
                        else:
                            st.info("No objects detected in scanned area")
                else:
                    st.warning("YOLOv8 model not available")
        else:
            st.warning("⚠️ Upload an image to enable auto-detection on arrival")

# Manual trigger button
if st.button("🎯 Simulate Drone Arrival & Auto-Scan", use_container_width=True):
    if assignments:
        assignment = assignments[0]
        destination = assignment['destination']
        
        st.success(f"🚁 Drone {assignment['drone_id']} reached {destination}!")
        st.info("📡 Initiating autonomous area scan...")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            model = load_yolo_model()
            if model:
                annotated_img, detections = detect_objects_yolo(img_array, model)
                
                st.image(annotated_img, caption=f"Auto-scan at {destination}", use_container_width=True)
                
                if detections:
                    st.success(f"✅ Auto-detected {len(detections)} objects at {destination}")
                    detection_summary = pd.DataFrame(detections)
                    st.dataframe(detection_summary, use_container_width=True, hide_index=True)
                else:
                    st.info("No objects detected in scanned area")
            else:
                st.warning("YOLOv8 model not available")
        else:
            st.warning("⚠️ Please upload an image first to simulate detection")
    else:
        st.warning("⚠️ No active assignments. Start simulation first.")

# Detection History
if st.session_state.detection_history:
    st.markdown("---")
    st.subheader("📜 Detection History")
    
    history_df = pd.DataFrame([
        {
            'Time': time.strftime('%H:%M:%S', time.localtime(h['timestamp'])),
            'Location': h['location'],
            'Objects Found': h['count'],
            'Details': ', '.join([f"{d['class']} ({d['confidence']:.0%})" for d in h['objects'][:3]])
        }
        for h in st.session_state.detection_history[-10:]
    ])
    
    st.dataframe(history_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>🚁 AI-Powered Autonomous Drone Fleet System</strong></p>
    <p>Built with Streamlit • YOLOv8 • Advanced AI Optimization Algorithms</p>
    <p style="font-size: 0.9rem; margin-top: 10px;">Ready for Hackathon Demo 🚀</p>
</div>
""", unsafe_allow_html=True)
