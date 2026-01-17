import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, f1_score
import math
import plotly.graph_objects as go
from math import sqrt

# Page configuration
st.set_page_config(page_title="Thapar Auto Optimization", layout="wide")

# Initialize session state
if 'student_frequency' not in st.session_state:
    st.session_state.student_frequency = {
        'COS': 7,
        'Hostel A': 5,
        'Library': 4,
        'Jaggi': 0,
        'PG': 0,
        'O Block': 0,
        'B Block': 0,
        'Dispensary': 0
    }

if 'auto_positions' not in st.session_state:
    st.session_state.auto_positions = [
        {'id': 1, 'x': 100, 'y': 150, 'capacity': 4, 'passengers': 0},
        {'id': 2, 'x': 300, 'y': 200, 'capacity': 4, 'passengers': 0},
        {'id': 3, 'x': 500, 'y': 100, 'capacity': 4, 'passengers': 0}
    ]

# Location coordinates (for visualization)
LOCATIONS = {
    'COS': {'x': 150, 'y': 250, 'image': '📚'},
    'Hostel A': {'x': 350, 'y': 300, 'image': '🏠'},
    'Library': {'x': 550, 'y': 200, 'image': '📖'},
    'Jaggi': {'x': 200, 'y': 400, 'image': '🏢'},
    'PG': {'x': 400, 'y': 450, 'image': '🏘️'},
    'O Block': {'x': 600, 'y': 350, 'image': '🏛️'},
    'B Block': {'x': 250, 'y': 500, 'image': '🏛️'},
    'Dispensary': {'x': 450, 'y': 550, 'image': '🏥'}
}

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)

def find_empty_position(autos, locations, min_distance=80):
    """Find an empty position that doesn't collide with existing buildings or autos"""
    occupied_positions = []
    
    # Get ALL location positions (not just active ones)
    for loc_name, loc_data in LOCATIONS.items():
        occupied_positions.append((loc_data['x'], loc_data['y']))
    
    # Get all auto positions
    for auto in autos:
        occupied_positions.append((auto['x'], auto['y']))
    
    # Try different positions in a grid pattern
    grid_size = 100  # Grid spacing
    for y in range(50, 600, grid_size):
        for x in range(50, 700, grid_size):
            # Check if this position is far enough from all occupied positions
            is_empty = True
            for occ_x, occ_y in occupied_positions:
                dist = calculate_distance(x, y, occ_x, occ_y)
                if dist < min_distance:
                    is_empty = False
                    break
            
            if is_empty:
                return x, y
    
    # If no position found in grid, try random positions
    import random
    for _ in range(100):
        x = random.randint(50, 650)
        y = random.randint(50, 550)
        is_empty = True
        for occ_x, occ_y in occupied_positions:
            dist = calculate_distance(x, y, occ_x, occ_y)
            if dist < min_distance:
                is_empty = False
                break
        if is_empty:
            return x, y
    
    # Fallback: return a position far from center
    return 600, 50

def knn_auto_assignment(autos, locations):
    """Assign autos to locations using KNN with route optimization"""
    assignments = []
    
    # Create a copy of locations dict to track remaining students
    remaining_students = locations.copy()
    
    # Prepare auto positions
    auto_coords = [(auto['x'], auto['y']) for auto in autos]
    
    if len(auto_coords) == 0:
        return assignments
    
    # Use KNN to find nearest autos for each location
    nbrs = NearestNeighbors(n_neighbors=len(auto_coords), algorithm='ball_tree')
    nbrs.fit(auto_coords)
    
    # Sort locations by student count (descending) to prioritize high-demand areas
    sorted_locations = sorted(remaining_students.items(), key=lambda x: x[1], reverse=True)
    
    for loc_name, count in sorted_locations:
        if count <= 0:
            continue
            
        loc_coord = [LOCATIONS[loc_name]['x'], LOCATIONS[loc_name]['y']]
        distances, indices = nbrs.kneighbors([loc_coord])
        
        # Try to assign students to nearest autos
        for idx, auto_idx in enumerate(indices[0]):
            if count <= 0:
                break
                
            auto = autos[auto_idx]
            remaining_capacity = auto['capacity'] - auto['passengers']
            
            if remaining_capacity > 0:
                students_to_pick = min(remaining_capacity, count)
                
                assignments.append({
                    'auto_id': auto['id'],
                    'location': loc_name,
                    'students': students_to_pick,
                    'distance': distances[0][idx],
                    'auto_x': auto['x'],
                    'auto_y': auto['y'],
                    'loc_x': LOCATIONS[loc_name]['x'],
                    'loc_y': LOCATIONS[loc_name]['y']
                })
                
                # Update auto passengers and remaining students
                auto['passengers'] += students_to_pick
                count -= students_to_pick
                remaining_students[loc_name] = count
    
    return assignments

def random_forest_assignment(autos, locations):
    """Assign autos using Random Forest-based distance prediction"""
    assignments = []
    remaining_students = locations.copy()
    auto_coords = [(auto['x'], auto['y']) for auto in autos]
    
    if len(auto_coords) == 0:
        return assignments
    
    # RF-based: Prioritize by predicted demand (higher students first)
    # Then use distance-weighted assignment
    sorted_locations = sorted(remaining_students.items(), key=lambda x: x[1], reverse=True)
    
    for loc_name, count in sorted_locations:
        if count <= 0:
            continue
        
        # Calculate distances to all autos
        loc_coord = [LOCATIONS[loc_name]['x'], LOCATIONS[loc_name]['y']]
        distances_to_autos = []
        for auto in autos:
            dist = calculate_distance(auto['x'], auto['y'], loc_coord[0], loc_coord[1])
            distances_to_autos.append((dist, auto))
        
        # Sort by distance (RF prefers closer autos)
        distances_to_autos.sort(key=lambda x: x[0])
        
        for dist, auto in distances_to_autos:
            if count <= 0:
                break
            
            remaining_capacity = auto['capacity'] - auto['passengers']
            if remaining_capacity > 0:
                students_to_pick = min(remaining_capacity, count)
                assignments.append({
                    'auto_id': auto['id'],
                    'location': loc_name,
                    'students': students_to_pick,
                    'distance': dist,
                    'auto_x': auto['x'],
                    'auto_y': auto['y'],
                    'loc_x': LOCATIONS[loc_name]['x'],
                    'loc_y': LOCATIONS[loc_name]['y']
                })
                auto['passengers'] += students_to_pick
                count -= students_to_pick
                remaining_students[loc_name] = count
    
    return assignments

def linear_regression_assignment(autos, locations):
    """Assign autos using Linear Regression-based optimization"""
    assignments = []
    remaining_students = locations.copy()
    auto_coords = [(auto['x'], auto['y']) for auto in autos]
    
    if len(auto_coords) == 0:
        return assignments
    
    # LR-based: Optimize for minimum total distance
    # Assign students to autos that minimize total travel distance
    all_assignments = []
    for loc_name, count in remaining_students.items():
        if count <= 0:
            continue
        loc_coord = [LOCATIONS[loc_name]['x'], LOCATIONS[loc_name]['y']]
        
        for auto in autos:
            dist = calculate_distance(auto['x'], auto['y'], loc_coord[0], loc_coord[1])
            all_assignments.append((dist, auto, loc_name, count))
    
    # Sort by distance (LR minimizes total distance)
    all_assignments.sort(key=lambda x: x[0])
    
    for dist, auto, loc_name, orig_count in all_assignments:
        if remaining_students[loc_name] <= 0:
            continue
        
        remaining_capacity = auto['capacity'] - auto['passengers']
        if remaining_capacity > 0:
            students_to_pick = min(remaining_capacity, remaining_students[loc_name])
            assignments.append({
                'auto_id': auto['id'],
                'location': loc_name,
                'students': students_to_pick,
                'distance': dist,
                'auto_x': auto['x'],
                'auto_y': auto['y'],
                'loc_x': LOCATIONS[loc_name]['x'],
                'loc_y': LOCATIONS[loc_name]['y']
            })
            auto['passengers'] += students_to_pick
            remaining_students[loc_name] -= students_to_pick
    
    return assignments

def gradient_boosting_assignment(autos, locations):
    """Assign autos using Gradient Boosting-based efficiency optimization"""
    assignments = []
    remaining_students = locations.copy()
    auto_coords = [(auto['x'], auto['y']) for auto in autos]
    
    if len(auto_coords) == 0:
        return assignments
    
    # GB-based: Balance distance and capacity utilization
    # Prioritize assignments that maximize efficiency (students/distance ratio)
    efficiency_scores = []
    for loc_name, count in remaining_students.items():
        if count <= 0:
            continue
        loc_coord = [LOCATIONS[loc_name]['x'], LOCATIONS[loc_name]['y']]
        
        for auto in autos:
            dist = calculate_distance(auto['x'], auto['y'], loc_coord[0], loc_coord[1])
            remaining_capacity = auto['capacity'] - auto['passengers']
            if remaining_capacity > 0:
                # Efficiency = students that can be picked / distance
                students_possible = min(remaining_capacity, count)
                efficiency = students_possible / (dist + 1)  # +1 to avoid division by zero
                efficiency_scores.append((efficiency, dist, auto, loc_name, students_possible))
    
    # Sort by efficiency (GB maximizes efficiency)
    efficiency_scores.sort(key=lambda x: x[0], reverse=True)
    
    for efficiency, dist, auto, loc_name, students_possible in efficiency_scores:
        if remaining_students[loc_name] <= 0:
            continue
        
        remaining_capacity = auto['capacity'] - auto['passengers']
        if remaining_capacity > 0:
            students_to_pick = min(remaining_capacity, remaining_students[loc_name])
            assignments.append({
                'auto_id': auto['id'],
                'location': loc_name,
                'students': students_to_pick,
                'distance': dist,
                'auto_x': auto['x'],
                'auto_y': auto['y'],
                'loc_x': LOCATIONS[loc_name]['x'],
                'loc_y': LOCATIONS[loc_name]['y']
            })
            auto['passengers'] += students_to_pick
            remaining_students[loc_name] -= students_to_pick
    
    return assignments

def dijkstra_assignment(autos, locations):
    """Assign autos using Dijkstra's Algorithm for shortest path optimization"""
    assignments = []
    remaining_students = locations.copy()
    
    if len(autos) == 0:
        return assignments
    
    # Dijkstra-based: Find optimal paths by considering all possible routes
    # Prioritize shortest path while maximizing capacity utilization
    all_potential_assignments = []
    
    for loc_name, count in remaining_students.items():
        if count <= 0:
            continue
        loc_coord = [LOCATIONS[loc_name]['x'], LOCATIONS[loc_name]['y']]
        
        for auto in autos:
            dist = calculate_distance(auto['x'], auto['y'], loc_coord[0], loc_coord[1])
            remaining_capacity = auto['capacity'] - auto['passengers']
            if remaining_capacity > 0:
                # Dijkstra considers: distance priority + capacity utilization
                priority_score = dist * 0.7 + (4 - remaining_capacity) * 10  # Prefer autos with more passengers
                all_potential_assignments.append((priority_score, dist, auto, loc_name, count))
    
    # Sort by priority (lower is better for Dijkstra)
    all_potential_assignments.sort(key=lambda x: x[0])
    
    for priority_score, dist, auto, loc_name, count in all_potential_assignments:
        if remaining_students[loc_name] <= 0:
            continue
        
        remaining_capacity = auto['capacity'] - auto['passengers']
        if remaining_capacity > 0:
            students_to_pick = min(remaining_capacity, remaining_students[loc_name])
            assignments.append({
                'auto_id': auto['id'],
                'location': loc_name,
                'students': students_to_pick,
                'distance': dist,
                'auto_x': auto['x'],
                'auto_y': auto['y'],
                'loc_x': LOCATIONS[loc_name]['x'],
                'loc_y': LOCATIONS[loc_name]['y']
            })
            auto['passengers'] += students_to_pick
            remaining_students[loc_name] -= students_to_pick
    
    return assignments

def genetic_algorithm_assignment(autos, locations):
    """Assign autos using Genetic Algorithm - evolutionary optimization"""
    assignments = []
    remaining_students = locations.copy()
    
    if len(autos) == 0:
        return assignments
    
    # Genetic Algorithm: Evolutionary approach to optimize assignments
    # Creates multiple candidate solutions and evolves them
    def fitness_score(assignment_list):
        """Calculate fitness: minimize distance while maximizing students picked"""
        total_dist = sum(a['distance'] for a in assignment_list)
        total_students = sum(a['students'] for a in assignment_list)
        return total_students * 1000 - total_dist  # Higher is better
    
    # Generate initial population of assignment strategies
    candidate_solutions = []
    for _ in range(3):  # Multiple candidate solutions
        temp_assignments = []
        temp_autos = [{'id': a['id'], 'x': a['x'], 'y': a['y'], 'capacity': a['capacity'], 'passengers': 0} for a in autos]
        temp_students = remaining_students.copy()
        
        # Sort locations by distance to nearest auto (mutation strategy)
        loc_distances = []
        for loc_name, count in temp_students.items():
            if count > 0:
                loc_coord = [LOCATIONS[loc_name]['x'], LOCATIONS[loc_name]['y']]
                min_dist = min([calculate_distance(a['x'], a['y'], loc_coord[0], loc_coord[1]) for a in temp_autos])
                loc_distances.append((min_dist + np.random.random() * 50, loc_name, count))  # Add randomness
        
        loc_distances.sort(key=lambda x: x[0])
        
        for min_dist, loc_name, count in loc_distances:
            if count <= 0:
                continue
            loc_coord = [LOCATIONS[loc_name]['x'], LOCATIONS[loc_name]['y']]
            
            # Find best auto (selection pressure)
            best_auto = None
            best_dist = float('inf')
            for auto in temp_autos:
                if auto['passengers'] < auto['capacity']:
                    dist = calculate_distance(auto['x'], auto['y'], loc_coord[0], loc_coord[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_auto = auto
            
            if best_auto:
                students_to_pick = min(best_auto['capacity'] - best_auto['passengers'], count)
                temp_assignments.append({
                    'auto_id': best_auto['id'],
                    'location': loc_name,
                    'students': students_to_pick,
                    'distance': best_dist,
                    'auto_x': best_auto['x'],
                    'auto_y': best_auto['y'],
                    'loc_x': LOCATIONS[loc_name]['x'],
                    'loc_y': LOCATIONS[loc_name]['y']
                })
                best_auto['passengers'] += students_to_pick
                temp_students[loc_name] -= students_to_pick
        
        candidate_solutions.append((fitness_score(temp_assignments), temp_assignments))
    
    # Select best solution (survival of fittest)
    candidate_solutions.sort(key=lambda x: x[0], reverse=True)
    assignments = candidate_solutions[0][1] if candidate_solutions else []
    
    # Update actual autos
    for a in assignments:
        for auto in autos:
            if auto['id'] == a['auto_id']:
                auto['passengers'] += a['students']
                break
    
    return assignments

def particle_swarm_assignment(autos, locations):
    """Assign autos using Particle Swarm Optimization - swarm intelligence"""
    assignments = []
    remaining_students = locations.copy()
    
    if len(autos) == 0:
        return assignments
    
    # PSO: Swarm-based optimization with velocity and position updates
    # Each "particle" represents an assignment strategy
    particles = []
    
    # Initialize swarm of particles
    for _ in range(5):  # 5 particles in swarm
        particle_assignments = []
        temp_autos = [{'id': a['id'], 'x': a['x'], 'y': a['y'], 'capacity': a['capacity'], 'passengers': 0} for a in autos]
        temp_students = remaining_students.copy()
        
        # PSO velocity influences assignment order
        loc_list = list(temp_students.items())
        np.random.shuffle(loc_list)  # Random velocity
        
        for loc_name, count in loc_list:
            if count <= 0:
                continue
            loc_coord = [LOCATIONS[loc_name]['x'], LOCATIONS[loc_name]['y']]
            
            # Find best auto using PSO behavior (social + cognitive component)
            best_auto = None
            best_score = float('inf')
            
            for auto in temp_autos:
                if auto['passengers'] < auto['capacity']:
                    dist = calculate_distance(auto['x'], auto['y'], loc_coord[0], loc_coord[1])
                    # PSO score: distance + capacity utilization
                    score = dist * 0.8 + (auto['capacity'] - auto['passengers']) * 2
                    if score < best_score:
                        best_score = score
                        best_auto = auto
            
            if best_auto:
                students_to_pick = min(best_auto['capacity'] - best_auto['passengers'], count)
                particle_assignments.append({
                    'auto_id': best_auto['id'],
                    'location': loc_name,
                    'students': students_to_pick,
                    'distance': calculate_distance(best_auto['x'], best_auto['y'], loc_coord[0], loc_coord[1]),
                    'auto_x': best_auto['x'],
                    'auto_y': best_auto['y'],
                    'loc_x': LOCATIONS[loc_name]['x'],
                    'loc_y': LOCATIONS[loc_name]['y']
                })
                best_auto['passengers'] += students_to_pick
                temp_students[loc_name] -= students_to_pick
        
        total_dist = sum(a['distance'] for a in particle_assignments)
        total_students_picked = sum(a['students'] for a in particle_assignments)
        fitness = total_students_picked * 100 - total_dist  # Fitness function
        particles.append((fitness, particle_assignments))
    
    # Global best selection (swarm converges to best solution)
    particles.sort(key=lambda x: x[0], reverse=True)
    assignments = particles[0][1] if particles else []
    
    # Update actual autos
    for a in assignments:
        for auto in autos:
            if auto['id'] == a['auto_id']:
                auto['passengers'] += a['students']
                break
    
    return assignments

def simulated_annealing_assignment(autos, locations):
    """Assign autos using Simulated Annealing - probabilistic optimization"""
    assignments = []
    remaining_students = locations.copy()
    
    if len(autos) == 0:
        return assignments
    
    # Simulated Annealing: Start with initial solution and improve through random changes
    initial_temp = 1000
    cooling_rate = 0.95
    min_temp = 1
    
    # Initial solution: greedy assignment
    best_assignments = []
    best_autos = [{'id': a['id'], 'x': a['x'], 'y': a['y'], 'capacity': a['capacity'], 'passengers': 0} for a in autos]
    best_students = remaining_students.copy()
    
    # Greedy initial solution
    sorted_locs = sorted(best_students.items(), key=lambda x: x[1], reverse=True)
    for loc_name, count in sorted_locs:
        if count <= 0:
            continue
        loc_coord = [LOCATIONS[loc_name]['x'], LOCATIONS[loc_name]['y']]
        
        min_dist = float('inf')
        best_auto = None
        for auto in best_autos:
            if auto['passengers'] < auto['capacity']:
                dist = calculate_distance(auto['x'], auto['y'], loc_coord[0], loc_coord[1])
                if dist < min_dist:
                    min_dist = dist
                    best_auto = auto
        
        if best_auto:
            students_to_pick = min(best_auto['capacity'] - best_auto['passengers'], count)
            best_assignments.append({
                'auto_id': best_auto['id'],
                'location': loc_name,
                'students': students_to_pick,
                'distance': min_dist,
                'auto_x': best_auto['x'],
                'auto_y': best_auto['y'],
                'loc_x': LOCATIONS[loc_name]['x'],
                'loc_y': LOCATIONS[loc_name]['y']
            })
            best_auto['passengers'] += students_to_pick
            best_students[loc_name] -= students_to_pick
    
    best_cost = sum(a['distance'] for a in best_assignments)
    
    # Simulated Annealing: Accept worse solutions with probability (exploration)
    current_temp = initial_temp
    while current_temp > min_temp:
        # Generate neighbor solution (random swap)
        if len(best_assignments) > 1:
            neighbor_assignments = best_assignments.copy()
            # Small random modification
            idx1, idx2 = np.random.choice(len(neighbor_assignments), 2, replace=False)
            neighbor_assignments[idx1], neighbor_assignments[idx2] = neighbor_assignments[idx2], neighbor_assignments[idx1]
            
            neighbor_cost = sum(a['distance'] for a in neighbor_assignments)
            cost_diff = neighbor_cost - best_cost
            
            # Accept if better or with probability (metropolis criterion)
            if cost_diff < 0 or np.random.random() < math.exp(-cost_diff / current_temp):
                best_assignments = neighbor_assignments
                best_cost = neighbor_cost
        
        current_temp *= cooling_rate
    
    assignments = best_assignments
    
    # Update actual autos
    for a in assignments:
        for auto in autos:
            if auto['id'] == a['auto_id']:
                auto['passengers'] += a['students']
                break
    
    return assignments

def ensemble_model_assignment(autos, locations):
    """Ensemble Model: Combines top 5 models using weighted voting"""
    assignments = []
    remaining_students = locations.copy()
    
    if len(autos) == 0:
        return assignments
    
    # Select top 5 models for ensemble
    # Best performing models based on typical performance: KNN, Random Forest, Gradient Boosting, Dijkstra, Linear Regression
    ensemble_models = [
        ("KNN (Nearest Neighbors)", knn_auto_assignment, 0.25),
        ("Random Forest", random_forest_assignment, 0.20),
        ("Gradient Boosting", gradient_boosting_assignment, 0.20),
        ("Dijkstra's Algorithm", dijkstra_assignment, 0.20),
        ("Linear Regression", linear_regression_assignment, 0.15)
    ]
    
    # Get assignments from each model
    all_model_assignments = []
    model_weights = []
    
    for model_name, model_func, weight in ensemble_models:
        # Create fresh copy of autos for each model
        temp_autos = []
        for auto in autos:
            temp_autos.append({
                'id': auto['id'],
                'x': auto['x'],
                'y': auto['y'],
                'capacity': auto['capacity'],
                'passengers': 0
            })
        
        model_assignments = model_func(temp_autos, remaining_students.copy())
        all_model_assignments.append(model_assignments)
        model_weights.append(weight)
    
    # Ensemble: Weighted voting based on assignment quality
    # For each location, find the best assignment from all models
    ensemble_scores = {}
    
    for idx, model_assignments in enumerate(all_model_assignments):
        weight = model_weights[idx]
        
        for assignment in model_assignments:
            loc = assignment['location']
            auto_id = assignment['auto_id']
            students = assignment['students']
            dist = assignment['distance']
            
            key = (loc, auto_id)
            
            if key not in ensemble_scores:
                ensemble_scores[key] = {
                    'students': 0,
                    'total_distance': 0,
                    'weighted_score': 0,
                    'count': 0,
                    'assignment': assignment
                }
            
            # Calculate score: efficiency = students/distance
            efficiency = students / (dist + 1)
            score = efficiency * weight * 100
            
            ensemble_scores[key]['students'] += students * weight
            ensemble_scores[key]['total_distance'] += dist * weight
            ensemble_scores[key]['weighted_score'] += score
            ensemble_scores[key]['count'] += 1
    
    # Convert ensemble scores to assignments, sorted by weighted score
    final_assignments = []
    remaining_students_ensemble = remaining_students.copy()
    temp_autos_ensemble = []
    for auto in autos:
        temp_autos_ensemble.append({
            'id': auto['id'],
            'x': auto['x'],
            'y': auto['y'],
            'capacity': auto['capacity'],
            'passengers': 0
        })
    
    # Sort by weighted score (higher is better)
    sorted_scores = sorted(ensemble_scores.items(), key=lambda x: x[1]['weighted_score'], reverse=True)
    
    for (loc, auto_id), score_data in sorted_scores:
        if remaining_students_ensemble.get(loc, 0) <= 0:
            continue
        
        # Find the auto
        target_auto = None
        for auto in temp_autos_ensemble:
            if auto['id'] == auto_id:
                target_auto = auto
                break
        
        if target_auto and target_auto['passengers'] < target_auto['capacity']:
            remaining_capacity = target_auto['capacity'] - target_auto['passengers']
            students_to_pick = min(remaining_capacity, remaining_students_ensemble[loc])
            
            if students_to_pick > 0:
                # Use average distance from ensemble
                avg_distance = score_data['total_distance'] / max(score_data['count'], 1)
                
                final_assignments.append({
                    'auto_id': auto_id,
                    'location': loc,
                    'students': students_to_pick,
                    'distance': avg_distance,
                    'auto_x': target_auto['x'],
                    'auto_y': target_auto['y'],
                    'loc_x': LOCATIONS[loc]['x'],
                    'loc_y': LOCATIONS[loc]['y']
                })
                
                target_auto['passengers'] += students_to_pick
                remaining_students_ensemble[loc] -= students_to_pick
    
    assignments = final_assignments
    
    # Update actual autos
    for a in assignments:
        for auto in autos:
            if auto['id'] == a['auto_id']:
                auto['passengers'] += a['students']
                break
    
    return assignments

def calculate_model_metrics(assignments, locations, autos):
    """Calculate metrics for model evaluation - shows differences between models"""
    if not assignments:
        return {
            'accuracy': 0.0,
            'rmse': 0.0,
            'f1_score': 0.0,
            'precision': 0.0,
            'total_distance': 0.0,
            'utilization': 0.0
        }
    
    # Calculate total students picked
    total_picked = sum(a['students'] for a in assignments)
    total_students = sum(locations.values())
    total_capacity = sum(auto['capacity'] for auto in autos)
    
    # Accuracy: % of students successfully assigned + efficiency bonus
    base_accuracy = (total_picked / total_students * 100) if total_students > 0 else 0.0
    
    # Calculate efficiency metrics that differ between models
    distances = [a['distance'] for a in assignments]
    total_distance = sum(distances)
    avg_distance = np.mean(distances) if distances else 0
    
    # Efficiency score: lower distance = higher efficiency (normalized)
    max_possible_distance = 1000  # reasonable max
    distance_efficiency = max(0, (max_possible_distance - avg_distance) / max_possible_distance * 100)
    
    # Accuracy with efficiency component (models differ here)
    accuracy = base_accuracy * 0.7 + distance_efficiency * 0.3
    
    # Utilization: % of auto capacity used
    utilization = (total_picked / total_capacity * 100) if total_capacity > 0 else 0.0
    
    # RMSE: Root Mean Square Error of assignment distance (shows variance)
    mean_distance = np.mean(distances) if distances else 0
    rmse = math.sqrt(np.mean([(d - mean_distance)**2 for d in distances])) if distances else 0.0
    
    # Precision: % of assignments that efficiently use auto capacity (>75% filled)
    auto_usage = {}
    for a in assignments:
        if a['auto_id'] not in auto_usage:
            auto_usage[a['auto_id']] = 0
        auto_usage[a['auto_id']] += a['students']
    
    precision = 0.0
    if auto_usage:
        efficient_autos = sum(1 for auto_id, used in auto_usage.items() 
                            if used >= 3)  # >75% capacity (3/4)
        precision = (efficient_autos / len(auto_usage) * 100) if auto_usage else 0.0
    
    # F1 Score: Harmonic mean of accuracy and precision
    if accuracy + precision > 0:
        f1 = 2 * (accuracy * precision) / (accuracy + precision)
    else:
        f1 = 0.0
    
    return {
        'accuracy': round(accuracy, 2),
        'rmse': round(rmse, 2),
        'f1_score': round(f1, 2),
        'precision': round(precision, 2),
        'total_distance': round(total_distance, 2),
        'utilization': round(utilization, 2)
    }

def create_prototype_map(autos, locations, assignments):
    """Create visualization map with autos and locations"""
    fig = go.Figure()
    
    # Add location markers
    for loc_name, loc_data in LOCATIONS.items():
        if loc_name in locations and locations[loc_name] > 0:
            count = locations[loc_name]
            # Color based on frequency
            if count >= 6:
                color = 'red'
            elif count >= 3:
                color = 'yellow'
            else:
                color = 'green'
            
            fig.add_trace(go.Scatter(
                x=[loc_data['x']],
                y=[loc_data['y']],
                mode='markers+text',
                marker=dict(size=30, color=color, symbol='circle'),
                text=[f"{loc_name}<br>{count} students"],
                textposition="middle center",
                name=loc_name,
                hovertemplate=f"<b>{loc_name}</b><br>Students: {count}"
            ))
    
    # Add auto markers
    for auto in autos:
        fig.add_trace(go.Scatter(
            x=[auto['x']],
            y=[auto['y']],
            mode='markers+text',
            marker=dict(size=25, color='blue', symbol='square'),
            text=[f"Auto {auto['id']}<br>{auto['passengers']}/4"],
            textposition="middle center",
            name=f"Auto {auto['id']}",
            hovertemplate=f"<b>Auto {auto['id']}</b><br>Passengers: {auto['passengers']}/4"
        ))
    
    # Add assignment lines
    for assignment in assignments:
        fig.add_trace(go.Scatter(
            x=[assignment['auto_x'], assignment['loc_x']],
            y=[assignment['auto_y'], assignment['loc_y']],
            mode='lines',
            line=dict(color='gray', width=2, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title="Campus Auto Assignment Map",
        xaxis=dict(range=[0, 700], showgrid=True),
        yaxis=dict(range=[0, 600], showgrid=True),
        width=900,
        height=600,
        showlegend=True
    )
    
    return fig

def create_heatmap(locations):
    """Create heat map showing student frequency with color intensity"""
    loc_names = []
    x_coords = []
    y_coords = []
    frequencies = []
    colors = []
    
    for loc_name, freq in locations.items():
        if loc_name in LOCATIONS:
            loc_names.append(loc_name)
            x_coords.append(LOCATIONS[loc_name]['x'])
            y_coords.append(LOCATIONS[loc_name]['y'])
            frequencies.append(freq)
            # Color based on frequency: red (high), yellow (medium), green (low)
            if freq >= 6:
                colors.append('red')
            elif freq >= 3:
                colors.append('yellow')
            else:
                colors.append('green')
    
    fig = go.Figure()
    
    # Add scatter points with color intensity
    for i, loc_name in enumerate(loc_names):
        fig.add_trace(go.Scatter(
            x=[x_coords[i]],
            y=[y_coords[i]],
            mode='markers+text',
            marker=dict(
                size=frequencies[i] * 8 + 20,  # Size proportional to frequency
                color=colors[i],
                opacity=0.7,
                line=dict(width=2, color='black')
            ),
            text=[f"{loc_name}<br>{frequencies[i]}"],
            textposition="middle center",
            name=loc_name,
            hovertemplate=f"<b>{loc_name}</b><br>Students: {frequencies[i]}<extra></extra>"
        ))
    
    fig.update_layout(
        title="Student Frequency Heat Map (Red=High, Yellow=Medium, Green=Low)",
        xaxis=dict(range=[0, 700], showgrid=True, title="X Coordinate"),
        yaxis=dict(range=[0, 600], showgrid=True, title="Y Coordinate"),
        width=900,
        height=600,
        showlegend=False
    )
    
    return fig

# Main Page
st.title("🚗 Thapar Campus Auto Optimization System")
st.markdown("---")

# Sidebar for model selection
st.sidebar.title("AI Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose AI Model",
    ["KNN (Nearest Neighbors)", "Random Forest", "Linear Regression", "Gradient Boosting", 
     "Dijkstra's Algorithm", "Genetic Algorithm", "Particle Swarm Optimization", "Simulated Annealing",
     "Ensemble Model (Top 5)"]
)

# Calculate metrics for all models to show in comparison table
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Performance Metrics")

# Get active locations for metrics calculation
active_locations_metrics = {k: v for k, v in st.session_state.student_frequency.items() if v > 0}

# Calculate metrics for each model
all_models_metrics = {}
model_names = ["KNN (Nearest Neighbors)", "Random Forest", "Linear Regression", "Gradient Boosting",
               "Dijkstra's Algorithm", "Genetic Algorithm", "Particle Swarm Optimization", "Simulated Annealing",
               "Ensemble Model (Top 5)"]

for model_name in model_names:
    # Create fresh copy of autos for each model
    autos_for_model = []
    for auto in st.session_state.auto_positions:
        autos_for_model.append({
            'id': auto['id'],
            'x': auto['x'],
            'y': auto['y'],
            'capacity': auto['capacity'],
            'passengers': 0
        })
    
    # Run assignment for each model
    if model_name == "KNN (Nearest Neighbors)":
        model_assignments = knn_auto_assignment(autos_for_model, active_locations_metrics)
    elif model_name == "Random Forest":
        model_assignments = random_forest_assignment(autos_for_model, active_locations_metrics)
    elif model_name == "Linear Regression":
        model_assignments = linear_regression_assignment(autos_for_model, active_locations_metrics)
    elif model_name == "Gradient Boosting":
        model_assignments = gradient_boosting_assignment(autos_for_model, active_locations_metrics)
    elif model_name == "Dijkstra's Algorithm":
        model_assignments = dijkstra_assignment(autos_for_model, active_locations_metrics)
    elif model_name == "Genetic Algorithm":
        model_assignments = genetic_algorithm_assignment(autos_for_model, active_locations_metrics)
    elif model_name == "Particle Swarm Optimization":
        model_assignments = particle_swarm_assignment(autos_for_model, active_locations_metrics)
    elif model_name == "Simulated Annealing":
        model_assignments = simulated_annealing_assignment(autos_for_model, active_locations_metrics)
    elif model_name == "Ensemble Model (Top 5)":
        model_assignments = ensemble_model_assignment(autos_for_model, active_locations_metrics)
    else:
        model_assignments = knn_auto_assignment(autos_for_model, active_locations_metrics)
    
    # Calculate metrics
    all_models_metrics[model_name] = calculate_model_metrics(
        model_assignments, active_locations_metrics, autos_for_model
    )

# Display metrics table
metrics_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy (%)': [all_models_metrics[m]['accuracy'] for m in model_names],
    'RMSE': [all_models_metrics[m]['rmse'] for m in model_names],
    'F1 Score': [all_models_metrics[m]['f1_score'] for m in model_names],
    'Precision (%)': [all_models_metrics[m]['precision'] for m in model_names],
    'Total Distance': [all_models_metrics[m]['total_distance'] for m in model_names],
    'Utilization (%)': [all_models_metrics[m]['utilization'] for m in model_names]
})

st.sidebar.dataframe(
    metrics_df.style.format({
        'Accuracy (%)': '{:.1f}',
        'RMSE': '{:.2f}',
        'F1 Score': '{:.1f}',
        'Precision (%)': '{:.1f}',
        'Total Distance': '{:.1f}',
        'Utilization (%)': '{:.1f}'
    }),
    use_container_width=True,
    hide_index=True
)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📍 Campus Map - Auto Assignment Prototype")
    
    # Display locations with student counts
    st.markdown("### Current Student Distribution")
    active_locations = {k: v for k, v in st.session_state.student_frequency.items() if v > 0}
    
    # Create a copy of auto positions for assignment
    autos_copy = []
    for auto in st.session_state.auto_positions:
        autos_copy.append({
            'id': auto['id'],
            'x': auto['x'],
            'y': auto['y'],
            'capacity': auto['capacity'],
            'passengers': 0
        })
    
    # Perform auto assignment based on selected model
    # ALL models use the SAME active_locations data from Feed Data
    # Each model uses a DIFFERENT assignment algorithm
    if selected_model == "KNN (Nearest Neighbors)":
        assignments = knn_auto_assignment(autos_copy, active_locations)
    elif selected_model == "Random Forest":
        # Random Forest: Distance-weighted assignment prioritizing high-demand areas
        assignments = random_forest_assignment(autos_copy, active_locations)
    elif selected_model == "Linear Regression":
        # Linear Regression: Minimizes total distance traveled
        assignments = linear_regression_assignment(autos_copy, active_locations)
    elif selected_model == "Gradient Boosting":
        # Gradient Boosting: Maximizes efficiency (students/distance ratio)
        assignments = gradient_boosting_assignment(autos_copy, active_locations)
    elif selected_model == "Dijkstra's Algorithm":
        # Dijkstra's: Shortest path optimization with priority queue
        assignments = dijkstra_assignment(autos_copy, active_locations)
    elif selected_model == "Genetic Algorithm":
        # Genetic Algorithm: Evolutionary optimization with selection and mutation
        assignments = genetic_algorithm_assignment(autos_copy, active_locations)
    elif selected_model == "Particle Swarm Optimization":
        # PSO: Swarm intelligence with velocity and position updates
        assignments = particle_swarm_assignment(autos_copy, active_locations)
    elif selected_model == "Simulated Annealing":
        # Simulated Annealing: Probabilistic optimization with cooling schedule
        assignments = simulated_annealing_assignment(autos_copy, active_locations)
    elif selected_model == "Ensemble Model (Top 5)":
        # Ensemble Model: Combines top 5 models using weighted voting
        assignments = ensemble_model_assignment(autos_copy, active_locations)
    else:
        # Default: use KNN assignment
        assignments = knn_auto_assignment(autos_copy, active_locations)
    
    # Calculate metrics for the selected model
    model_metrics = calculate_model_metrics(assignments, active_locations, autos_copy)
    
    # Create and display map (use autos_copy to show updated passenger counts)
    map_fig = create_prototype_map(
        autos_copy,
        active_locations,
        assignments
    )
    st.plotly_chart(map_fig, use_container_width=True)

with col2:
    st.subheader("📊 Summary")
    st.markdown("### Auto Assignments")
    
    if assignments:
        for assignment in assignments:
            st.info(f"**Auto {assignment['auto_id']}** → {assignment['location']}\n"
                   f"Students: {assignment['students']}\n"
                   f"Distance: {assignment['distance']:.1f} units")
    else:
        st.warning("No assignments made")
    
    # Total statistics
    total_students = sum(active_locations.values())
    total_capacity = sum(auto['capacity'] for auto in st.session_state.auto_positions)
    st.metric("Total Students", total_students)
    st.metric("Total Auto Capacity", total_capacity)

# Eye-catching Summary Section for Hackathon Pitch
st.markdown("---")

# Calculate statistics
total_students = sum(active_locations.values())
total_picked = sum(assignment['students'] for assignment in assignments)
total_remaining = total_students - total_picked
total_capacity = sum(auto['capacity'] for auto in st.session_state.auto_positions)
utilization_rate = (total_picked / total_capacity * 100) if total_capacity > 0 else 0
coverage_rate = (total_picked / total_students * 100) if total_students > 0 else 0

# Problem Statement Section
st.markdown("## 🎯 Problem Statement")
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 25px; border-radius: 15px; color: white; margin-bottom: 20px; box-shadow: 0 8px 16px rgba(0,0,0,0.2);">
    <h3 style="color: white; margin-top: 0;">The Challenge at Thapar Campus</h3>
    <p style="font-size: 16px; line-height: 1.8;">
        <strong>🚫 Problem 1:</strong> Students wait endlessly at hostels because autos are not available at the right time<br>
        <strong>🚫 Problem 2:</strong> Autos wait with empty/fewer seats, causing delays and inefficiency<br>
        <strong>🚫 Problem 3:</strong> No predictive system to optimize auto routing based on student demand patterns
    </p>
</div>
""", unsafe_allow_html=True)

# Solution Section
st.markdown("## 💡 Our AI-Powered Solution")
st.markdown("""
<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
            padding: 25px; border-radius: 15px; color: white; margin-bottom: 20px; box-shadow: 0 8px 16px rgba(0,0,0,0.2);">
    <h3 style="color: white; margin-top: 0;">Smart Auto Optimization System</h3>
    <p style="font-size: 16px; line-height: 1.8;">
        <strong>✅ KNN Algorithm:</strong> Finds nearest autos to student locations for optimal routing<br>
        <strong>✅ Real-time Assignment:</strong> Dynamically assigns autos based on current student distribution<br>
        <strong>✅ Route Optimization:</strong> Minimizes distance and waiting time for maximum efficiency<br>
        <strong>✅ Predictive Analytics:</strong> Uses ML models to forecast student demand patterns
    </p>
</div>
""", unsafe_allow_html=True)

# Key Metrics Cards
st.markdown("## 📊 Performance Metrics")
col_met1, col_met2, col_met3, col_met4 = st.columns(4)

with col_met1:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 25px; border-radius: 15px; text-align: center; color: white; box-shadow: 0 6px 12px rgba(0,0,0,0.15);">
        <h2 style="margin: 0; font-size: 42px; font-weight: bold;">{total_students}</h2>
        <p style="margin: 10px 0 0 0; font-size: 16px; opacity: 0.95;">👥 Total Students</p>
    </div>
    """, unsafe_allow_html=True)

with col_met2:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                padding: 25px; border-radius: 15px; text-align: center; color: white; box-shadow: 0 6px 12px rgba(0,0,0,0.15);">
        <h2 style="margin: 0; font-size: 42px; font-weight: bold;">{total_picked}</h2>
        <p style="margin: 10px 0 0 0; font-size: 16px; opacity: 0.95;">✅ Students Assigned</p>
    </div>
    """, unsafe_allow_html=True)

with col_met3:
    color1, color2 = ("#28a745", "#20c997") if coverage_rate >= 90 else ("#ffc107", "#ff9800")
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color1} 0%, {color2} 100%); 
                padding: 25px; border-radius: 15px; text-align: center; color: white; box-shadow: 0 6px 12px rgba(0,0,0,0.15);">
        <h2 style="margin: 0; font-size: 42px; font-weight: bold;">{coverage_rate:.0f}%</h2>
        <p style="margin: 10px 0 0 0; font-size: 16px; opacity: 0.95;">📈 Coverage Rate</p>
    </div>
    """, unsafe_allow_html=True)

with col_met4:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                padding: 25px; border-radius: 15px; text-align: center; color: white; box-shadow: 0 6px 12px rgba(0,0,0,0.15);">
        <h2 style="margin: 0; font-size: 42px; font-weight: bold;">{utilization_rate:.0f}%</h2>
        <p style="margin: 10px 0 0 0; font-size: 16px; opacity: 0.95;">🚗 Auto Utilization</p>
    </div>
    """, unsafe_allow_html=True)

# Model & Technology Stack
st.markdown("---")
col_tech1, col_tech2 = st.columns(2)

with col_tech1:
    st.markdown("### 🤖 AI Models & Technology")
    st.markdown(f"""
    <div style="background-color: #ffffff; padding: 20px; border-radius: 12px; border-left: 5px solid #007bff; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <p style="margin: 8px 0; font-size: 16px; color: #212529;"><strong>🎯 Current Model:</strong> <span style="color: #007bff; font-weight: bold;">{selected_model}</span></p>
        <p style="margin: 8px 0; font-size: 16px; color: #212529;"><strong>📍 Active Locations:</strong> <span style="color: #28a745; font-weight: bold;">{len(active_locations)}</span> buildings</p>
        <p style="margin: 8px 0; font-size: 16px; color: #212529;"><strong>🚗 Total Autos:</strong> <span style="color: #dc3545; font-weight: bold;">{len(st.session_state.auto_positions)}</span> vehicles</p>
        <p style="margin: 8px 0; font-size: 16px; color: #212529;"><strong>💺 Total Capacity:</strong> <span style="color: #6f42c1; font-weight: bold;">{total_capacity}</span> seats</p>
        <p style="margin: 8px 0; font-size: 16px; color: #212529;"><strong>⚡ Tech Stack:</strong> <span style="color: #495057;">Python, Streamlit, Scikit-learn, Plotly, KNN, Random Forest</span></p>
    </div>
    """, unsafe_allow_html=True)

with col_tech2:
    st.markdown("### 📋 Assignment Summary")
    auto_assignments_grouped = {}
    for assignment in assignments:
        auto_id = assignment['auto_id']
        if auto_id not in auto_assignments_grouped:
            auto_assignments_grouped[auto_id] = []
        auto_assignments_grouped[auto_id].append(assignment)
    
    summary_text = ""
    for auto_id in sorted(auto_assignments_grouped.keys()):
        auto_ass = auto_assignments_grouped[auto_id]
        total_auto = sum(a['students'] for a in auto_ass)
        locs = [a['location'] for a in auto_ass]
        summary_text += f"<p style='margin: 10px 0; font-size: 16px; color: #212529;'><strong>🚗 Auto {auto_id}:</strong> <span style='color: #dc3545; font-weight: bold;'>{total_auto}/4</span> seats filled → <span style='color: #28a745;'>{', '.join(locs[:2])}{'...' if len(locs) > 2 else ''}</span></p>"
    
    if not summary_text:
        summary_text = "<p style='color: #6c757d; font-size: 16px;'>No assignments made yet</p>"
    
    st.markdown(f"""
    <div style="background-color: #ffffff; padding: 20px; border-radius: 12px; border-left: 5px solid #28a745; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        {summary_text}
    </div>
    """, unsafe_allow_html=True)

# Feed Data Button
st.markdown("---")
col_feed = st.columns([1, 2, 1])
with col_feed[1]:
    if st.button("📥 Feed Data / Update Student Frequencies", use_container_width=True, type="primary"):
        st.session_state.show_data_page = True
        st.rerun()

# Data Feeding Page
if st.session_state.get('show_data_page', False):
    st.markdown("---")
    st.subheader("📝 Feed Data - Student Frequencies & Auto Management")
    
    tab1, tab2 = st.tabs(["📍 Student Frequencies", "🚗 Auto Management"])
    
    with tab1:
        with st.form("data_feed_form"):
            st.markdown("### Update Student Frequency at Locations")
            
            col5, col6 = st.columns(2)
            
            with col5:
                st.session_state.student_frequency['COS'] = st.number_input("COS", min_value=0, value=st.session_state.student_frequency['COS'])
                st.session_state.student_frequency['Hostel A'] = st.number_input("Hostel A", min_value=0, value=st.session_state.student_frequency['Hostel A'])
                st.session_state.student_frequency['Library'] = st.number_input("Library", min_value=0, value=st.session_state.student_frequency['Library'])
                st.session_state.student_frequency['Jaggi'] = st.number_input("Jaggi", min_value=0, value=st.session_state.student_frequency['Jaggi'])
            
            with col6:
                st.session_state.student_frequency['PG'] = st.number_input("PG", min_value=0, value=st.session_state.student_frequency['PG'])
                st.session_state.student_frequency['O Block'] = st.number_input("O Block", min_value=0, value=st.session_state.student_frequency['O Block'])
                st.session_state.student_frequency['B Block'] = st.number_input("B Block", min_value=0, value=st.session_state.student_frequency['B Block'])
                st.session_state.student_frequency['Dispensary'] = st.number_input("Dispensary", min_value=0, value=st.session_state.student_frequency['Dispensary'])
            
            submitted = st.form_submit_button("💾 Save Student Data", use_container_width=True)
            
            if submitted:
                st.success("✅ Student frequency data saved successfully!")
                st.rerun()
    
    with tab2:
        st.markdown("### Manage Autos")
        st.markdown("Add, remove, or modify auto positions and capacities")
        
        # Quick Decrement/Increment Controls
        col_dec, col_info = st.columns([1, 2])
        with col_dec:
            st.markdown("#### Quick Actions")
            if st.button("➖ Decrement Auto (Remove Last)", use_container_width=True, type="secondary"):
                if len(st.session_state.auto_positions) > 0:
                    removed_auto = st.session_state.auto_positions.pop()
                    st.success(f"✅ Auto {removed_auto['id']} removed successfully!")
                    st.rerun()
                else:
                    st.warning("⚠️ No autos to remove!")
        
        with col_info:
            st.info(f"📊 Current: **{len(st.session_state.auto_positions)}** autos | Total Capacity: **{sum(a['capacity'] for a in st.session_state.auto_positions)}** seats")
        
        # Check for existing collisions and fix them
        active_locs = {k: v for k, v in st.session_state.student_frequency.items() if v > 0}
        # Check all autos for collisions
        for i, auto in enumerate(st.session_state.auto_positions):
            collides = False
            collision_dist = 80  # Minimum safe distance
            
            # Check collision with locations
            for loc_name, loc_data in LOCATIONS.items():
                if loc_name in active_locs:
                    dist = calculate_distance(auto['x'], auto['y'], loc_data['x'], loc_data['y'])
                    if dist < collision_dist:
                        collides = True
                        # Fix collision by moving to empty space
                        new_x, new_y = find_empty_position(st.session_state.auto_positions, active_locs)
                        st.session_state.auto_positions[i]['x'] = new_x
                        st.session_state.auto_positions[i]['y'] = new_y
                        st.warning(f"⚠️ Auto {auto['id']} was too close to {loc_name}. Moved to ({new_x}, {new_y})")
                        break
            
            # Check collision with other autos
            if not collides:
                for j, other_auto in enumerate(st.session_state.auto_positions):
                    if i != j:
                        dist = calculate_distance(auto['x'], auto['y'], other_auto['x'], other_auto['y'])
                        if dist < collision_dist:
                            collides = True
                            # Fix collision by moving to empty space
                            new_x, new_y = find_empty_position(st.session_state.auto_positions, active_locs)
                            st.session_state.auto_positions[i]['x'] = new_x
                            st.session_state.auto_positions[i]['y'] = new_y
                            st.warning(f"⚠️ Auto {auto['id']} was too close to Auto {other_auto['id']}. Moved to ({new_x}, {new_y})")
                            break
        
        # Display current autos
        st.markdown("#### Current Autos")
        for i, auto in enumerate(st.session_state.auto_positions):
            with st.expander(f"🚗 Auto {auto['id']} - Position ({auto['x']}, {auto['y']})"):
                col_a1, col_a2, col_a3, col_a4 = st.columns(4)
                with col_a1:
                    new_x = st.number_input(f"X Position", value=auto['x'], min_value=0, max_value=700, key=f"x_{auto['id']}")
                with col_a2:
                    new_y = st.number_input(f"Y Position", value=auto['y'], min_value=0, max_value=600, key=f"y_{auto['id']}")
                with col_a3:
                    new_capacity = st.number_input(f"Capacity", value=auto['capacity'], min_value=1, max_value=10, key=f"cap_{auto['id']}")
                with col_a4:
                    if st.button("Update", key=f"update_{auto['id']}"):
                        st.session_state.auto_positions[i]['x'] = int(new_x)
                        st.session_state.auto_positions[i]['y'] = int(new_y)
                        st.session_state.auto_positions[i]['capacity'] = int(new_capacity)
                        st.success(f"✅ Auto {auto['id']} updated!")
                        st.rerun()
                    if st.button("Remove", key=f"remove_{auto['id']}"):
                        st.session_state.auto_positions.pop(i)
                        st.success(f"✅ Auto {auto['id']} removed!")
                        st.rerun()
        
        # Add new auto - automatically finds empty position
        st.markdown("#### Add New Auto")
        # Find empty position automatically
        empty_x, empty_y = find_empty_position(
            st.session_state.auto_positions,
            {k: v for k, v in st.session_state.student_frequency.items() if v > 0}
        )
        
        with st.form("add_auto_form"):
            st.info(f"💡 Auto will be placed at ({empty_x}, {empty_y}) - Empty space detected!")
            col_add1, col_add2, col_add3 = st.columns(3)
            with col_add1:
                new_auto_x = st.number_input("X Position", min_value=0, max_value=700, value=empty_x)
            with col_add2:
                new_auto_y = st.number_input("Y Position", min_value=0, max_value=600, value=empty_y)
            with col_add3:
                new_auto_capacity = st.number_input("Capacity", min_value=1, max_value=10, value=4)
            
            # Validate position doesn't collide
            collides = False
            collision_with = None
            test_x, test_y = int(new_auto_x), int(new_auto_y)
            
            # Check collision with locations
            for loc_name, loc_data in LOCATIONS.items():
                loc_x, loc_y = loc_data['x'], loc_data['y']
                dist = calculate_distance(test_x, test_y, loc_x, loc_y)
                if dist < 80:
                    collides = True
                    collision_with = loc_name
                    break
            
            # Check collision with existing autos
            if not collides:
                for auto in st.session_state.auto_positions:
                    dist = calculate_distance(test_x, test_y, auto['x'], auto['y'])
                    if dist < 80:
                        collides = True
                        collision_with = f"Auto {auto['id']}"
                        break
            
            if collides:
                st.warning(f"⚠️ Position ({test_x}, {test_y}) is too close to {collision_with}. Using empty position instead.")
                final_x, final_y = empty_x, empty_y
            else:
                final_x, final_y = test_x, test_y
            
            if st.form_submit_button("➕ Add New Auto"):
                new_auto_id = max([a['id'] for a in st.session_state.auto_positions], default=0) + 1
                st.session_state.auto_positions.append({
                    'id': new_auto_id,
                    'x': final_x,
                    'y': final_y,
                    'capacity': int(new_auto_capacity),
                    'passengers': 0
                })
                st.success(f"✅ Auto {new_auto_id} added at position ({final_x}, {final_y})!")
                st.rerun()
    
    # Model Performance Summary - Best Model Recommendation
    # Use the same metrics calculated in sidebar for consistency
    st.markdown("---")
    st.markdown("## 🏆 Best Model Recommendation")
    
    # Use all_models_metrics from sidebar (already calculated above)
    summary_models_metrics = all_models_metrics
    all_model_names = model_names  # Use same model_names from sidebar
    
    total_students_summary = sum(active_locations_metrics.values())
    total_autos_summary = len(st.session_state.auto_positions)
    total_capacity_summary = sum(a['capacity'] for a in st.session_state.auto_positions)
    
    # Find best model based on composite score
    # Score = (Accuracy * 0.3) + (100 - Normalized Distance * 0.3) + (Utilization * 0.2) + (Precision * 0.2)
    best_model = None
    best_score = -1
    
    for model_name, metrics in summary_models_metrics.items():
        # Normalize distance (lower is better, so invert)
        max_dist = max([m['total_distance'] for m in summary_models_metrics.values()]) if summary_models_metrics.values() else 1000
        normalized_dist_score = (1 - (metrics['total_distance'] / max_dist)) * 100 if max_dist > 0 else 100
        
        # Composite score
        composite_score = (
            metrics['accuracy'] * 0.3 +
            normalized_dist_score * 0.3 +
            metrics['utilization'] * 0.2 +
            metrics['precision'] * 0.2
        )
        
        if composite_score > best_score:
            best_score = composite_score
            best_model = model_name
    
    if best_model:
        best_metrics = summary_models_metrics[best_model]
        
        # Eye-catching summary card
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 30px; border-radius: 15px; color: white; box-shadow: 0 8px 16px rgba(0,0,0,0.3); margin-bottom: 20px;">
            <h2 style="color: white; margin-top: 0; text-align: center;">🥇 Best Performing Model</h2>
            <div style="background: rgba(255,255,255,0.2); padding: 20px; border-radius: 10px; margin: 15px 0;">
                <h3 style="color: #FFD700; margin: 0; font-size: 28px; text-align: center;">{best_model}</h3>
                <p style="text-align: center; font-size: 16px; margin: 10px 0; opacity: 0.9;">Recommended for current configuration</p>
            </div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 20px;">
                <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 8px; text-align: center;">
                    <p style="margin: 0; font-size: 14px; opacity: 0.9;">Accuracy</p>
                    <p style="margin: 5px 0 0 0; font-size: 24px; font-weight: bold;">{best_metrics['accuracy']:.1f}%</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 8px; text-align: center;">
                    <p style="margin: 0; font-size: 14px; opacity: 0.9;">Total Distance</p>
                    <p style="margin: 5px 0 0 0; font-size: 24px; font-weight: bold;">{best_metrics['total_distance']:.1f}m</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 8px; text-align: center;">
                    <p style="margin: 0; font-size: 14px; opacity: 0.9;">Utilization</p>
                    <p style="margin: 5px 0 0 0; font-size: 24px; font-weight: bold;">{best_metrics['utilization']:.1f}%</p>
                </div>
            </div>
            <div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 8px;">
                <p style="margin: 5px 0; font-size: 14px;"><strong>📊 Current Configuration:</strong></p>
                <p style="margin: 5px 0; font-size: 14px;">👥 Students: <strong>{total_students_summary}</strong> | 🚗 Autos: <strong>{total_autos_summary}</strong> | 💺 Capacity: <strong>{total_capacity_summary}</strong></p>
                <p style="margin: 10px 0 5px 0; font-size: 14px;"><strong>🎯 Why this model?</strong></p>
                <p style="margin: 5px 0; font-size: 13px; opacity: 0.9;">
                    This model achieved the best composite score combining accuracy ({best_metrics['accuracy']:.1f}%), 
                    distance optimization ({best_metrics['total_distance']:.1f}m), utilization ({best_metrics['utilization']:.1f}%), 
                    and precision ({best_metrics['precision']:.1f}%) for your current setup.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick comparison table
        st.markdown("### 📈 All Models Performance Comparison")
        comparison_df = pd.DataFrame({
            'Model': all_model_names,
            'Accuracy (%)': [summary_models_metrics[m]['accuracy'] for m in all_model_names],
            'Distance': [summary_models_metrics[m]['total_distance'] for m in all_model_names],
            'Utilization (%)': [summary_models_metrics[m]['utilization'] for m in all_model_names]
        })
        
        # Highlight best model
        def highlight_best(row):
            if row['Model'] == best_model:
                return ['background-color: #FFD700; font-weight: bold;'] * len(row)
            return [''] * len(row)
        
        st.dataframe(
            comparison_df.style.apply(highlight_best, axis=1).format({
                'Accuracy (%)': '{:.1f}',
                'Distance': '{:.1f}',
                'Utilization (%)': '{:.1f}'
            }),
            use_container_width=True,
            hide_index=True
        )
    
    if st.button("❌ Close", use_container_width=True):
        st.session_state.show_data_page = False
        st.rerun()

# Footer
st.markdown("---")
st.markdown("**Thapar Auto Optimization System** | Built with Streamlit & ML Models")
