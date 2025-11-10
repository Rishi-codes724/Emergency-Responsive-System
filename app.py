import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from env import RuralEnv
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Smart Ambulance Dispatch",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .failure-card {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .grid-container {
        display: flex;
        justify-content: center;
        font-size: 2rem;
        margin: 2rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'simulation_history' not in st.session_state:
    st.session_state.simulation_history = []
if 'total_simulations' not in st.session_state:
    st.session_state.total_simulations = 0
if 'success_count' not in st.session_state:
    st.session_state.success_count = 0

# Header
st.markdown('<div class="main-header">🚑 Smart Ambulance Dispatch Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Emergency Response System using Reinforcement Learning</div>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/ambulance.png", width=150)
    st.header("⚙️ Configuration")
    
    # Simulation settings
    st.subheader("Simulation Settings")
    n_episodes = st.slider("Number of Episodes", 1, 10, 2, help="Number of emergency scenarios to simulate")
    animation_speed = st.slider("Animation Speed", 0.5, 3.0, 1.0, 0.5, help="Seconds between episodes")
    
    # Environment settings
    st.subheader("Environment Setup")
    n_hospitals = st.selectbox("Number of Hospitals", [5, 6, 7, 8], index=1)
    n_ambulances = st.selectbox("Number of Ambulances", [2, 3, 4, 5], index=1)
    grid_size = st.selectbox("Grid Size", ["4x4", "5x5", "6x6"], index=0)
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        show_detailed_logs = st.checkbox("Show Detailed Logs", value=False)
        auto_run = st.checkbox("Auto-run Simulation", value=False)
        show_probabilities = st.checkbox("Show Decision Probabilities", value=False)
    
    st.divider()
    
    # Statistics
    st.subheader("📊 Session Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Runs", st.session_state.total_simulations)
    with col2:
        success_rate = (st.session_state.success_count / st.session_state.total_simulations * 100) if st.session_state.total_simulations > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    if st.button("🔄 Reset Statistics", use_container_width=True):
        st.session_state.simulation_history = []
        st.session_state.total_simulations = 0
        st.session_state.success_count = 0
        st.rerun()

# Parse grid size
grid_rows, grid_cols = map(int, grid_size.split('x'))

# Load Q-table
q_table_loaded = False
try:
    Q = np.load("results/q_table.npy")
    q_table_loaded = True
    st.success("✅ Trained Q-table loaded successfully!")
except:
    Q = None
    st.warning("⚠️ No trained Q-table found. Using random policy for demonstration.")

# Initialize environment
env = RuralEnv(grid=(grid_rows, grid_cols), n_hospitals=n_hospitals, n_ambulances=n_ambulances)

def draw_grid_visual(patient_zone, ambulance_zone=None, hospital_zones=[], selected_hospital=None):
    """Create an enhanced visual grid representation"""
    rows, cols = grid_rows, grid_cols
    grid = [["⬜" for _ in range(cols)] for _ in range(rows)]
    
    # Place patient
    pr, pc = patient_zone // cols, patient_zone % cols
    grid[pr][pc] = "🧍‍♂️"
    
    # Place ambulance
    if ambulance_zone is not None:
        ar, ac = ambulance_zone // cols, ambulance_zone % cols
        if grid[ar][ac] == "⬜":
            grid[ar][ac] = "🚑"
        else:
            grid[ar][ac] = grid[ar][ac] + "🚑"
    
    # Place hospitals
    for idx, hz in enumerate(hospital_zones):
        hr, hc = hz // cols, hz % cols
        if selected_hospital is not None and idx == selected_hospital:
            hospital_icon = "🏥✨"  # Highlight selected hospital
        else:
            hospital_icon = "🏥"
        
        if grid[hr][hc] == "⬜":
            grid[hr][hc] = hospital_icon
        else:
            grid[hr][hc] = grid[hr][hc] + hospital_icon
    
    return "\n".join([" ".join(row) for row in grid])

def create_distance_heatmap(patient_zone, ambulance_zones, hospital_zones):
    """Create a heatmap showing distances"""
    rows, cols = grid_rows, grid_cols
    heatmap_data = np.zeros((rows, cols))
    
    pr, pc = patient_zone // cols, patient_zone % cols
    
    for r in range(rows):
        for c in range(cols):
            dist = abs(r - pr) + abs(c - pc)
            heatmap_data[r][c] = dist
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        colorscale='RdYlGn_r',
        showscale=True,
        colorbar=dict(title="Distance")
    ))
    
    # Add markers
    fig.add_trace(go.Scatter(
        x=[pc], y=[pr],
        mode='markers+text',
        marker=dict(size=20, color='red', symbol='circle'),
        text=['Patient'],
        textposition='top center',
        name='Patient'
    ))
    
    for az in ambulance_zones:
        ar, ac = az // cols, az % cols
        fig.add_trace(go.Scatter(
            x=[ac], y=[ar],
            mode='markers+text',
            marker=dict(size=15, color='blue', symbol='square'),
            text=['Ambulance'],
            textposition='bottom center',
            name='Ambulance'
        ))
    
    for hz in hospital_zones:
        hr, hc = hz // cols, hz % cols
        fig.add_trace(go.Scatter(
            x=[hc], y=[hr],
            mode='markers+text',
            marker=dict(size=15, color='green', symbol='cross'),
            text=['Hospital'],
            textposition='top center',
            name='Hospital'
        ))
    
    fig.update_layout(
        title="Distance Heatmap (Manhattan Distance)",
        xaxis=dict(showgrid=True, zeroline=False),
        yaxis=dict(showgrid=True, zeroline=False, autorange='reversed'),
        height=400,
        showlegend=False
    )
    
    return fig

def create_performance_chart(history):
    """Create performance metrics chart"""
    if len(history) == 0:
        return None
    
    df = pd.DataFrame(history)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['episode'],
        y=df['reward'],
        mode='lines+markers',
        name='Reward',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Episode Rewards Over Time",
        xaxis_title="Episode",
        yaxis_title="Reward",
        height=300,
        template="plotly_white"
    )
    
    return fig

# Main simulation controls
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    run_simulation = st.button("▶️ Run Simulation", use_container_width=True, type="primary")
with col2:
    if st.button("📊 View Analytics", use_container_width=True):
        st.session_state.show_analytics = True
with col3:
    if st.button("💾 Export Results", use_container_width=True):
        if st.session_state.simulation_history:
            df = pd.DataFrame(st.session_state.simulation_history)
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "simulation_results.csv",
                "text/csv",
                key='download-csv'
            )

st.divider()

# Run simulation
if run_simulation or auto_run:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for episode_idx in range(n_episodes):
        status_text.text(f"Running Episode {episode_idx + 1}/{n_episodes}...")
        progress_bar.progress((episode_idx + 1) / n_episodes)
        
        # Reset environment
        state = env.reset()
        info = env.render_state_readable()
        
        # Create episode container
        with st.container():
            st.markdown(f"### 📋 Episode {episode_idx + 1}")
            
            # Create three columns for layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Patient Information
                patient = info["patient"]
                severity_colors = {0: "🟢", 1: "🟡", 2: "🔴"}
                severity_names = {0: "Low", 1: "Medium", 2: "High"}
                
                st.markdown(f"""
                <div class="info-box">
                    <h4>🚨 Emergency Call</h4>
                    <p><strong>Location:</strong> Zone {patient['zone']}</p>
                    <p><strong>Severity:</strong> {severity_colors[patient['severity']]} {severity_names[patient['severity']]}</p>
                    <p><strong>Required Specialty:</strong> {patient['required_specialty'] if patient['required_specialty'] else 'General'}</p>
                    <p><strong>Timestamp:</strong> {datetime.now().strftime('%H:%M:%S')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Grid visualization
                st.markdown("#### 🗺️ Grid View")
                
                # Get action first to show selected hospital
                if Q is not None:
                    row = Q[state]
                    action = int(np.argmax(row))
                else:
                    action = np.random.randint(0, n_ambulances * n_hospitals)
                
                amb_idx = action // env.n_hospitals
                hosp_idx = action % env.n_hospitals
                
                patient_zone = patient["zone"]
                ambulance_zone = info["ambulances"][amb_idx]["zone"]
                hospital_zones = [h["zone"] for h in info["hospitals"]]
                
                grid_text = draw_grid_visual(patient_zone, ambulance_zone, hospital_zones, hosp_idx)
                st.code(grid_text, language=None)
                
                # Distance heatmap
                with st.expander("📍 View Distance Heatmap"):
                    fig = create_distance_heatmap(
                        patient_zone,
                        [a["zone"] for a in info["ambulances"]],
                        hospital_zones
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Available Resources
                st.markdown("#### 🚑 Available Ambulances")
                for a in info['ambulances']:
                    st.markdown(f"""
                    <div style='background-color: #e3f2fd; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0;'>
                        <strong>Ambulance {a['id']}</strong><br/>
                        📍 Zone: {a['zone']}
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("#### 🏥 Hospital Status")
                for h in info['hospitals']:
                    bed_percentage = (h['available_beds'] / h['total_beds']) * 100
                    color = '#4caf50' if bed_percentage > 50 else '#ff9800' if bed_percentage > 20 else '#f44336'
                    
                    specs = ', '.join([k.title() for k, v in h['specialties'].items() if v]) or 'General'
                    
                    st.markdown(f"""
                    <div style='background-color: #f5f5f5; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0; border-left: 4px solid {color};'>
                        <strong>Hospital {h['id']}</strong> (Zone {h['zone']})<br/>
                        🛏️ Beds: {h['available_beds']}/{h['total_beds']}<br/>
                        🏥 ICU: {h['icu_available']}<br/>
                        💊 {specs}
                    </div>
                    """, unsafe_allow_html=True)
            
            # AI Decision
            st.markdown("#### 🤖 AI Decision")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Selected Ambulance", f"🚑 {amb_idx}")
            with col_b:
                st.metric("Target Hospital", f"🏥 {hosp_idx}")
            with col_c:
                dist = abs((patient_zone // grid_cols) - (hospital_zones[hosp_idx] // grid_cols)) + \
                       abs((patient_zone % grid_cols) - (hospital_zones[hosp_idx] % grid_cols))
                st.metric("Distance", f"{dist} zones")
            
            # Execute action
            ns, reward, done, step_info = env.step(action)
            
            # Result
            if step_info.get('success', False):
                st.markdown(f"""
                <div class="success-card">
                    <h4>✅ Dispatch Successful!</h4>
                    <p><strong>Travel Time:</strong> {step_info.get('travel_time_min', 'N/A')} minutes</p>
                    <p><strong>Distance Traveled:</strong> {step_info.get('dist_hops', 'N/A')} zones</p>
                    <p><strong>Reward:</strong> {reward:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                st.session_state.success_count += 1
            else:
                reason = step_info.get('reason', 'Unknown')
                reason_text = {
                    'no_beds': 'Hospital has no available beds',
                    'no_specialty': 'Hospital lacks required specialty',
                    'timeout': 'Response time exceeded threshold'
                }.get(reason, reason)
                
                st.markdown(f"""
                <div class="failure-card">
                    <h4>❌ Dispatch Failed</h4>
                    <p><strong>Reason:</strong> {reason_text}</p>
                    <p><strong>Penalty:</strong> {reward:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed logs
            if show_detailed_logs:
                with st.expander("🔍 Detailed Logs"):
                    st.json(step_info)
            
            # Save to history
            st.session_state.simulation_history.append({
                'episode': st.session_state.total_simulations + 1,
                'reward': reward,
                'success': step_info.get('success', False),
                'travel_time': step_info.get('travel_time_min', 0),
                'distance': step_info.get('dist_hops', 0),
                'severity': patient['severity'],
                'specialty': patient['required_specialty']
            })
            st.session_state.total_simulations += 1
            
            st.divider()
            
        # Animation delay
        time.sleep(animation_speed)
    
    status_text.text("Simulation complete!")
    progress_bar.empty()
    
    # Show summary
    st.success(f"✅ Completed {n_episodes} episodes!")
    
    # Performance chart
    if len(st.session_state.simulation_history) > 1:
        st.markdown("### 📈 Performance Summary")
        fig = create_performance_chart(st.session_state.simulation_history[-n_episodes:])
        if fig:
            st.plotly_chart(fig, use_container_width=True)

# Analytics section
if st.session_state.get('show_analytics', False) and len(st.session_state.simulation_history) > 0:
    st.markdown("## 📊 Analytics Dashboard")
    
    df = pd.DataFrame(st.session_state.simulation_history)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Episodes", len(df))
    with col2:
        st.metric("Avg Reward", f"{df['reward'].mean():.2f}")
    with col3:
        st.metric("Success Rate", f"{df['success'].mean()*100:.1f}%")
    with col4:
        st.metric("Avg Travel Time", f"{df['travel_time'].mean():.1f} min")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Success rate by severity
        severity_stats = df.groupby('severity')['success'].agg(['mean', 'count'])
        fig = px.bar(
            severity_stats.reset_index(),
            x='severity',
            y='mean',
            title='Success Rate by Severity Level',
            labels={'severity': 'Severity', 'mean': 'Success Rate'},
            color='mean',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Travel time distribution
        fig = px.histogram(
            df,
            x='travel_time',
            title='Travel Time Distribution',
            labels={'travel_time': 'Travel Time (min)'},
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚑 Smart Ambulance Dispatch Simulator | Powered by Reinforcement Learning</p>
    <p>Built with Streamlit | © 2025</p>
</div>
""", unsafe_allow_html=True)
