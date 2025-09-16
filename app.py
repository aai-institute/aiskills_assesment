import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import base64
from io import BytesIO
import plotly.express as px
import numpy as np
import re
import os

# Set page config
st.set_page_config(
    page_title="AI Skills Assessment",
    page_icon="🤖",
    layout="wide"
)

@st.cache_data
def load_skills_dataframe():
    skills_df = pd.read_csv('data/skills_with_coordinates.csv', sep = ';')
    skills_df['level'] = 0  # Initialize all to level 0
    return skills_df

def sanitize_filename(text):
    """Sanitize text for use in filenames"""
    # Replace spaces with underscores and remove special characters
    sanitized = re.sub(r'[^\w\-_.]', '_', text)
    # Remove multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized

def load_logo():
    """Load and encode logo for display"""
    logo_path = "images/appliedAI_horizontal_rgb_RZ.png"
    if os.path.exists(logo_path):
        try:
            with open(logo_path, "rb") as f:
                logo_data = base64.b64encode(f.read()).decode()
            return f"data:image/png;base64,{logo_data}"
        except Exception as e:
            st.warning(f"Could not load logo: {e}")
            return None
    return None

def create_ai_skills_plot(skills_df, name="Individual", add_logo=True):
    """Create the plotly scatter plot for AI Skills Framework with level-based fill colors"""
    
    # Base stroke colors for each purpose
    stroke_colors = {
        'USE AI': '#FF611A',        # Orange
        'INTEGRATE AI': '#704DFF',   # Purple  
        'BUILD AI': '#18A5A7',       # Teal
        'FRAME AI': '#46ADD5'        # Light Blue
    }
    
    # Level-based fill colors for each purpose
    fill_color_palettes = {
        'USE AI': {
            0: '#FFFFFF',    # don't know - white/empty
            1: '#FFE6D9',    # know - very light orange
            2: '#FFCCB3',    # use - light orange  
            3: '#FF9966',    # apply - medium orange
            4: '#FF611A'     # live - full orange
        },
        'INTEGRATE AI': {
            0: '#FFFFFF',    # don't know - white/empty
            1: '#E6DDFF',    # know - very light purple
            2: '#CCBBFF',    # use - light purple
            3: '#9977FF',    # apply - medium purple
            4: '#704DFF'     # live - full purple
        },
        'BUILD AI': {
            0: '#FFFFFF',    # don't know - white/empty
            1: '#D9F2F2',    # know - very light teal
            2: '#B3E5E5',    # use - light teal
            3: '#66CCCC',    # apply - medium teal
            4: '#18A5A7'     # live - full teal
        },
        'FRAME AI': {
            0: '#FFFFFF',    # don't know - white/empty
            1: '#E6F5F9',    # know - very light blue
            2: '#CCEBF3',    # use - light blue
            3: '#80D1E6',    # apply - medium blue
            4: '#46ADD5'     # live - full blue
        }
    }
    
    fig = go.Figure()
    
    # Calculate plot bounds preserving aspect ratio
    all_x = skills_df['x'].tolist()
    all_y = skills_df['y'].tolist()
    
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    # Calculate center and size
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Use uniform margin as percentage of the larger range to preserve aspect ratio
    max_range = max(x_range, y_range)
    margin = max_range * 0.15
    
    # Apply same margin to both axes to preserve aspect ratio
    x_range_new = x_range + 2 * margin
    y_range_new = y_range + 2 * margin
    
    # Calculate final bounds centered around the data
    x_min_final = x_center - x_range_new / 2
    x_max_final = x_center + x_range_new / 2
    y_min_final = y_center - y_range_new / 2
    y_max_final = y_center + y_range_new / 2
    
    # Add traces for each purpose
    for purpose in stroke_colors.keys():
        if purpose in skills_df['purpose'].values:
            df_purpose = skills_df[skills_df['purpose'] == purpose]
            
            # Create hover text with skill and domains
            hover_text = []
            for _, row in df_purpose.iterrows():
                level_name = ['Don\'t Know', 'Know', 'Use', 'Adapt', 'Live'][row['level']]
                hover_text.append(f"{row['skill_component']})")
            
            # Get fill colors based on levels
            fill_colors = [fill_color_palettes[purpose][level] for level in df_purpose['level']]
            
            fig.add_trace(go.Scatter(
                x=df_purpose['x'],
                y=df_purpose['y'],
                mode='markers',
                name=purpose,
                marker=dict(
                    color=fill_colors,
                    size=12,
                    line=dict(
                        color=stroke_colors[purpose],
                        width=1.0
                    )
                ),
                text=hover_text,
                hovertemplate='<span style="color:white;">%{text}</span><extra></extra>',
                hoverlabel=dict(
                    bgcolor=stroke_colors[purpose],
                    bordercolor=stroke_colors[purpose],
                    font=dict(color='white', size=11)
                ),
                showlegend=False
            ))
    
    # Calculate center positions for each cluster to place text labels
    cluster_centers = {}
    all_cluster_mins = []
    
    for purpose in stroke_colors.keys():
        if purpose in skills_df['purpose'].values:
            df_purpose = skills_df[skills_df['purpose'] == purpose]
            center_x = df_purpose['x'].mean()
            cluster_min_y = df_purpose['y'].min()
            all_cluster_mins.append(cluster_min_y)
            cluster_centers[purpose] = (center_x, cluster_min_y)
    
    # Find the lowest point across all clusters and use it for all labels
    global_min_y = min(all_cluster_mins) if all_cluster_mins else 0
    label_y_position = global_min_y - margin * 0.3
    
    # Add text labels below each cluster
    for purpose, (center_x, _) in cluster_centers.items():
        fig.add_annotation(
            x=center_x,
            y=label_y_position,
            text=f"<b>{purpose}</b>",
            showarrow=False,
            font=dict(size=14, color=stroke_colors[purpose]),
            xanchor='center',
            yanchor='top'
        )
    
    # Add title
    fig.add_annotation(
        x=x_min_final + (x_max_final - x_min_final) * 0.1,
        y=y_max_final - margin * 0.02,
        text=f"AI Skills Profile - <b>{name}</b>",
        showarrow=False,
        font=dict(size=26, color='#084059'),
        xanchor='left',
        yanchor='top'
    )
    
    fig.update_layout(
        title="",
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[x_min_final, x_max_final],
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[y_min_final, y_max_final]
        ),
        plot_bgcolor='white',
        width=1000,
        height=600,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0)
    )

        # Add logo to plot
    logo_data = load_logo()
    if logo_data:
        fig.add_layout_image(
            dict(
                source=logo_data,
                xref="paper", yref="paper",
                x=0.95, y=0.95,
                sizex=0.2, sizey=0.2,
                xanchor="right", yanchor="top",
                layer="above"
            )
        )
        # Add footer text
    fig.add_annotation(
        x=x_max_final - (x_max_final - x_min_final) * 0.1,
        y=y_min_final,
        text="<i>based on the AI Skills Framework v.01 - appliedAI Institute for Europe</i>",
        showarrow=False,
        font=dict(size=12, color='#666666'),
        xanchor='right',
        yanchor='bottom'
    )    
    
    return fig


def create_domain_skills_plot(skills_df):
    """Create domain-grouped visualization with manually adjustable spacing"""
    
    # MANUAL SPACING CONTROLS - Change these values to adjust layout
    circle_spacing_x = 0.3    # Horizontal spacing between circles within domain
    circle_spacing_y = 0.3    # Vertical spacing between circles within domain
    domain_spacing_x = 3.0    # Horizontal spacing between domains
    domain_spacing_y = 2.5    # Vertical spacing between domain rows
    
    # Level-based fill colors
    level_colors = {
        0: '#FFFFFF',  # No fill
        1: '#D9DEE1',  # Light blue-gray
        2: '#B6C4CA',  # Medium blue-gray  
        3: '#658695',  # Darker blue
        4: '#084059'   # Brand color
    }
    
    # Extract individual domains and create expanded dataframe
    expanded_skills = []
    for _, row in skills_df.iterrows():
        domains = [d.strip() for d in str(row['domains']).split(',')]
        for domain in domains:
            expanded_skills.append({
                'skill_component': row['skill_component'],
                'domain': domain,
                'level': row['level']
            })
    
    expanded_df = pd.DataFrame(expanded_skills)
    
    # Define 5x2 layout: 5 domains on top row, remaining on bottom
    domain_positions = {
        # Row 1: First 5 domains
        'AI Strategy': (0, 0),
        'AI Ethics': (0, 1), 
        'AI Regulation': (0, 2),
        'GenAI Proficiency': (0, 3),
        'Data Competence': (0, 4),
        # Row 2: Remaining domains
        'Machine Learning': (1, 0),
        'MLOps/Infrastructure': (1, 3),
        'Programming': (1, 1),           
        'Software Design': (1, 2)        
    }
    
    fig = go.Figure()
    
    for domain, (grid_row, grid_col) in domain_positions.items():
        if domain not in expanded_df['domain'].values:
            continue
            
        domain_skills = expanded_df[expanded_df['domain'] == domain]
        num_skills = len(domain_skills)
        
        # Calculate grid for this domain's skills
        skills_per_row = min(5, max(3, int(np.ceil(np.sqrt(num_skills)))))
        
        # Base position for this domain cluster
        base_x = grid_col * domain_spacing_x
        base_y = -(grid_row * domain_spacing_y)  # Negative because we go down
        
        x_coords = []
        y_coords = []
        fill_colors = []
        hover_texts = []
        
        for i, (_, skill) in enumerate(domain_skills.iterrows()):
            skill_row = i // skills_per_row
            skill_col = i % skills_per_row
            
            # Position within the cluster using manual spacing controls
            x = base_x + skill_col * circle_spacing_x
            y = base_y - skill_row * circle_spacing_y
            
            x_coords.append(x)
            y_coords.append(y)
            fill_colors.append(level_colors[skill['level']])
            hover_texts.append(skill['skill_component'])
        
        # Add scatter trace for this domain
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            name=domain,
            marker=dict(
                color=fill_colors,
                size=12,  # Same size as purpose plot
                line=dict(color='#084059', width=1.0)
            ),
            text=hover_texts,
            hovertemplate='%{text}<extra></extra>',
            hoverlabel=dict(
                bgcolor='#084059',
                bordercolor='#084059',
                font=dict(color='white', size=11)
            ),
            showlegend=False
        ))
        
        # Add domain label at top-left of cluster
        fig.add_annotation(
            x=base_x-0.2,
            y=base_y + 0.3,
            text=f"<b>{domain}</b>",
            showarrow=False,
            font=dict(size=14, color='#084059'),
            xanchor='left',
            yanchor='bottom'
        )
    
    # Calculate bounds based on actual positions
    all_x = []
    all_y = []
    for trace in fig.data:
        if hasattr(trace, 'x') and hasattr(trace, 'y'):
            all_x.extend(trace.x)
            all_y.extend(trace.y)
    
    if all_x and all_y:
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        # Add margins
        x_margin = 1.0
        y_margin = 1.0
        
        x_range = [x_min - x_margin, x_max + x_margin]
        y_range = [y_min - y_margin, y_max + y_margin]
    else:
        x_range = [0, 10]
        y_range = [0, -10]
    
    fig.update_layout(
        title="",
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=x_range,
            scaleanchor="y",  # Lock aspect ratio
            scaleratio=1
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=y_range
        ),
        plot_bgcolor='white',
        width=1000,
        height=600,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    return fig

def download_plot_as_png(fig, filename, width=1417, height=1063):
    """Create a download button for plotly figure as PNG optimized for 10x15cm print"""
    try:
        # Convert plotly figure to PNG bytes
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
        
        st.download_button(
            label=f"📥 Download {filename}",
            data=img_bytes,
            file_name=f"{filename}.png",
            mime="image/png"
        )
    except Exception as e:
        st.error(f"Error creating download: {e}")
        st.info("💡 Install kaleido for plot downloads: `pip install kaleido`")

def create_skills_filter():
    """Create filtering options for skills"""
    skills_df = load_skills_dataframe()
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_purposes = st.multiselect(
            "Filter by Purpose",
            options=skills_df['purpose'].unique(),
            default=skills_df['purpose'].unique()
        )
    
    with col2:
        # Extract unique domains
        all_domains = set()
        for domains_str in skills_df['domains'].dropna():
            for domain in domains_str.split(','):
                all_domains.add(domain.strip())
        
        selected_domains = st.multiselect(
            "Filter by Domain",
            options=sorted(all_domains),
            default=sorted(all_domains)
        )
    
    
    return selected_purposes, selected_domains

def create_survey_form(skills_df, selected_purposes=None, selected_domains=None):
    """Create the survey form grouped by purpose and domain with filtering"""
    
    # Filter skills based on selections
    if selected_purposes:
        skills_df = skills_df[skills_df['purpose'].isin(selected_purposes)]
    
    if selected_domains:
        # Filter by domains (handle multi-domain entries)
        mask = skills_df['domains'].apply(
            lambda x: any(domain.strip() in selected_domains for domain in str(x).split(','))
        )
        skills_df = skills_df[mask]
    
    if skills_df.empty:
        st.warning("No skills match your current filters.")
        return None
    
    st.header("AI Skills Self-Assessment Survey")
    st.write(f"Rating {len(skills_df)} skills based on your current filters:")
    st.write("**0** = Unknown| **1** = Know about | **2** = Use it | **3** = Adapt it | **4** = Live i")
    
    # Initialize session state
    if 'skill_levels' not in st.session_state:
        original_df = load_skills_dataframe()
        st.session_state.skill_levels = {row['skill_component']: 0 for _, row in original_df.iterrows()}
    
    # Create collapsible form sections
    purpose_colors = {
        'USE AI': '#FF611A',
        'INTEGRATE AI': '#704DFF', 
        'BUILD AI': '#18A5A7',
        'FRAME AI': '#46ADD5'
    }
    
    skill_ratings = {}
    
    for purpose in ['USE AI', 'INTEGRATE AI', 'BUILD AI', 'FRAME AI']:
        if purpose in skills_df['purpose'].values:
            purpose_skills = skills_df[skills_df['purpose'] == purpose]
            
            with st.expander(f"📊 {purpose} ({len(purpose_skills)} skills)", expanded=False):
                st.markdown(f"### <span style='color:{purpose_colors[purpose]}'>{purpose}</span>", unsafe_allow_html=True)
                
                # Group by domain within purpose
                unique_domains = purpose_skills['domains'].dropna().unique()
                
                for domain in sorted(unique_domains):
                    st.write(f"**{domain}**")
                    
                    domain_skills = purpose_skills[purpose_skills['domains'] == domain]
                    
                    for idx, skill_row in domain_skills.iterrows():
                        skill_name = skill_row['skill_component']
                        current_value = st.session_state.skill_levels.get(skill_name, 0)
                        
                        # Use full skill name (no truncation)
                        rating = st.slider(
                            label=skill_name,
                            min_value=0,
                            max_value=4,
                            value=current_value,
                            key=f"skill_slider_{idx}_{purpose.replace(' ', '_')}",
                            help=f"Domain: {domain}"
                        )
                        
                        skill_ratings[skill_name] = rating
                    
                    st.write("")
    
    # Update session state automatically
    if skill_ratings:
        st.session_state.skill_levels.update(skill_ratings)
    
    return skill_ratings

def display_assessment_overview():
    """Display assessment overview with knowledge gaps by purpose"""
    if not st.session_state.get('skill_levels'):
        return
        
    skills_df = load_skills_dataframe()
    updated_df = update_skills_dataframe(skills_df)
    
    current_levels = list(st.session_state.skill_levels.values())
    total_skills = len(current_levels)
    
    st.subheader("📄 Assessment Overview")
    
    col1, col2, col3, col4 = st.columns(4)
 
    with col1:
        skills_above_2 = sum(1 for level in current_levels if level >= 2)
        st.metric("Usable Skills", skills_above_2)
    
    with col2:
        expert_skills = sum(1 for level in current_levels if level == 4)
        st.metric("Expert Level", expert_skills)
    
    # Knowledge gaps by purpose - split into two columns
    purposes = ['USE AI', 'INTEGRATE AI', 'BUILD AI', 'FRAME AI']
    
    with col3:
        st.write("**Knowledge Gaps:**")
        for purpose in purposes[:2]:  # First two
            purpose_skills = updated_df[updated_df['purpose'] == purpose]
            if len(purpose_skills) > 0:
                gaps = (purpose_skills['level'] == 0).sum()
                total_purpose_skills = len(purpose_skills)
                percentage = (gaps/total_purpose_skills)*100
                st.write(f"• {purpose}: {gaps} ({percentage:.0f}%)")
    
    with col4:
        st.write("**&nbsp;**")  # Empty header
        for purpose in purposes[2:]:  # Last two
            purpose_skills = updated_df[updated_df['purpose'] == purpose]
            if len(purpose_skills) > 0:
                gaps = (purpose_skills['level'] == 0).sum()
                total_purpose_skills = len(purpose_skills)
                percentage = (gaps/total_purpose_skills)*100
                st.write(f"• {purpose}: {gaps} ({percentage:.0f}%)")


def display_domain_distribution():
    """Display skill distribution by domain with vertical bar charts (excluding level 0)"""
    if not st.session_state.get('skill_levels'):
        return
        
    skills_df = load_skills_dataframe()
    updated_df = update_skills_dataframe(skills_df)
    
    st.subheader("📈 Skill Distribution by Domain")
    
    # Extract domains and create expanded dataframe
    expanded_skills = []
    for _, row in updated_df.iterrows():
        if row['level'] > 0:  # Exclude level 0
            domains = [d.strip() for d in str(row['domains']).split(',')]
            for domain in domains:
                if domain:  # Make sure domain is not empty
                    expanded_skills.append({
                        'domain': domain,
                        'level': row['level'],
                        'skill_component': row['skill_component']
                    })
    
    if not expanded_skills:
        st.write("No skills above level 0 to display.")
        return
    
    expanded_df = pd.DataFrame(expanded_skills)
    domains = sorted(expanded_df['domain'].unique())

    # Calculate global maximum for consistent y-axis scaling
    global_max = 0
    for domain in domains:
        domain_skills = expanded_df[expanded_df['domain'] == domain]
        level_counts = domain_skills['level'].value_counts()
        domain_max = level_counts.max() if len(level_counts) > 0 else 0
        global_max = max(global_max, domain_max)
    
    # Ensure minimum scale of 1 for readability
    y_max = max(global_max, 1)
    
    # Display domains in rows of 3 - always use 3 columns for consistent sizing
    for i in range(0, len(domains), 3):
        domain_row = domains[i:i+3]
        cols = st.columns(3)  # Always create 3 columns
        
        for j, domain in enumerate(domain_row):
            with cols[j]:
                domain_skills = expanded_df[expanded_df['domain'] == domain]
                level_counts = domain_skills['level'].value_counts().sort_index()
                
                # Create bar chart data - always show all 4 levels
                level_names = {1: 'Know', 2: 'Use', 3: 'Adapt', 4: 'Live'}
                x_labels = []
                counts = []
                colors = []
                color_map = {1: '#D9DEE1', 2: '#B6C4CA', 3: '#658695', 4: '#084059'}
                
                for level in [1, 2, 3, 4]:
                    count = level_counts.get(level, 0)
                    x_labels.append(level_names[level])
                    counts.append(count)
                    colors.append(color_map[level])
                
                # Always create vertical bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=x_labels,
                        y=counts,
                        marker_color=colors
                    )
                ])
                
                fig.update_layout(
                    title=f"{domain}",
                    height=250,
                    showlegend=False,
                    margin=dict(l=20, r=20, t=60, b=60),
                    xaxis_title="",
                    yaxis_title="",
                    title_font_size=12,
                    xaxis=dict(
                        categoryorder='array',
                        categoryarray=['Know', 'Use', 'Adapt', 'Live'],
                        showgrid=False
                    ),
                    yaxis=dict(
                        showgrid=False,
                        dtick=1,  # Show only integer ticks
                        tickmode='linear',
                        range = (0, y_max)
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Fill remaining columns if there are fewer than 3 domains in the row
        for k in range(len(domain_row), 3):
            with cols[k]:
                st.empty()  # Empty placeholder to maintain layout

def update_skills_dataframe(skills_df, skill_ratings=None):
    """Update the skills dataframe with new ratings"""
    updated_df = skills_df.copy()
    
    for idx, row in updated_df.iterrows():
        skill_name = row['skill_component']
        if skill_name in st.session_state.skill_levels:
            updated_df.at[idx, 'level'] = st.session_state.skill_levels[skill_name]
    
    return updated_df

def export_results(name, position):
    """Export assessment results with custom filename"""
    if not st.session_state.get('skill_levels'):
        st.warning("No assessment data to export.")
        return
    
    skills_df = load_skills_dataframe()
    export_df = update_skills_dataframe(skills_df)
    
    # Create summary statistics
    summary_stats = {
        'Total Skills': len(export_df),
        'Average Level': export_df['level'].mean(),
        'Skills at Level 0': (export_df['level'] == 0).sum(),
        'Skills at Level 1': (export_df['level'] == 1).sum(),
        'Skills at Level 2': (export_df['level'] == 2).sum(),
        'Skills at Level 3': (export_df['level'] == 3).sum(),
        'Skills at Level 4': (export_df['level'] == 4).sum(),
    }
    
    # Convert to CSV
    csv = export_df.to_csv(index=False)
    
    # Create filename with sanitized name and position
    sanitized_name = sanitize_filename(name) if name else "Individual"
    sanitized_position = sanitize_filename(position) if position else "Position"
    filename = f"{sanitized_name}_{sanitized_position}_aiskills_selfassessment.csv"
    
    st.download_button(
        label="📥 Download Assessment Results (CSV)",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )
    
    # Show summary
    st.json(summary_stats)

def main():
    # Minimal CSS - only for multiselect brand color
    st.markdown("""
    <style>
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #084059 !important;
        color: white !important;
    }
    
    .stMultiSelect [data-baseweb="tag"] span {
        color: white !important;
    }
    </style>
                
    """, unsafe_allow_html=True)


# Add logo to top right of page using columns
    logo_data = load_logo()
    if logo_data:
        # Create header with logo
        header_col1, header_col2 = st.columns([3, 1])
        with header_col2:
            st.markdown(f"""
            <div style="text-align: right; padding: 10px 0;">
                <img src="{logo_data}" alt="appliedAI Logo" style="height: 100px; width: auto;">
            </div>
            """, unsafe_allow_html=True)
        
        # Move title and description to the first column
        with header_col1:
            st.title("AI Skills Assessment Tool")
            st.markdown("*Assess your AI skills across different competency areas and visualize your profile*")
    
    # Load data
    skills_df = load_skills_dataframe()
    
    if skills_df.empty:
        st.error("❌ Could not load skills data. Please check your CSV file.")
        st.stop()
    
    # Sidebar with correct order
    with st.sidebar:
        user_name = st.text_input("👤 Your Name", value="Individual")
        user_position = st.text_input("💼 Your Position", value="")
        
        st.header("📖 Level Guide")
        st.markdown("""
        **0 - Unknown**: No awareness of the skill concept
                    
        **1 - Know about**: Understands the concept theoretically but lacks practical application ability
                    
        **2 - Use it**: Can execute the skill in familiar, well-defined situations
                    
        **3 - Adapt it**: Can modify and transfer the skill to novel or complex scenarios
                    
        **4 - Live it**: Integrates the skill fluidly with other competencies and can guide others            
        """)
        
        st.header("📊 Export")
        export_results(user_name, user_position)
    
    # Main tabs
    tab1, tab2 = st.tabs(["📝 Assessment", "📊 Skills Profile"])
    
    with tab1:
        # Assessment form with filters
        st.header("🔍 Filter Skills")
        selected_purposes, selected_domains = create_skills_filter()
        
        # Create filtered survey
        skill_ratings = create_survey_form(skills_df, selected_purposes, selected_domains)
        
        # Auto-save notification
        if skill_ratings:
            st.success("✅ Assessment automatically saved!")
    
    with tab2:
        # Visualization
        st.header("Your AI Skills Profile")
        
        display_df = update_skills_dataframe(skills_df)
        
        try:

            # Purpose-based skills plot
            fig = create_ai_skills_plot(display_df, name=user_name, add_logo=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add download button for purpose plot
            download_plot_as_png(fig, f"{user_name.replace(' ', '_')}_AI_Skills_Purpose_Plot")
            
            # Domain-based skills plot
            st.subheader("📊 Skills by Domain")
            domain_fig = create_domain_skills_plot(display_df)
            st.plotly_chart(domain_fig, use_container_width=True)

          
            display_assessment_overview()
            display_domain_distribution()
                
        except Exception as e:
            st.error(f"❌ Error creating visualization: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("*based on the AI Skills Framework v.01 - appliedAI Institute for Europe*")

if __name__ == "__main__":
    main()