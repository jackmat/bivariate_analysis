import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
from matplotlib.table import Table
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def jitter(a_series, noise_reduction=1000000):
    return (np.random.random(len(a_series))*a_series.std()/noise_reduction)-(a_series.std()/(2*noise_reduction))
    
def numeric_decilecuts(df, column_name, n_deciles=10):
    # Use qcut on non-null values to create deciles
    df['X_decile'] = pd.qcut(df[column_name].dropna()+ jitter(df[column_name].dropna()),
                             n_deciles,
                             labels=range(1, n_deciles + 1),
                             duplicates='drop')

    # Only add a 'Missing' category if there are NaN values
    if df[column_name].isna().any():
        df['X_decile'] = df['X_decile'].cat.add_categories(['Missing'])
        df['X_decile'] = df['X_decile'].fillna('Missing')   

    return df
def numeric_treecuts(df, column_name, Y):

    breakpoints = tree_cuts(df[df[column_name].notnull()],column_name, Y)
    # Apply the updated function to the 'X_max' column with the breakpoints DataFrame
    df['X_decile'] = df[column_name].apply(lambda x: find_group(x, breakpoints))
    # Only add a 'Missing' category if there are NaN values
    df.loc[df[column_name].isna(), "X_decile"]= "Missing"
    return df

def tree_cuts(df, X_var, Y_var): 
    X = df[[X_var]]  # Features
    y = df[Y_var]  # Target

    # Initialize and train the decision tree
    tree = DecisionTreeRegressor(random_state=42, min_samples_leaf=int(len(X)/10))
    tree.fit(X, y)

    # Extract information from the tree
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    values = tree.tree_.value
    impurities = tree.tree_.impurity
    samples = tree.tree_.n_node_samples

    # Initialize the table
    table = pd.DataFrame(columns=["min", "max", "sample", "value", "squared error"])

    # Function to traverse the tree and extract information for the table
    def recurse(node, lower_bound=float('-inf'), upper_bound=float('inf')):
        if children_left[node] == children_right[node]:  # It's a leaf
            value = values[node][0, 0]
            sample_count = samples[node]
            mse = impurities[node] * samples[node]  # Total squared error for this node
            table.loc[len(table)] = [lower_bound, upper_bound, sample_count, value, mse]
        else:
            # Continue the recursion on both children
            if children_left[node] != -1:
                recurse(children_left[node], lower_bound, threshold[node])
            if children_right[node] != -1:
                recurse(children_right[node], threshold[node], upper_bound)

    # Start recursion from root
    recurse(0)
    return table

def find_group(value, breakpoints):
    """
    Determines the group index for a given value based on specified breakpoints.

    Parameters:
    - value: The value to classify into a group.
    - breakpoints: DataFrame with 'min' and 'max' columns defining group boundaries.

    Returns:
    - The index of the group if a match is found, otherwise None.
    """
    for index, row in breakpoints.iterrows():
        if row['min'] <= value < row['max']:
            return index
    return None  # Return None if no group matches
def categorical_cuts(df, column_name):
    df['X_decile'] = df[column_name].astype(str)

    # Get unique categories of X_decile
    unique_deciles = df['X_decile'].unique()

    # Create an empty DataFrame with the specified columns and unique categories of X_decile
    columns = ['X_min', 'X_max', 'X_median', 'X_25%', 'X_75%']
    decile_stats = pd.DataFrame(index=unique_deciles, columns=columns)

    return decile_stats


def categorize_into_deciles_with_stats(df, column_name, Y, n_deciles=10, f_decile_tree =False):
    """
    Categorizes a specified column in a DataFrame into deciles (or specified quantiles) for numerical data,
    or uses existing categories for categorical data, and calculates various statistics for each group.

    Parameters:
    - df: Pandas DataFrame containing the data.
    - column_name: string, the name of the column to categorize.
    - Y: string, the name of another column to compute statistics for each group.
    - n_deciles: integer, default 10. Specifies the number of groups for numerical data.

    Returns:
    - DataFrame: Returns a DataFrame with group statistics.
    """
    # Check if the column is numeric or categorical
    if pd.api.types.is_numeric_dtype(df[column_name]):
        n_cuts = len(df[column_name].unique())
        print(n_cuts)
        
        # if too little numerical values, then it is treated as categorical
        if n_deciles>= n_cuts or n_cuts<=20:
            decile_stats = categorical_cuts(df, column_name)
        else:
            if f_decile_tree:
                df = numeric_treecuts(df, column_name, Y) 
            else:
                try: 
                    df = numeric_decilecuts(df, column_name, n_deciles=n_deciles-1)
                except:
                    df[column_name] =df[column_name].astype(str)
                    decile_stats = categorical_cuts(df, column_name)
            decile_stats = df.groupby('X_decile', observed=False)[column_name].agg([
            'min', 
            'max', 
            'median', 
            lambda x: x.quantile(0.25), 
            lambda x: x.quantile(0.75)
        ])
        # Rename the columns
        decile_stats.columns = ['X_min', 'X_max', 'X_median', 'X_25%', 'X_75%']
    
    else:
        
        decile_stats = categorical_cuts(df, column_name)
    decile_stats.reset_index()    

    # Calculate statistics for Y within each group
    y_stats = df.groupby('X_decile',dropna = False, observed=False)[Y].agg(['mean', 'std', 'median',
                                                            lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), 'count'])
    y_stats.columns = [f'{Y}_mean', f'{Y}_std', f'{Y}_median', f'{Y}_25%', f'{Y}_75%', 'n']
    total_sum = y_stats['n'].sum()
    y_stats['n_percentage'] = (y_stats['n'] / total_sum) * 100
    
    # Join decile_stats and y_stats
    combined_stats = decile_stats.join(y_stats)

    # Calculate overall mean for Y and discrepancy metrics
    overall_median_Y = df[Y].mean()
    combined_stats[f'gen_{Y}_mean'] = overall_median_Y
    combined_stats['discr'] = abs(combined_stats[f'{Y}_mean'] - overall_median_Y) / overall_median_Y
    combined_stats['max_discr'] = combined_stats['discr'].max()
    

    # Insert variable name at the start
    combined_stats.insert(0, 'varname', column_name)
    combined_stats.reset_index(inplace=True)
    combined_stats.columns = ['X_decile'] + combined_stats.columns[1:].tolist()
    combined_stats['X_min_str'] = combined_stats['X_min'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "")
    combined_stats['X_max_str'] = combined_stats['X_max'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "")
    # combined_stats.insert(0, 'X_decile', combined_stats.index)
    # Create x_string column based on the conditions
    combined_stats['x_string'] = np.where(
        combined_stats['X_min'].notna(),
        '[' + combined_stats['X_min_str'] + '-' + combined_stats['X_max_str'] + ']',
        combined_stats['X_decile']
    )
    combined_stats.drop(columns=['X_min_str',"X_max_str"], inplace=True)

    return combined_stats





# Updated plot_data_by_varname to work with subplots and add a table
def plot_data_by_varname(ax, df, var_name, Y):
    # ── colour palette ───────────────────────────────────────────────────
    COLOR_LINE = "#1f77b4"   # muted blue  – mean line
    COLOR_MEAN = "#d62728"   # brick red   – overall mean reference
    COLOR_BAR  = "#aec7e8"   # light blue  – n_percentage bars

    filtered_df = df[df['varname'] == var_name].copy()
    if filtered_df.empty:
        return

    filtered_df['x_string'] = filtered_df['x_string'].fillna('Missing').astype(str)
    gen_y_mean  = filtered_df[f'gen_{Y}_mean'].iloc[0]
    x_positions = list(range(len(filtered_df)))
    x_labels    = filtered_df['x_string'].tolist()

    # ── secondary axis – n_percentage histogram bars ─────────────────────
    n_pct_prop = filtered_df['n_percentage'] / 100   # convert % to proportion (0-1)
    ax2 = ax.twinx()
    ax2.bar(
        x_positions,
        n_pct_prop,
        color=COLOR_BAR,
        alpha=0.40,
        width=0.6,
        zorder=1,
        label='n (prop)'
    )
    ax2.set_ylabel('Population (proportion)', fontsize=9, color='grey')
    ax2.tick_params(axis='y', labelcolor='grey', labelsize=8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.set_ylim(0, n_pct_prop.max() * 3.5)   # push bars to bottom third
    ax2.spines['right'].set_color('lightgrey')
    ax2.spines['top'].set_visible(False)

    # ── primary axis – mean line + ±1 std band ───────────────────────────
    y_mean = filtered_df[f'{Y}_mean'].values
    y_std  = filtered_df[f'{Y}_std'].values

    ax.fill_between(
        x_positions,
        y_mean - y_std,
        y_mean + y_std,
        color=COLOR_LINE,
        alpha=0.12,
        zorder=2,
        label='±1 std'
    )
    ax.plot(
        x_positions,
        y_mean,
        marker='o',
        markersize=6,
        linewidth=2.5,
        color=COLOR_LINE,
        zorder=3,
        label=f'{Y} mean'
    )
    ax.axhline(
        y=gen_y_mean,
        color=COLOR_MEAN,
        linestyle='--',
        linewidth=1.5,
        zorder=4,
        label=f'Overall mean ({gen_y_mean:.2f})'
    )

    # ── axes styling ──────────────────────────────────────────────────────
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=40, ha='right', fontsize=8)
    ax.set_title(f'{Y} by {var_name}', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel(var_name, fontsize=10, labelpad=6)
    ax.set_ylabel(f'{Y} mean', fontsize=10, color=COLOR_LINE)
    ax.tick_params(axis='y', labelcolor=COLOR_LINE, labelsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, right=False, offset=6, trim=False)

    # ── combined legend ───────────────────────────────────────────────────
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2,
        labels1  + labels2,
        fontsize=8,
        loc='upper left',
        framealpha=0.7,
        edgecolor='lightgrey'
    )
    if ax2.get_legend():
        ax2.get_legend().remove()

    # ── summary table ─────────────────────────────────────────────────────
    fmt = lambda v: f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)
    col_labels = ['Bucket', f'{Y}_mean', f'{Y}_std', f'{Y}_median',
                  f'{Y}_25%', f'{Y}_75%', 'n', 'n%']
    table_rows = [
        [
            row['x_string'],
            fmt(row[f'{Y}_mean']),
            fmt(row[f'{Y}_std']),
            fmt(row[f'{Y}_median']),
            fmt(row[f'{Y}_25%']),
            fmt(row[f'{Y}_75%']),
            str(int(row['n'])),
            f"{row['n_percentage']:.1f}%"
        ]
        for _, row in filtered_df.iterrows()
    ]

    n_cols = len(col_labels)
    table  = Table(ax, bbox=[0, -0.42, 1, 0.28])
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)

    for i, label in enumerate(col_labels):
        cell = table.add_cell(0, i, width=1.0/n_cols, height=0.12,
                              text=label, loc='center', facecolor='#4472C4')
        cell.get_text().set_color('white')
        cell.get_text().set_fontweight('bold')

    for row_idx, row in enumerate(table_rows):
        bg = '#EEF2FF' if row_idx % 2 == 0 else 'white'
        for col_idx, cell_value in enumerate(row):
            table.add_cell(row_idx + 1, col_idx, width=1.0/n_cols, height=0.12,
                           text=cell_value, loc='center', facecolor=bg)

    ax.add_table(table)


def plot_data_by_varname_plotly(df, var_name, Y):
    """Interactive plotly version of plot_data_by_varname.

    Returns a plotly Figure with:
    - Left axis : Y mean line + ±1 std shaded band + overall mean reference
    - Right axis: n_percentage bars (histogram style)
    - Summary table below the chart
    """
    filtered_df = df[df['varname'] == var_name].copy()
    if filtered_df.empty:
        return go.Figure()

    filtered_df['x_string'] = filtered_df['x_string'].fillna('Missing').astype(str)
    gen_y_mean = filtered_df[f'gen_{Y}_mean'].iloc[0]
    x_labels   = filtered_df['x_string'].tolist()
    y_mean     = filtered_df[f'{Y}_mean'].values
    y_std      = filtered_df[f'{Y}_std'].values

    # ── layout: chart (row 1) + table (row 2) ────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.06,
        specs=[[{"secondary_y": True}],
               [{"secondary_y": False}]],
    )

    # ── ±1 std band ───────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=x_labels + x_labels[::-1],
            y=list(y_mean + y_std) + list((y_mean - y_std)[::-1]),
            fill='toself',
            fillcolor='rgba(31,119,180,0.12)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=True,
            name='±1 std',
        ),
        row=1, col=1, secondary_y=False,
    )

    # ── mean line ─────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=y_mean,
            mode='lines+markers',
            line=dict(color='#1f77b4', width=2.5),
            marker=dict(size=7, color='#1f77b4'),
            name=f'{Y} mean',
            hovertemplate='<b>%{x}</b><br>mean: %{y:.2f}<extra></extra>',
        ),
        row=1, col=1, secondary_y=False,
    )

    # ── overall mean reference line ───────────────────────────────────────
    fig.add_hline(
        y=gen_y_mean,
        line=dict(color='#d62728', width=1.5, dash='dash'),
        annotation_text=f'Overall mean ({gen_y_mean:.2f})',
        annotation_position='top right',
        annotation_font=dict(color='#d62728', size=10),
        row=1, col=1,
    )

    # ── n_percentage bars (secondary y) ───────────────────────────────────
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=filtered_df['n_percentage'].values,
            name='n %',
            marker=dict(color='#aec7e8', opacity=0.45),
            hovertemplate='<b>%{x}</b><br>n%%: %{y:.1f}%%<extra></extra>',
            yaxis='y2',
        ),
        row=1, col=1, secondary_y=True,
    )

    # ── summary table ─────────────────────────────────────────────────────
    fmt = lambda v: f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)
    col_labels = ['Bucket', f'{Y}_mean', f'{Y}_std', f'{Y}_median',
                  f'{Y}_25%', f'{Y}_75%', 'n', 'n%']
    cell_values = [
        filtered_df['x_string'].tolist(),
        [fmt(v) for v in filtered_df[f'{Y}_mean']],
        [fmt(v) for v in filtered_df[f'{Y}_std']],
        [fmt(v) for v in filtered_df[f'{Y}_median']],
        [fmt(v) for v in filtered_df[f'{Y}_25%']],
        [fmt(v) for v in filtered_df[f'{Y}_75%']],
        [str(int(v)) for v in filtered_df['n']],
        [f"{v:.1f}%" for v in filtered_df['n_percentage']],
    ]
    n_rows = len(filtered_df)
    row_fill = ['#EEF2FF' if i % 2 == 0 else 'white' for i in range(n_rows)]

    fig.add_trace(
        go.Table(
            header=dict(
                values=[f'<b>{c}</b>' for c in col_labels],
                fill_color='#4472C4',
                font=dict(color='white', size=11),
                align='center',
                height=28,
            ),
            cells=dict(
                values=cell_values,
                fill_color=[row_fill] * len(col_labels),
                align='center',
                font=dict(size=10),
                height=24,
            ),
        ),
        row=2, col=1,
    )

    # ── axes & layout polish ──────────────────────────────────────────────
    fig.update_yaxes(
        title_text=f'{Y} mean',
        title_font=dict(color='#1f77b4'),
        tickfont=dict(color='#1f77b4'),
        gridcolor='rgba(0,0,0,0.08)',
        row=1, col=1, secondary_y=False,
    )
    fig.update_yaxes(
        title_text='Population (%)',
        title_font=dict(color='grey'),
        tickfont=dict(color='grey'),
        showgrid=False,
        range=[0, filtered_df['n_percentage'].max() * 3.5],
        row=1, col=1, secondary_y=True,
    )
    fig.update_xaxes(tickangle=-40, row=1, col=1)
    fig.update_layout(
        title=dict(text=f'<b>{Y} by {var_name}</b>', font=dict(size=15), x=0.05),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(t=80, b=20, l=60, r=60),
        bargap=0.3,
        hovermode='x unified',
    )

    return fig
