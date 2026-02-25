import flet as ft
import numpy as np
from kinetics_logic import fit_data
import re
import matplotlib
import matplotlib.pyplot as plt
from flet.matplotlib_chart import MatplotlibChart
import os

matplotlib.use("svg")

def main(page: ft.Page):
    page.title = "Enzyme Kinetics Analyzer"
    page.window_icon = "icon-192.png"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 20
    page.spacing = 20
    
    # --- Help Dialog ---
    
    import io
    import base64

    def latex_to_base64(formula, fontsize=18):
        """Renders LaTeX formula to a base64 encoded PNG for Flet."""
        plt.figure(figsize=(0.1, 0.1)) # Tiny initial figure
        # Create a text-only figure
        fig = plt.figure(figsize=(5, 0.6))
        fig.text(0, 0.5, f"${formula}$", fontsize=fontsize, va='center', ha='left')
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        plt.close() # Close both
        
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    # Pre-render equations
    mm_img = latex_to_base64(r"v = \frac{V_{max} [S]}{K_m + [S]}")
    inh_img = latex_to_base64(r"v = \frac{V_{max} [S]}{K_m + [S] + \frac{[S]^2}{K_i}}")
    comp_img = latex_to_base64(r"v = \frac{V_{max} [S]}{K_m(1 + \frac{[I]}{K_i}) + [S]}")
    uncomp_img = latex_to_base64(r"v = \frac{V_{max} [S]}{K_m + [S](1 + \frac{[I]}{K_i})}")
    noncomp_img = latex_to_base64(r"v = \frac{V_{max} [S]}{(1 + \frac{[I]}{K_i})(K_m + [S])}")
    mixed_img = latex_to_base64(r"v = \frac{V_{max} [S]}{K_m(1 + \frac{[I]}{K_i}) + [S](1 + \frac{[I]}{\alpha K_i})}")

    help_controls = [
        ft.Text("USER GUIDE", size=24, weight=ft.FontWeight.BOLD, color=ft.colors.BLUE_700),
        ft.Text("This application analyzes enzyme kinetics data to determine key parameters (Vmax, Km, Ki) using non-linear regression.", size=14),
        ft.Divider(),
        
        ft.Text("KINETIC MODELS", size=18, weight=ft.FontWeight.BOLD),
        
        ft.Text("1. Michaelis-Menten", weight=ft.FontWeight.BOLD),
        ft.Image(src_base64=mm_img, height=50),
        
        ft.Text("2. Substrate Inhibition (Haldane)", weight=ft.FontWeight.BOLD),
        ft.Image(src_base64=inh_img, height=60),
        
        ft.Text("3. Competitive Inhibition", weight=ft.FontWeight.BOLD),
        ft.Image(src_base64=comp_img, height=60),
        
        ft.Text("4. Uncompetitive Inhibition", weight=ft.FontWeight.BOLD),
        ft.Image(src_base64=uncomp_img, height=60),
        
        ft.Text("5. Noncompetitive (Pure) Inhibition", weight=ft.FontWeight.BOLD),
        ft.Image(src_base64=noncomp_img, height=60),
        
        ft.Text("6. Mixed Inhibition", weight=ft.FontWeight.BOLD),
        ft.Image(src_base64=mixed_img, height=60),
        
        ft.Divider(),
        ft.Text("DATA FORMAT", size=18, weight=ft.FontWeight.BOLD),
        ft.Text("Standard (non-inhibition):", weight=ft.FontWeight.W_500),
        ft.Text("[S], Rate\n1.0, 5.0\n2.0, 8.5", bgcolor=ft.colors.GREY_100, font_family="Consolas"),
        
        ft.Text("Matrix (inhibition):", weight=ft.FontWeight.W_500),
        ft.Text("[S], I_0, I_10, I_50\n1.0, 5.0, 4.2, 2.1\n2.0, 8.5, 7.1, 4.0", bgcolor=ft.colors.GREY_100, font_family="Consolas"),
        ft.Divider(),
        ft.Text("WEIGHTING", size=18, weight=ft.FontWeight.BOLD),
        ft.Text("• None: Ordinary Least Squares.\n• 1/y: Errors scale with rate (Poisson).\n• 1/y²: Consistent relative error (weights low rates heavily).", size=14),
    ]

    def close_help(e):
        help_dialog.open = False
        page.update()

    help_dialog = ft.AlertDialog(
        modal=False,
        title=ft.Text("Help & Documentation"),
        content=ft.Container(
            content=ft.Column(help_controls, scroll=ft.ScrollMode.AUTO),
            width=600,
            height=400,
            padding=10,
        ),
        actions=[
            ft.TextButton("Close", on_click=close_help),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )

    def open_help(e):
        page.dialog = help_dialog
        help_dialog.open = True
        page.update()

    # --- UI Components ---
    
    # Title Row
    header = ft.Row([
        ft.Text(
            "Enzyme Kinetics Analyzer",
            size=32,
            weight=ft.FontWeight.BOLD,
            color=ft.colors.BLUE_700,
            expand=True
        ),
        ft.IconButton(
            icon=ft.icons.HELP_OUTLINE, 
            icon_size=30,
            tooltip="User Guide",
            on_click=open_help
        )
    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
    
    # Input Area
    data_input = ft.TextField(
        label="Enter Data",
        multiline=True,
        min_lines=10,
        max_lines=10, 
        text_size=12,
        hint_text="[S], Rate\nOR\n[S], I1, I2... (Matrix)",
        expand=True,
        text_style=ft.TextStyle(font_family="Consolas")
    )
    
    # Settings
    model_dropdown = ft.Dropdown(
        label="Model",
        options=[
            ft.dropdown.Option("michaelis_menten", "Michaelis-Menten"),
            ft.dropdown.Option("substrate_inhibition", "Substrate Inhibition"),
            ft.dropdown.Option("competitive", "Competitive Inhibition"),
            ft.dropdown.Option("uncompetitive", "Uncompetitive Inhibition"),
            ft.dropdown.Option("noncompetitive", "Noncompetitive (Pure)"),
            ft.dropdown.Option("mixed", "Mixed Inhibition"),
        ],
        value="michaelis_menten",
        width=300
    )

    weight_dropdown = ft.Dropdown(
        label="Weighting",
        options=[
            ft.dropdown.Option("None", "None (OLS)"),
            ft.dropdown.Option("1/y", "1/y (Poisson)"),
            ft.dropdown.Option("1/y2", "1/y² (Relative)"),
        ],
        value="None",
        width=150
    )

    unit_s = ft.TextField(label="[S] Unit", value="µM", width=100)
    unit_v = ft.TextField(label="Rate Unit", value="µM/sec", width=100)

    # Matplotlib Figure
    fig = plt.figure(figsize=(7, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
    ax_main = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1])
    
    chart = MatplotlibChart(fig, expand=True)

    results_text = ft.Column([
        ft.Text("Results will appear here", size=16, color=ft.colors.GREY_700),
    ], scroll=ft.ScrollMode.AUTO)

    # --- Functions ---

    def parse_data_matrix(raw_text, model_type="michaelis_menten"):
        """
        Parses text data. Supports:
        1. Simple 2-column: [S], v
        2. Matrix with header: [S], i1, i2... \n s1, v1_1, v1_2...
        
        If model_type is simple (Michaelis-Menten or Substrate Inhibition), 
        it forces parsing of only the first 2 columns ([S], v) and ignores headers/matrix structure.
        """
        if not raw_text: return [], [], []
        
        lines = [line.strip() for line in raw_text.strip().split('\n') if line.strip() and not line.startswith('#')]
        if not lines: return [], [], []

        simple_models = ["michaelis_menten", "substrate_inhibition"]
        force_simple = model_type in simple_models

        # Check for header row or 2-col vs multi-col
        first_line_parts = re.split(r'[,\t\s]+', lines[0])
        is_matrix = False
        inhibitor_cols = []
        
        if not force_simple:
            # Heuristic: verify if it looks like a matrix header
            try:
                potential_i_strs = first_line_parts[1:]
                potential_i = []
                valid_header = True
                for s in potential_i_strs:
                    try:
                        potential_i.append(float(s))
                    except ValueError:
                        valid_header = False
                        break
                
                if valid_header and len(potential_i) > 0:
                    is_matrix = True
                    inhibitor_cols = potential_i
                    data_lines = lines[1:]
                else:
                    data_lines = lines
                    
            except Exception:
                data_lines = lines
        else:
            # For simple models, treat everything as data lines, ignore potential header if it fails float conversion
            # But if the user pasted a matrix with a header, the first line might be "[S], 0, 10" which fails float conv for [S]
            # So we should probably try to skip the first line if it looks like a header (non-numeric first token)
            try:
                float(first_line_parts[0])
                data_lines = lines
            except ValueError:
                # First token not a number, likely a header like "[S]"
                data_lines = lines[1:]

        s_list = []
        v_list = []
        i_list = []

        for line in data_lines:
            parts = re.split(r'[,\t\s]+', line)
            parts = [p for p in parts if p]
            
            if len(parts) < 2: continue
            
            try:
                s_val = float(parts[0])
                
                if is_matrix:
                    # Matrix mode
                    for idx, i_val in enumerate(inhibitor_cols):
                        if idx + 1 < len(parts):
                            v_str = parts[idx + 1]
                            v_val = float(v_str)
                            s_list.append(s_val)
                            v_list.append(v_val)
                            i_list.append(i_val)
                else:
                    # Simple mode: s, v. Ignore extra columns.
                    v_val = float(parts[1])
                    i_val = 0.0 # No inhibitor for simple models
                    s_list.append(s_val)
                    v_list.append(v_val)
                    i_list.append(i_val)
                    
            except ValueError:
                continue
                
        return s_list, v_list, i_list

    def calculate_kinetics(e):
        
        # Get settings first to determine parsing mode
        model = model_dropdown.value
        weighting = weight_dropdown.value if weight_dropdown.value != "None" else None

        raw_text = data_input.value
        s_data, v_data, i_data = parse_data_matrix(raw_text, model_type=model)
        
        if len(s_data) < 3:
            page.show_snack_bar(ft.SnackBar(content=ft.Text("Need at least 3 valid points")))
            return

        # Get settings
        model = model_dropdown.value
        weighting = weight_dropdown.value if weight_dropdown.value != "None" else None
        
        # Check if inhibition model selected but no varying I
        if model in ['competitive', 'uncompetitive', 'noncompetitive', 'mixed']:
            unique_i = set(i_data)
            if len(unique_i) < 2 and list(unique_i)[0] == 0:
                 page.show_snack_bar(ft.SnackBar(content=ft.Text("Warning: Inhibition model selected but no Inhibitor data found (I=0).")))

        results = fit_data(s_data, v_data, i_data, model_type=model, weighting=weighting, robust=False)
        
        if results:
            # Prepare Report
            lines = []
            
            def format_val_err(val, err):
                if err is None or np.isnan(err):
                    return f"{val:.4f} ± Undefined"
                if err == 0:
                    # Only happens if DOFs < 1 or perfect fit
                    return f"{val:.4f} ± 0.0000"
                return f"{val:.4f} ± {err:.4f}"

            vmax = results.get('vmax', 0)
            km = results.get('km', 0)
            lines.append(f"Vmax: {format_val_err(vmax, results.get('vmax_err'))} {unit_v.value}")
            lines.append(f"Km:   {format_val_err(km, results.get('km_err'))} {unit_s.value}")
            
            if 'ki' in results:
                 lines.append(f"Ki:   {format_val_err(results['ki'], results.get('ki_err'))} {unit_s.value}")
            
            if 'alpha' in results:
                 lines.append(f"Alpha:{results['alpha']:.4f}")
                 if 'ki_prime' in results:
                     ki_p = results['ki_prime']
                     # Rough error propagation for Ki' if possible, else just value
                     lines.append(f"Ki':  {ki_p:.4f}")

            lines.append("-" * 20)
            lines.append(f"R²:  {results.get('r_squared', 0):.4f}")
            lines.append(f"AIC: {results.get('aic', 0):.2f}")
            lines.append(f"RSS: {results.get('rss', 0):.4e}")
            
            # Update UI Text
            results_controls = [ft.Text("Analysis Results", size=18, weight=ft.FontWeight.BOLD)]
            results_controls.append(ft.Text(f"Model: {model}", size=14, italic=True))
            for line in lines:
                results_controls.append(ft.Text(line, size=15, font_family="Consolas"))
            
            results_text.controls = results_controls
            
            # --- Update Plots ---
            ax_main.clear()
            ax_res.clear()
            
            # Formatting
            ax_main.set_ylabel(f"Rate ({unit_v.value})")
            ax_main.set_title(f"Kinetics Fit ({model})")
            ax_main.grid(True, linestyle='--', alpha=0.5)
            ax_res.set_xlabel(f"Concentration [{unit_s.value}]")
            ax_res.set_ylabel("Resid.")
            ax_res.axhline(0, color='black', linewidth=0.8)
            ax_res.grid(True, linestyle='--', alpha=0.5)

            # --- Plot Data and Fits ---
            
            # Organize data by inhibitor concentration for clearer plotting
            unique_i = sorted(list(set(i_data)))
            # Colormap
            cmap = matplotlib.cm.get_cmap('viridis')
            norm = matplotlib.colors.Normalize(vmin=min(unique_i), vmax=max(unique_i))
            
            # If too many unique I, fallback to simple scatter
            if len(unique_i) > 10:
                ax_main.plot(s_data, v_data, 'o', color='gray', alpha=0.5)
            else:
                for idx, i_val in enumerate(unique_i):
                    # Filter data
                    # Use numpy-like comparison for float safety
                    mask = [abs(x - i_val) < 1e-9 for x in i_data]
                    s_sub = [s for s, m in zip(s_data, mask) if m]
                    v_sub = [v for v, m in zip(v_data, mask) if m]
                    
                    if len(unique_i) > 1:
                        color = cmap(norm(i_val))
                        label = f"I={i_val:g}"
                    else:
                        color = 'tab:blue'
                        label = "Data"
                    
                    ax_main.plot(s_sub, v_sub, 'o', label=label, color=color, alpha=0.7)
                    
                    # Plot Fit Curve (if available)
                    # We need to find the matching curve in results['fitted_curves'] or results['fitted_curve']
                    
                    if 'fitted_curves' in results:
                        # Find closest key in dict
                        curves = results['fitted_curves']
                        # key search
                        matched_key = None
                        for k in curves.keys():
                            if abs(k - i_val) < 1e-9:
                                matched_key = k
                                break
                        
                        if matched_key is not None:
                            xs, ys = curves[matched_key]
                            ax_main.plot(xs, ys, '-', color=color, alpha=0.9)

                    elif 'fitted_curve' in results:
                        # Single curve models (MM / Substrate Inh)
                        # Only plot once
                        if idx == 0:
                            xs, ys = results['fitted_curve']
                            ax_main.plot(xs, ys, '-', color='tab:red', label='Fit')

            if len(unique_i) <= 10:
                ax_main.legend()

            # --- Plot Residuals ---
            residuals = results.get('residuals', [])
            if residuals:
                # Need to sort s_data for stem plot?
                # Stem plot x=s. Scatter is better if unordered.
                # Let's use scatter for residuals
                 ax_res.scatter(s_data, residuals, alpha=0.6, color='tab:purple')
                 # Add zero line
                 ax_res.axhline(0, color='black', alpha=0.5)

            chart.update()
            page.update()
            
        else:
             page.show_snack_bar(ft.SnackBar(content=ft.Text("Fitting failed. Check data format.")))

    # --- Layout ---
    
    calc_button = ft.ElevatedButton("Calculate", icon=ft.icons.CALCULATE, 
                                    on_click=calculate_kinetics,
                                    style=ft.ButtonStyle(bgcolor=ft.colors.BLUE_600, color=ft.colors.WHITE))
                                    
    # Disable expand on data_input to avoid layout issues in the nested structure
    data_input.expand = False

    left_col = ft.Column([
        ft.Row([
            # Data Section
            ft.Column([
                ft.Text("Data", size=16, weight=ft.FontWeight.W_500),
                data_input
            ], width=300), # Fixed smaller width for data

            ft.VerticalDivider(width=20),

            # Settings Section
            ft.Column([
                ft.Text("Settings", size=16, weight=ft.FontWeight.W_500),
                model_dropdown,
                weight_dropdown,
                ft.Row([unit_s, unit_v]),
                ft.Divider(),
                calc_button,
            ], expand=True), # Settings takes remaining space
        ], vertical_alignment=ft.CrossAxisAlignment.START),
        
        ft.Divider(),
        
        ft.Container(height=10),
        ft.Container(
            content=results_text,
            padding=10,
            bgcolor=ft.colors.BLUE_GREY_50,
            border_radius=8,
            border=ft.border.all(1, ft.colors.BLUE_GREY_100),
            height=200, 
            width=400
        )
    ], width=700, spacing=10)

    right_col = ft.Column([
        ft.Text("Plots", size=20, weight=ft.FontWeight.W_500),
        ft.Container(
            content=chart,
            padding=5,
            bgcolor=ft.colors.WHITE,
            border_radius=10,
            border=ft.border.all(1, ft.colors.GREY_300),
            expand=True
        )
    ], expand=True)

    footer = ft.Row([
        ft.Text("Developed in Molecular Microbiology Laboratory, Dept of Bioscience and Biotechnology, IIT Kharagpur", 
               size=10, color=ft.colors.GREY_500)
    ], alignment=ft.MainAxisAlignment.END)

    page.add(
        header,
        ft.Row([left_col, ft.VerticalDivider(width=1), right_col], 
               expand=True, 
               alignment=ft.MainAxisAlignment.START, 
               vertical_alignment=ft.CrossAxisAlignment.START),
        footer
    )

ft.app(target=main, assets_dir=".")
