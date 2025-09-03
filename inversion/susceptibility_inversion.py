import os, warnings, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from discretize import TensorMesh
from discretize.utils import active_from_xyz
from simpeg.potential_fields import magnetics
from simpeg import data, maps, data_misfit, regularization, optimization, inverse_problem, inversion, directives
from PIL import Image
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def run_3d_susceptibility_inversion(
    csv_path: str,
    out_dir: str,
    cols: dict,
    B_total: float = 50000.0,
    incl: float = 10.0,
    decl: float = 0.0,
    target_cells_xy: int = 32,
    target_cells_z: int = 16,
    top_buffer_m: float = 50.0,
    bottom_depth_m: float = 600.0,
    padding_m: float = 200.0,
    noise_floor: float = 1.5,
    max_iter: int = 10,
):
    """
    Run 3D magnetic susceptibility inversion using SimPEG.
    """
    os.makedirs(out_dir, exist_ok=True)
    warnings.filterwarnings("ignore")
    
    # --- Load data ---
    df = pd.read_csv(csv_path)
    need = [cols["x"], cols["y"], cols["z"], cols["tmi"]]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing column in CSV: {c}")
    df = df.dropna(subset=need).copy()
    logging.info(f"N points = {len(df)} | lines = {df[cols['line']].nunique() if cols['line'] in df.columns else 'NA'}")

    # --- Demean per line + robust clip ---
    if cols["line"] in df.columns:
        df["tmi_demean"] = df.groupby(cols["line"])[cols["tmi"]].transform(lambda s: s - s.mean())
    else:
        df["tmi_demean"] = df[cols["tmi"]] - df[cols["tmi"]].mean()
    p_lo, p_hi = np.percentile(df["tmi_demean"], [0.2, 99.8])
    df["tmi_use"] = df["tmi_demean"].clip(p_lo, p_hi)

    # --- Robust per-line sigma weights ---
    if cols["line"] in df.columns:
        mad_sigma = df.groupby(cols["line"])["tmi_use"].transform(lambda s: 1.4826*(s - s.median()).abs().median())
    else:
        mad_sigma = pd.Series(1.4826*(df["tmi_use"] - df["tmi_use"].median()).abs().median(), index=df.index)
    wd = np.clip(mad_sigma.to_numpy(float), noise_floor, 8.0)

    # --- Remove weak regional plane ---
    Xreg = np.c_[np.ones(len(df)), df[cols["x"]].values, df[cols["y"]].values]
    coef, *_ = np.linalg.lstsq(Xreg, df["tmi_use"].values, rcond=None)
    plane = Xreg @ coef
    dobs_clean = (df["tmi_use"].values - plane).astype(float)
    rx_locs = df[[cols["x"], cols["y"], cols["z"]]].to_numpy(float)

    # --- Mesh ---
    x_min, x_max = df[cols["x"]].min(), df[cols["x"]].max()
    y_min, y_max = df[cols["y"]].min(), df[cols["y"]].max()
    z_min, z_max = df[cols["z"]].min(), df[cols["z"]].max()

    x0, x1 = x_min - padding_m, x_max + padding_m
    y0, y1 = y_min - padding_m, y_max + padding_m
    surf_z = z_min - 5.0
    top_z = z_max + top_buffer_m
    bot_z = surf_z - bottom_depth_m

    hx = np.ones(target_cells_xy) * ((x1 - x0)/target_cells_xy)
    hy = np.ones(target_cells_xy) * ((y1 - y0)/target_cells_xy)
    hz = np.ones(target_cells_z) * ((top_z - bot_z)/target_cells_z)
    mesh = TensorMesh([hx, hy, hz], x0=(x0, y0, bot_z))
    logging.info(f"[mesh] cells={mesh.nC} | dx={hx[0]:.2f}, dy={hy[0]:.2f}, dz={hz[0]:.2f}")

    # --- Active cells (topo-aware) ---
    xy_round = df[[cols["x"], cols["y"]]].round(1).to_numpy()
    z_vals = df[cols["z"]].to_numpy()
    topo_df = pd.DataFrame({"x":xy_round[:,0],"y":xy_round[:,1],"z":z_vals}).groupby(["x","y"], as_index=False)["z"].min()
    topo_xyz = topo_df[["x","y","z"]].to_numpy(float)
    actv = active_from_xyz(mesh, topo_xyz, grid_reference="CC")
    logging.info(f"[active] {actv.sum()} / {mesh.nC}")

    # --- Survey ---
    receiver = magnetics.receivers.Point(rx_locs, components=["tmi"])
    try:
        source = magnetics.sources.UniformBackgroundField(receiver_list=[receiver], amplitude=B_total,
                                                          inclination=incl, declination=decl)
    except TypeError:
        source = magnetics.sources.UniformBackgroundField(receiver_list=[receiver], parameters=np.r_[B_total, incl, decl])
    survey = magnetics.survey.Survey(source)
    chi_map = maps.IdentityMap(nP=int(actv.sum()))

    # --- Forward simulation ---
    store_mode = "ram"
    try:
        problem = magnetics.simulation.Simulation3DIntegral(survey=survey, mesh=mesh, chiMap=chi_map,
                                                            active_cells=actv, store_sensitivities=store_mode)
        _ = problem.dpred(np.zeros(int(actv.sum())))
    except Exception as e:
        logging.warning(f"[sens] RAM failed ({e}); using disk.")
        store_mode = "disk"
        problem = magnetics.simulation.Simulation3DIntegral(survey=survey, mesh=mesh, chiMap=chi_map,
                                                            active_cells=actv, store_sensitivities=store_mode)

    # --- Misfit + regularization ---
    data_obj = data.Data(survey, dobs=dobs_clean, standard_deviation=wd)
    dmis = data_misfit.L2DataMisfit(data=data_obj, simulation=problem)
    reg = regularization.WeightedLeastSquares(mesh, active_cells=actv)
    reg.alpha_s, reg.alpha_x, reg.alpha_y, reg.alpha_z = 0.06, 1.0, 0.7, 0.8

    # --- Optimization + inversion ---
    opt = optimization.ProjectedGNCG(maxIter=max_iter, lower=0.0, upper=0.1, maxIterCG=20, tolG=1e-3)
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
    dirs = [directives.UpdateSensitivityWeights(),
            directives.BetaEstimate_ByEig(beta0_ratio=1.0),
            directives.BetaSchedule(coolingFactor=2.0, coolingRate=1),
            directives.TargetMisfit(chifact=1.0)]
    inv = inversion.BaseInversion(inv_prob, directiveList=dirs)

    m0 = np.full(int(actv.sum()), 1e-4, dtype=float)
    mrec = inv.run(m0)
    np.save(os.path.join(out_dir, "chi_coarse.npy"), mrec)
    logging.info("[done] inversion complete")

    # --- QC plots ---
    dpred_clean = problem.dpred(mrec)
    dpred = dpred_clean + plane
    dobs = df["tmi_use"].to_numpy(float)
    rmse = float(np.sqrt(np.mean((dpred - dobs)**2)))

    logging.info(f"RMSE predicted vs observed: {rmse:.2f} nT")

    plt.figure(figsize=(6,5))
    plt.scatter(dobs, dpred, s=6, alpha=0.6)
    lim = np.percentile(np.r_[dobs, dpred],[1,99])
    plt.plot(lim, lim,"k--", lw=1)
    plt.xlabel("Observed TMI (nT)")
    plt.ylabel("Predicted TMI (nT)")
    plt.title(f"Predicted vs Observed (RMSE={rmse:.2f} nT)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"qc_pred_vs_obs.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7,6))
    vmin, vmax = np.percentile(dobs, [1,99])
    plt.scatter(df[cols["x"]], df[cols["y"]], c=dpred, s=6, vmin=vmin, vmax=vmax, cmap="viridis")
    plt.colorbar(label="Predicted TMI (nT)")
    plt.xlabel("Easting"); plt.ylabel("Northing")
    plt.title("Predicted TMI map")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"qc_map.png"), dpi=150)
    plt.close()

    logging.info(f"[outputs] {out_dir} | sensitivities stored in: {store_mode}")
    return mrec, mesh, actv

import os
import logging
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def overlay_terrain_chi(
    xyz_path: str = None,
    photo_path: str = None,
    photo_path2: str = None,
    html_out: str = None,
    title: str = "terrain_overlay.html",
    mesh=None,
    actv=None,
    mrec=None,
    IMG_MAX: int = 768,
    PT_KEEP: float = 1.0,
    DSM_BIN: float = 0.0,
    MODEL_STRIDE: tuple = (1, 1, 1),
    FLIP_V: bool = True,
    BORDER_EPS: float = 1e-3,
    TERRAIN_OPACITY_LEFT: float = 1.0,
    TERRAIN_OPACITY_RIGHT: float = 1.0,
    TERRAIN_LIGHT: dict = dict(ambient=0.65, diffuse=0.6, specular=0.05),
    ISO_LEVEL_LEFT: float = 0.0035,
    ISO_SURFACES_LEFT: int = 1,
    ISO_COLORS_LEFT: str = "Hot",
    ISO_OPACITY_LEFT: float = 1.0,
    FULL_MODE: str = "isosurface",
    FULL_ISO_SURFACES: int = 25,
    FULL_ISO_OPACITY: float = 0.25,
    FULL_PERC_RANGE: tuple = (5.0, 99.0)
):
    """
    Generate side-by-side Plotly visualization of terrain (DSM + image) 
    and 3D magnetic susceptibility. Highlights χ >= ISO_LEVEL_LEFT without
    filtering full model, to save memory.
    """
    
    terrain_left = terrain_right = None
    X = Y = Z = VAL_clip = VAL_left = np.array([])  # placeholders

    # -----------------------------
    # 1) Terrain DSM + Orthophoto
    # -----------------------------
    if xyz_path is not None and photo_path is not None:
        logging.info("Loading XYZ terrain and orthophoto...")
        xyz = np.loadtxt(xyz_path, delimiter=",").astype(np.float32)

        # Optional DSM thinning
        if DSM_BIN > 0:
            xmin, ymin = xyz[:,0].min(), xyz[:,1].min()
            gx = np.floor((xyz[:,0] - xmin) / DSM_BIN).astype(np.int32)
            gy = np.floor((xyz[:,1] - ymin) / DSM_BIN).astype(np.int32)
            keys = (gx.astype(np.int64) << 32) | gy.astype(np.int64)
            _, keep_idx = np.unique(keys, return_index=True)
            xyz = xyz[np.sort(keep_idx)]
        elif 0 < PT_KEEP < 1.0:
            rng = np.random.default_rng(42)
            idx = rng.choice(xyz.shape[0], size=int(xyz.shape[0]*PT_KEEP), replace=False)
            xyz = xyz[np.sort(idx)]

        x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
        tri = Delaunay(np.c_[x, y])

        # Load orthophoto
        im = Image.open(photo_path).convert("RGB")
        w, h = im.size
        scale = min(IMG_MAX / max(w, h), 1.0)
        if scale < 1.0:
            im = im.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        ARR = np.asarray(im).astype(np.uint8)
        H, W, _ = ARR.shape

        # UV mapping
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        u = (x - xmin) / max(1e-9, (xmax - xmin))
        v = (y - ymin) / max(1e-9, (ymax - ymin))
        v_img = 1.0 - v if FLIP_V else v
        px = np.clip((u * (W-1)).round().astype(np.int32), 0, W-1)
        py = np.clip((v_img * (H-1)).round().astype(np.int32), 0, H-1)
        vertex_rgb = ARR[py, px]

        # Trim triangles outside image
        uc = (u[tri.simplices.T[0]] + u[tri.simplices.T[1]] + u[tri.simplices.T[2]]) / 3.0
        vc = (v_img[tri.simplices.T[0]] + v_img[tri.simplices.T[1]] + v_img[tri.simplices.T[2]]) / 3.0
        inside_tri = (uc >= BORDER_EPS) & (uc <= 1.0 - BORDER_EPS) & (vc >= BORDER_EPS) & (vc <= 1.0 - BORDER_EPS)
        i2, j2, k2 = tri.simplices.T[0][inside_tri], tri.simplices.T[1][inside_tri], tri.simplices.T[2][inside_tri]

        terrain_left = go.Mesh3d(
            x=x, y=y, z=z, i=i2, j=j2, k=k2,
            vertexcolor=vertex_rgb,
            opacity=TERRAIN_OPACITY_LEFT,
            lighting=TERRAIN_LIGHT,
            flatshading=False,
            showscale=False,
            name="DSM + Image"
        )
        terrain_right = terrain_left.update(opacity=TERRAIN_OPACITY_RIGHT)

    # -----------------------------
    # 2) Magnetic Susceptibility
    # -----------------------------
    vmin_full = vmax_full = iso_max_left = ISO_LEVEL_LEFT
    if mesh is not None and actv is not None and mrec is not None:
        logging.info("Preparing χ model overlay...")
        chi_full = np.zeros(mesh.nC, dtype=np.float32)
        chi_full[actv] = mrec.astype(np.float32)

        nx, ny, nz = mesh.shape_cells
        CHI = chi_full.reshape((nx, ny, nz), order="F")
        ACT = actv.reshape((nx, ny, nz), order="F")

        ix = np.where(ACT.any(axis=(1,2)))[0]; sx = slice(ix.min(), ix.max()+1, MODEL_STRIDE[0])
        iy = np.where(ACT.any(axis=(0,2)))[0]; sy = slice(iy.min(), iy.max()+1, MODEL_STRIDE[1])
        iz = np.where(ACT.any(axis=(0,1)))[0]; sz = slice(iz.min(), iz.max()+1, MODEL_STRIDE[2])
        V = CHI[sx, sy, sz].copy().astype(np.float32)

        x0, y0, z0 = mesh.x0
        hx, hy, hz = mesh.h
        xe, ye, ze = x0 + np.r_[0.0, np.cumsum(hx)], y0 + np.r_[0.0, np.cumsum(hy)], z0 + np.r_[0.0, np.cumsum(hz)]
        xc = 0.5*(xe[:-1] + xe[1:])[sx].astype(np.float32)
        yc = 0.5*(ye[:-1] + ye[1:])[sy].astype(np.float32)
        zc = 0.5*(ze[:-1] + ze[1:])[sz].astype(np.float32)
        Xd, Yd, Zd = np.meshgrid(xc, yc, zc, indexing="ij")
        X, Y, Z = Xd.ravel(order="F"), Yd.ravel(order="F"), Zd.ravel(order="F")
        VAL = V.ravel(order="F")

        # Save full model CSV
        if html_out is not None:
            os.makedirs(html_out, exist_ok=True)
            df_all = pd.DataFrame({
                "Easting": X,
                "Northing": Y,
                "Elevation": Z,
                "Susceptibility": VAL
            })
            df_all.to_csv(os.path.join(html_out, "susceptibility_model_full.csv"), index=False)

        # Clip below DSM for coloring purposes
        if xyz_path is not None:
            surf = LinearNDInterpolator(np.c_[x, y], z, fill_value=np.nan)
            Zsurf = surf(np.c_[X, Y])
            keep = np.isfinite(Zsurf) & (Z < (Zsurf - 0.5)) & (tri.find_simplex(np.c_[X, Y]) >= 0)
        else:
            keep = np.ones_like(VAL, dtype=bool)

        VAL_clip = VAL.copy()
        VAL_clip[~keep] = np.nan

        # Highlight array: set values below threshold to 0 for coloring
        VAL_highlight = np.zeros_like(VAL_clip)
        VAL_highlight[VAL_clip >= ISO_LEVEL_LEFT] = VAL_clip[VAL_clip >= ISO_LEVEL_LEFT]

        vals_ok = VAL_clip[np.isfinite(VAL_clip)]
        if vals_ok.size:
            vmin_full = float(np.percentile(vals_ok, FULL_PERC_RANGE[0]))
            vmax_full = float(np.percentile(vals_ok, FULL_PERC_RANGE[1]))
        iso_max_left = float(np.nanmax(V)) if np.isfinite(V).any() else ISO_LEVEL_LEFT*2

    # -----------------------------
    # 3) Plotting
    # -----------------------------
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("Terrain + Full Susceptibility", f"Terrain + χ ≥ {ISO_LEVEL_LEFT}")
    )

    # Full model surface
    if VAL_clip.size:
        fig.add_trace(go.Isosurface(
            x=X, y=Y, z=Z-50, value=VAL_clip,
            isomin=vmin_full, isomax=vmax_full,
            surface_count=FULL_ISO_SURFACES,
            opacity=FULL_ISO_OPACITY,
            colorscale="Viridis",
            caps=dict(x_show=False, y_show=False, z_show=False),
            showscale=True, colorbar=dict(title="χ (SI)")
        ), row=1, col=1)

    # Highlight surface
    if VAL_clip.size:
        fig.add_trace(go.Isosurface(
            x=X, y=Y, z=Z-50, value=VAL_highlight,
            isomin=ISO_LEVEL_LEFT, isomax=iso_max_left,
            surface_count=ISO_SURFACES_LEFT,
            colorscale=ISO_COLORS_LEFT,
            opacity=ISO_OPACITY_LEFT,
            caps=dict(x_show=False, y_show=False, z_show=False),
            showscale=True, colorbar=dict(title=f"χ ≥ {ISO_LEVEL_LEFT}")
        ), row=1, col=2)

    # Terrain overlay
    if terrain_left is not None:
        fig.add_trace(terrain_left, row=1, col=1)
    if terrain_right is not None:
        fig.add_trace(terrain_right, row=1, col=2)

    fig.update_layout(
        width=1600, height=850,
        title="Side-by-side Terrain + Susceptibility Overlay",
        margin=dict(t=60, l=10, r=10, b=10),
        scene=dict(xaxis_title="Easting", yaxis_title="Northing", zaxis_title="Elevation", aspectmode="data", bgcolor="white"),
        scene2=dict(xaxis_title="Easting", yaxis_title="Northing", zaxis_title="Elevation", aspectmode="data", bgcolor="white")
    )
    cam = dict(eye=dict(x=1.6, y=1.6, z=1.1))
    fig.update_scenes(camera=cam)
    fig.layout.scene2.camera = cam

    if html_out is not None:
        os.makedirs(html_out, exist_ok=True)
        fig.write_html(os.path.join(html_out, title))
        logging.info(f"Saved HTML: {html_out}/{title}")

    fig.show()


# def overlay_terrain_chi(
#     xyz_path: str = None,
#     photo_path: str = None,
#     photo_path2: str = None,
#     html_out: str = None,
#     title: str = "terrain_overlay.html",
#     mesh=None,
#     actv=None,
#     mrec=None,
#     IMG_MAX: int = 768,
#     PT_KEEP: float = 1.0,
#     DSM_BIN: float = 0.0,
#     MODEL_STRIDE: tuple = (1, 1, 1),
#     FLIP_V: bool = True,
#     BORDER_EPS: float = 1e-3,
#     TERRAIN_OPACITY_LEFT: float = 1.0,
#     TERRAIN_OPACITY_RIGHT: float = 1.0,
#     TERRAIN_LIGHT: dict = dict(ambient=0.65, diffuse=0.6, specular=0.05),
#     ISO_LEVEL_LEFT: float = 0.0035,
#     ISO_SURFACES_LEFT: int = 1,
#     ISO_COLORS_LEFT: str = "Hot",
#     ISO_OPACITY_LEFT: float = 1.0,
#     FULL_MODE: str = "isosurface",
#     FULL_ISO_SURFACES: int = 25,
#     FULL_ISO_OPACITY: float = 0.25,
#     FULL_PERC_RANGE: tuple = (5.0, 99.0)
# ):
#     """
#     Generate side-by-side Plotly visualization of terrain (DSM + image) 
#     and 3D magnetic susceptibility. Skips layers that are None.
#     """
    
#     terrain_left = terrain_right = None
#     X = Y = Z = VAL_clip = VAL_left = np.array([])  # placeholders

#     # -----------------------------
#     # 1) Terrain DSM + Orthophoto
#     # -----------------------------
#     if xyz_path is not None and photo_path is not None:
#         logging.info("Loading XYZ terrain and orthophoto...")
#         xyz = np.loadtxt(xyz_path, delimiter=",").astype(np.float32)

#         # Optional DSM thinning
#         if DSM_BIN > 0:
#             xmin, ymin = xyz[:,0].min(), xyz[:,1].min()
#             gx = np.floor((xyz[:,0] - xmin) / DSM_BIN).astype(np.int32)
#             gy = np.floor((xyz[:,1] - ymin) / DSM_BIN).astype(np.int32)
#             keys = (gx.astype(np.int64) << 32) | gy.astype(np.int64)
#             _, keep_idx = np.unique(keys, return_index=True)
#             xyz = xyz[np.sort(keep_idx)]
#         elif 0 < PT_KEEP < 1.0:
#             rng = np.random.default_rng(42)
#             idx = rng.choice(xyz.shape[0], size=int(xyz.shape[0]*PT_KEEP), replace=False)
#             xyz = xyz[np.sort(idx)]

#         x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
#         tri = Delaunay(np.c_[x, y])

#         # Load orthophoto
#         im = Image.open(photo_path).convert("RGB")
#         w, h = im.size
#         scale = min(IMG_MAX / max(w, h), 1.0)
#         if scale < 1.0:
#             im = im.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
#         ARR = np.asarray(im).astype(np.uint8)
#         H, W, _ = ARR.shape

#         # UV mapping
#         xmin, xmax = x.min(), x.max()
#         ymin, ymax = y.min(), y.max()
#         u = (x - xmin) / max(1e-9, (xmax - xmin))
#         v = (y - ymin) / max(1e-9, (ymax - ymin))
#         v_img = 1.0 - v if FLIP_V else v
#         px = np.clip((u * (W-1)).round().astype(np.int32), 0, W-1)
#         py = np.clip((v_img * (H-1)).round().astype(np.int32), 0, H-1)
#         vertex_rgb = ARR[py, px]

#         # Trim triangles outside image
#         uc = (u[tri.simplices.T[0]] + u[tri.simplices.T[1]] + u[tri.simplices.T[2]]) / 3.0
#         vc = (v_img[tri.simplices.T[0]] + v_img[tri.simplices.T[1]] + v_img[tri.simplices.T[2]]) / 3.0
#         inside_tri = (uc >= BORDER_EPS) & (uc <= 1.0 - BORDER_EPS) & (vc >= BORDER_EPS) & (vc <= 1.0 - BORDER_EPS)
#         i2, j2, k2 = tri.simplices.T[0][inside_tri], tri.simplices.T[1][inside_tri], tri.simplices.T[2][inside_tri]

#         terrain_left = go.Mesh3d(
#             x=x, y=y, z=z, i=i2, j=j2, k=k2,
#             vertexcolor=vertex_rgb,
#             opacity=TERRAIN_OPACITY_LEFT,
#             lighting=TERRAIN_LIGHT,
#             flatshading=False,
#             showscale=False,
#             name="DSM + Image"
#         )
#         terrain_right = terrain_left.update(opacity=TERRAIN_OPACITY_RIGHT)

#     # -----------------------------
#     # 2) Magnetic Susceptibility
#     # -----------------------------
#     vmin_full = vmax_full = iso_max_left = ISO_LEVEL_LEFT
#     if mesh is not None and actv is not None and mrec is not None:
#         logging.info("Preparing χ model overlay...")
#         chi_full = np.zeros(mesh.nC, dtype=np.float32)
#         chi_full[actv] = mrec.astype(np.float32)

#         nx, ny, nz = mesh.shape_cells
#         CHI = chi_full.reshape((nx, ny, nz), order="F")
#         ACT = actv.reshape((nx, ny, nz), order="F")

#         ix = np.where(ACT.any(axis=(1,2)))[0]; sx = slice(ix.min(), ix.max()+1, MODEL_STRIDE[0])
#         iy = np.where(ACT.any(axis=(0,2)))[0]; sy = slice(iy.min(), iy.max()+1, MODEL_STRIDE[1])
#         iz = np.where(ACT.any(axis=(0,1)))[0]; sz = slice(iz.min(), iz.max()+1, MODEL_STRIDE[2])
#         V = CHI[sx, sy, sz].copy().astype(np.float32)

#         x0, y0, z0 = mesh.x0
#         hx, hy, hz = mesh.h
#         xe, ye, ze = x0 + np.r_[0.0, np.cumsum(hx)], y0 + np.r_[0.0, np.cumsum(hy)], z0 + np.r_[0.0, np.cumsum(hz)]
#         xc = 0.5*(xe[:-1] + xe[1:])[sx].astype(np.float32)
#         yc = 0.5*(ye[:-1] + ye[1:])[sy].astype(np.float32)
#         zc = 0.5*(ze[:-1] + ze[1:])[sz].astype(np.float32)
#         Xd, Yd, Zd = np.meshgrid(xc, yc, zc, indexing="ij")
#         X, Y, Z = Xd.ravel(order="F"), Yd.ravel(order="F"), Zd.ravel(order="F")
#         VAL = V.ravel(order="F")

#         # ---- Save full model ----
#         df_all = pd.DataFrame({
#             "Easting": X,
#             "Northing": Y,
#             "Elevation": Z,
#             "Susceptibility": VAL
#         })
#         df_all.to_csv(f"{html_out}/susceptibility_model_full.csv", index=False)


#         # Clip below DSM if terrain exists
#         if xyz_path is not None:
#             surf = LinearNDInterpolator(np.c_[x, y], z, fill_value=np.nan)
#             Zsurf = surf(np.c_[X, Y])
#             keep = np.isfinite(Zsurf) & (Z < (Zsurf - 0.5)) & (tri.find_simplex(np.c_[X, Y]) >= 0)
#         else:
#             keep = np.ones_like(VAL, dtype=bool)

#         VAL_clip = np.where(keep, VAL, np.nan)
#         VAL_left = np.where(VAL_clip >= ISO_LEVEL_LEFT, VAL_clip, np.nan)

#         vals_ok = VAL_clip[np.isfinite(VAL_clip)]
#         if vals_ok.size:
#             vmin_full = float(np.percentile(vals_ok, FULL_PERC_RANGE[0]))
#             vmax_full = float(np.percentile(vals_ok, FULL_PERC_RANGE[1]))
#         iso_max_left = float(np.nanmax(V)) if np.isfinite(V).any() else ISO_LEVEL_LEFT*2

#         plot_susceptibility_slices(V, xc, yc, zc, z_index=10, y_index=5, output_dir=html_out)

#     # -----------------------------
#     # 3) Plotting
#     # -----------------------------
#     fig = make_subplots(
#         rows=1, cols=2,
#         specs=[[{'type': 'scene'}, {'type': 'scene'}]],
#         subplot_titles=("Terrain + Full Susceptibility", f"Terrain + χ ≥ {ISO_LEVEL_LEFT}", )
#     )

#     if terrain_right is not None:
#         fig.add_trace(terrain_right, row=1, col=2)
#     if VAL_clip.size:
#         fig.add_trace(go.Isosurface(x=X, y=Y, z=Z-50, value=VAL_clip,
#                                     isomin=vmin_full, isomax=vmax_full,
#                                     surface_count=FULL_ISO_SURFACES, opacity=FULL_ISO_OPACITY,
#                                     colorscale="Viridis",
#                                     caps=dict(x_show=False, y_show=False, z_show=False),
#                                     showscale=True, colorbar=dict(title="χ (SI)")),
#                       row=1, col=1)

#     if terrain_left is not None:
#         fig.add_trace(terrain_left, row=1, col=1)
#     if VAL_left.size:
#         fig.add_trace(go.Isosurface(x=X, y=Y, z=Z-50, value=VAL_left,
#                                     isomin=ISO_LEVEL_LEFT, isomax=iso_max_left,
#                                     surface_count=ISO_SURFACES_LEFT, colorscale=ISO_COLORS_LEFT,
#                                     opacity=ISO_OPACITY_LEFT,
#                                     caps=dict(x_show=False, y_show=False, z_show=False),
#                                     showscale=True, colorbar=dict(title="χ (SI)")),
#                       row=1, col=2)

#     fig.update_layout(
#         width=1600, height=850,
#         title="Side-by-side Terrain + Susceptibility Overlay",
#         margin=dict(t=60, l=10, r=10, b=10),
#         scene=dict(xaxis_title="Easting", yaxis_title="Northing", zaxis_title="Elevation", aspectmode="data", bgcolor="white"),
#         scene2=dict(xaxis_title="Easting", yaxis_title="Northing", zaxis_title="Elevation", aspectmode="data", bgcolor="white")
#     )
#     cam = dict(eye=dict(x=1.6, y=1.6, z=1.1))
#     fig.update_scenes(camera=cam)
#     fig.layout.scene2.camera = cam

#     if html_out is not None:
#         os.makedirs(html_out, exist_ok=True)
#         fig.write_html(os.path.join(html_out, title))
#         logging.info(f"Saved HTML: {html_out}/{title}")

#     fig.show()

def plot_susceptibility_slices(CHI_crop, xc, yc, zc, 
                                z_index=10, y_index=5, 
                                output_dir: str=None):
    """
    Plot horizontal slice, vertical XZ cross-section, and max χ projection.
    
    Parameters
    ----------
    CHI_crop : np.ndarray
        3D susceptibility array (nx, ny, nz)
    xc, yc, zc : np.ndarray
        Cell-center coordinates along x, y, z axes
    z_index : int
        Index along Z for horizontal slice
    y_index : int
        Index along Y for vertical cross-section
    """
    logging.info(f"Plotting horizontal XY slice at z_index={z_index} (elevation ≈ {zc[z_index]:.2f} m)")
    slice_xy = CHI_crop[:, :, z_index]
    plt.figure(figsize=(6,5))
    plt.imshow(slice_xy.T, origin="lower", cmap="viridis",
               extent=[xc.min(), xc.max(), yc.min(), yc.max()])
    plt.colorbar(label="χ (SI)")
    plt.xlabel("Easting")
    plt.ylabel("Northing")
    plt.title(f"Susceptibility at elevation ≈ {zc[z_index]:.2f} m")
    plt.tight_layout()
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"Susceptibility at elevation ≈ {zc[z_index]:.2f} m.png")
        plt.savefig(save_path, dpi=300)
    plt.show()

    logging.info(f"Plotting vertical XZ cross-section at y_index={y_index}")
    slice_xz = CHI_crop[:, y_index, :]
    plt.figure(figsize=(6,5))
    plt.imshow(slice_xz.T, origin="lower", cmap="inferno",
               extent=[xc.min(), xc.max(), zc.min(), zc.max()], aspect="auto")
    plt.colorbar(label="χ (SI)")
    plt.xlabel("Easting")
    plt.ylabel("Elevation")
    plt.title("Vertical XZ Cross-section")
    plt.tight_layout()
    if output_dir is not None:
      save_path = os.path.join(output_dir, "Vertical XZ Cross-section.png")
      plt.savefig(save_path, dpi=300)
    plt.show()

    logging.info("Plotting maximum χ projection (plan view)")
    proj_max = np.nanmax(CHI_crop, axis=2)
    plt.figure(figsize=(6,5))
    plt.imshow(proj_max.T, origin="lower", cmap="magma",
               extent=[xc.min(), xc.max(), yc.min(), yc.max()])
    plt.colorbar(label="Max χ (SI)")
    plt.xlabel("Easting")
    plt.ylabel("Northing")
    plt.title("Maximum χ Projection (plan view)")
    plt.tight_layout()
    if output_dir is not None:
      save_path = os.path.join(output_dir, "Maximum χ Projection (plan view)")
      plt.savefig(save_path, dpi=300)
    plt.show()
