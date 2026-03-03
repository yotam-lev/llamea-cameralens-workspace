import math
import subprocess

import adjustText  # pip install adjustText
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import lensgopt.optics.model as model
import lensgopt.optics.optics as optics
import lensgopt.optics.shapes as shapes


class VisualLensElement:
    def __init__(self, material, catalog, curvature1, curvature2, thickness, aperture):
        self.material = material
        self.catalog = catalog
        self.thickness = thickness
        self.curvature1 = curvature1
        self.curvature2 = curvature2
        self.aperture = aperture


def get_y_star(r, max_h, xc, x, y2):
    is_long = y2 <= 0
    if r > 0 and (x - xc) > 1e-5:
        is_long = True
    if r < 0 and (xc - x) > 1e-5:
        is_long = True
    if not is_long:
        h = np.sqrt(y2)
        if h > max_h:
            y_star = max_h
        else:
            y_star = np.sqrt(y2)
    else:
        if abs(r) > max_h:
            y_star = max_h
        else:
            y_star = abs(r)
    return y_star


def do_negative(r, xc, max_h, x, y2):
    y_star = get_y_star(r, max_h, xc, x, y2)
    alpha_star = np.arcsin(y_star / abs(r))
    theta = np.linspace(-alpha_star, alpha_star, 100)
    x_ = xc + abs(r) * np.cos(theta)
    y_ = abs(r) * np.sin(theta)
    return x_, y_


def do_positive(r, xc, max_h, x, y2):
    y_star = get_y_star(r, max_h, xc, x, y2)
    alpha_star = np.pi - np.arcsin(y_star / abs(r))
    theta = np.linspace(alpha_star, 2 * np.pi - alpha_star, 100)
    x_ = xc + abs(r) * np.cos(theta)
    y_ = abs(r) * np.sin(theta)
    return x_, y_


def __plot_lenses(
    lenses,
    dists,
    max_H=100,
    is_text=True,
    is_fill=True,
    max_system_length=None,
    z_sensor=None,
    is_colorbar=True,
    width_hieght_cm=None,
    is_axis_names=False,
    material_to_name=None,
):
    if width_hieght_cm is not None:
        w_cm, h_cm = width_hieght_cm
        fig, ax = plt.subplots(figsize=(w_cm * 0.3937, h_cm * 0.3937))
    else:
        fig, ax = plt.subplots(figsize=(18.5, 10.5))
    current_position = 0
    num_lens = 0
    LW = 2

    texts = []
    texts_targets_x = []
    texts_targets_y = []
    for lens, dist in zip(lenses, dists):
        zorder = 2 * num_lens + 10
        # Calculate radii of curvatures
        R1 = 1 / lens.curvature1 if lens.curvature1 != 0 else 1e5
        R2 = 1 / lens.curvature2 if lens.curvature2 != 0 else 1e5

        x1 = 0
        x2 = lens.thickness - R1 + R2
        x = (R1**2 - R2**2 + x2**2) / (2 * x2)
        y2 = R2**2 - (x - x2) ** 2
        sh = current_position + R1

        num_material = len(lens.catalog)
        mycmap = plt.get_cmap("jet", num_material)
        # mycmap = colorblind_safe_cmap.mpl_colormap
        to_color = lambda m: mycmap(m / num_material)
        color = to_color(lens.material)
        max_h = min(max_H, lens.aperture)

        if R1 < 0:
            x1_, y1_ = do_negative(R1, x1, max_h, x, y2)
        else:
            x1_, y1_ = do_positive(R1, x1, max_h, x, y2)
            if is_fill:
                x1_ = x1_[::-1]
                y1_ = y1_[::-1]
        if is_fill:
            x1_cl_top = sh + x1_[y1_ > 0]
            y1_cl_top = y1_[y1_ > 0]
            x1_cl_bot = sh + x1_[y1_ <= 0]
            y1_cl_bot = y1_[y1_ <= 0]
        x1_star = sh + x1_[0]
        y1_star = abs(y1_[0])
        ax.plot(sh + x1_, y1_, c=color, linewidth=LW, zorder=zorder)
        if R2 < 0:
            x2_, y2_ = do_negative(R2, x2, max_h, x, y2)
            if is_fill:
                x2_ = x2_[::-1]
                y2_ = y2_[::-1]
        else:
            x2_, y2_ = do_positive(R2, x2, max_h, x, y2)
        if is_fill:
            x2_cl_top = sh + x2_[y2_ > 0]
            y2_cl_top = y2_[y2_ > 0]
            x2_cl_bot = sh + x2_[y2_ <= 0]
            y2_cl_bot = y2_[y2_ <= 0]
        x2_star = sh + x2_[0]
        y2_star = abs(y2_[0])
        ax.plot(sh + x2_, y2_, c=color, linewidth=LW, zorder=zorder)

        empty = lambda: np.array([])
        (
            x_cl_top,
            y_cl_top,
            x_cl_bot,
            y_cl_bot,
            xl_cl_top,
            yl_cl_top,
            xl_cl_bot,
            yl_cl_bot,
            xr_cl_top,
            yr_cl_top,
            xr_cl_bot,
            yr_cl_bot,
        ) = (
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
        )
        if np.abs(x1_star - x2_star) > 1e-5:  # need hor. lines above and below
            _n_samples = 10
            ax.plot(
                [x1_star, x2_star], [max_h, max_h], c=color, linewidth=LW, zorder=zorder
            )
            ax.plot(
                [x1_star, x2_star],
                [-max_h, -max_h],
                c=color,
                linewidth=LW,
                zorder=zorder,
            )
            top_y = max_h
            if np.abs(y1_star - max_h) > 1e-5:  # need vert. lines on the left
                ax.plot(
                    [x1_star, x1_star],
                    [y1_star, max_h],
                    c=color,
                    linewidth=LW,
                    zorder=zorder,
                )
                ax.plot(
                    [x1_star, x1_star],
                    [-y1_star, -max_h],
                    c=color,
                    linewidth=LW,
                    zorder=zorder,
                )
                if is_fill:
                    xl_cl_top = np.ones(_n_samples) * x1_star
                    yl_cl_top = np.linspace(y1_star, max_h, _n_samples)
                    xl_cl_bot = np.ones(_n_samples) * x1_star
                    yl_cl_bot = np.linspace(-max_h, -y1_star, _n_samples)
            if np.abs(y2_star - max_h) > 1e-5:  # need vert. lines on the right
                ax.plot(
                    [x2_star, x2_star],
                    [y2_star, max_h],
                    c=color,
                    linewidth=LW,
                    zorder=zorder,
                )
                ax.plot(
                    [x2_star, x2_star],
                    [-y2_star, -max_h],
                    c=color,
                    linewidth=LW,
                    zorder=zorder,
                )
                if is_fill:
                    xr_cl_top = np.ones(_n_samples) * x2_star
                    yr_cl_top = np.linspace(y2_star, max_h, _n_samples)[::-1]
                    xr_cl_bot = np.ones(_n_samples) * x2_star
                    yr_cl_bot = np.linspace(-max_h, -y2_star, _n_samples)[::-1]
            if is_fill:
                x_cl_top = np.linspace(x1_star, x2_star, _n_samples)
                y_cl_top = np.ones(_n_samples) * max_h
                x_cl_bot = np.linspace(x1_star, x2_star, _n_samples)[::-1]
                y_cl_bot = -np.ones(_n_samples) * max_h
        else:
            top_y = y1_star
        if is_fill:
            clockwise_x_borders = np.concatenate(
                [
                    x1_cl_top,
                    xl_cl_top,
                    x_cl_top,
                    xr_cl_top,
                    x2_cl_top,
                    x2_cl_bot,
                    xr_cl_bot,
                    x_cl_bot,
                    xl_cl_bot,
                    x1_cl_bot,
                ]
            )
            clockwise_y_borders = np.concatenate(
                [
                    y1_cl_top,
                    yl_cl_top,
                    y_cl_top,
                    yr_cl_top,
                    y2_cl_top,
                    y2_cl_bot,
                    yr_cl_bot,
                    y_cl_bot,
                    yl_cl_bot,
                    y1_cl_bot,
                ]
            )
            ax.fill(
                clockwise_x_borders,
                clockwise_y_borders,
                color=color,
                alpha=0.5,
                zorder=zorder,
            )
        mid_x = (x1_star + x2_star) / 2
        if is_text:
            if not material_to_name is None:
                material_name_str = material_to_name(lens.material)
            else:
                material_name_str = f"${lens.material}$"
            ymin, ymax = ax.get_ylim()
            texts.append(
                ax.text(
                    mid_x,
                    -top_y,
                    material_name_str,
                    ha="center",
                    va="bottom",
                    zorder=10**5,
                )
            )
            texts_targets_x.append(mid_x)
            texts_targets_y.append(-top_y)
        current_position += lens.thickness + dist
        num_lens += 1

    if z_sensor is not None:
        ax.axvline([z_sensor], c="k")
    elif max_system_length is not None:
        xax_min, xax_max = ax.get_xlim()
        yax_min, yax_max = ax.get_ylim()
        ax.set_xlim((xax_min, max_system_length))
        # ax.set_ylim((-9, 9))

    if is_colorbar:
        cmin, cmax = 0, num_material
        sm = plt.cm.ScalarMappable(cmap=mycmap, norm=plt.Normalize(cmin, cmax))
        toi = lambda a: [round(ai) for ai in a]
        fig.colorbar(sm, ax=ax, ticks=toi(np.linspace(cmin, cmax - 1, 6)), shrink=0.7)
    if is_text:
        adjustText.adjust_text(
            texts,
            target_x=texts_targets_x,
            target_y=texts_targets_y,
            arrowprops=dict(arrowstyle="-", color="black"),
        )

    # ax.set_aspect('equal', adjustable='box')
    if is_axis_names:
        ax.set_xlabel("Position (mm)")
        ax.set_ylabel("Height (mm)")
    # ax.set_title('Lens System Visualization')

    return fig, ax


def sample_rays_2D(
    n_rays_row: int,
    n_wl: int,
    r_ep: float,
    z_ep: float,
    angles_z_deg: jnp.ndarray,
) -> optics.RayRMSSpotSize:
    """
    Generate a grid of rays for multiple tilt angles and wavelengths, distributed only along the Y-axis.

    Args:
        n_rays_row   : number of Y-samples per angle (int-like).
        n_wl         : number of wavelengths.
        r_ep         : entrance pupil radius (float-like).
        z_ep         : entrance pupil z-coordinate (float-like).
        angles_z_deg : 1D array of tilt angles in degrees (shape: (n_ang,)).

    Returns:
        RayRMSSpotSize:
            o    Origins array, shape (n_ang, n_wl, n_rays, 3)
            d    Directions array, shape (n_ang, n_wl, n_rays, 3)
            mask Boolean mask of shape (n_ang, n_wl, n_rays), all True inside the EP
    """
    ys = jnp.linspace(-r_ep, r_ep, n_rays_row)  # (n,)
    n = ys.shape[0]

    # Tilt information
    n_ang = angles_z_deg.shape[0]
    chief_ray_y = optics.aim_ray_y(z_ep + 5.0, angles_z_deg)  # (n_ang,)
    cy = chief_ray_y[:, None]  # (n_ang, 1)

    # Compute origins: shape (n_ang, n, 3)
    xx = jnp.zeros((n_ang, n))  # (n_ang, n)
    yy = ys[None, :] + cy  # (n_ang, n)
    zz = jnp.ones((n_ang, n)) * -5.0  # (n_ang, n)
    origins = jnp.stack([xx, yy, zz], axis=-1)  # (n_ang, n, 3)

    # Broadcast to wavelengths
    o_full = jnp.broadcast_to(
        origins[:, None, :, :], (n_ang, n_wl, n, 3)
    )  # (n_ang, n_wl, n, 3)

    # Direction vectors
    rad = angles_z_deg * (jnp.pi / 180.0)  # (n_ang,)
    sinz = jnp.sin(rad)
    cosz = jnp.cos(rad)
    dir_vecs = jnp.stack([jnp.zeros_like(rad), sinz, cosz], axis=1)  # (n_ang, 3)
    d_full = jnp.broadcast_to(
        dir_vecs[:, None, None, :], (n_ang, n_wl, n, 3)
    )  # (n_ang, n_wl, n, 3)
    mask = jnp.ones((n_ang, n_wl, n), dtype=bool)

    return optics.RayRMSSpotSize(o=o_full, d=d_full, mask=mask)


def create_rays_2D(
    cnst: model.LensSystemConstants,
    props: model.LensSystemComputedProperties,
    n_rays_y: int,
    n_wl: int,
    field_factors: tuple[float, ...],
):
    match cnst.lens_field.type:
        case "angle":
            if cnst.object_z != -jnp.inf:
                raise NotImplementedError(
                    "Finite size of object support is not implemented yet"
                )
            else:
                if cnst.aperture.type != "ED":
                    raise NotImplementedError(
                        f"Sampling rays for non ED aperture type is not implemented yet"
                    )
                r_ep = cnst.aperture.max_d / 2
                angles_z_deg = cnst.lens_field.max_field * jnp.array(field_factors)
                return sample_rays_2D(
                    n_rays_row=n_rays_y,
                    n_wl=n_wl,
                    r_ep=r_ep,
                    z_ep=props.entrance_pupil_z,
                    angles_z_deg=angles_z_deg,
                )
        case "objectHeight":
            raise NotImplementedError(f"Field type objectHeight is not implemented yet")
        case _:
            raise ValueError(f"Field type {cnst.lens_field.type} not implemented yet")


def record_rays(
    cnst: model.LensSystemConstants,
    vars: model.LensSystemVariables,
    props: model.LensSystemComputedProperties,
    n_rays_y,
    field_factors=None,
):
    if field_factors is None:
        field_factors = [0, 0.7, 1]

    rays = create_rays_2D(
        cnst=cnst, props=props, n_rays_y=n_rays_y, n_wl=1, field_factors=field_factors
    )
    # rays = optics.RayRMSSpotSize.create_edge_thickness_rays(cnst.edge_thickness_field_factors,
    #                                                         cnst.edge_thickness_entrance_pupil_factors,
    #                                                         props.entrance_pupil_z,
    #                                                         props.entrance_pupil_diameter,
    #                                                         cnst.lens_field )
    ys, zs, valids = [], [], []

    def on_intersect_callback(surf_num, rays, valid_intersection, o):
        ys.append(np.array(o[..., 1]))
        zs.append(np.array(o[..., 2]))
        valids.append(np.array(rays.mask & valid_intersection))

    def on_refraction_callback(surf_num, rays, o1, valid_refraction, d1):
        if surf_num == 9:
            surface_class = shapes.get_surface_class_by_name(
                cnst.surface_types[surf_num]
            )
            surface = surface_class.create(
                props.vertices_z[surf_num],
                vars.curvature_parameters[surf_num : surf_num + 1],
            )
            normal = surface.normal(o1[..., 0], o1[..., 1], o1[..., 2])
            # print(vars.iors[surf_num + 1])
            # print(normal)
            cos_alpha = jnp.sum(rays.d * normal, axis=-1)
            sin_alpha = jnp.sqrt(1 - jnp.square(cos_alpha))

            cos_beta = jnp.sum(d1 * normal, axis=-1)
            sin_beta = jnp.sqrt(1 - cos_beta**2)

            # print("sin_alpha:", sin_alpha)
            # print("sin_beta*n2", sin_beta * vars.iors[surf_num + 1][0])
            print("ior", vars.iors[surf_num + 1][0])
            # assert jnp.isclose(sin_alpha, sin_beta * vars.iors[surf_num + 1][0], atol=0.0).all()
            assert np.isclose(
                sin_alpha, sin_beta * vars.iors[surf_num + 1][0], atol=1e-12
            ).all()

    sensor_plane = shapes.Spheric(props.sensor_z, jnp.inf)

    on_intersect_callback(None, rays, rays.mask, rays.o)

    optics.real_trace_forward_with_subscribers(
        flat_params=vars.curvature_parameters,
        vertices=props.vertices_z,
        cnt_per_surface=cnst.num_parameters_per_surface,
        types=cnst.surface_types,
        iors=vars.iors[..., 0][:, None],
        limiting_vertices_z=props.limiting_vertices_z,
        up_corners_y=props.up_corners_y,
        rays=rays,
        sensor=sensor_plane,
        on_intersection=on_intersect_callback,
        on_refraction=lambda surf_num, rays, o1, valid_refraction, d1: None,
    )
    return ys, zs, valids


def find_apertures(ys_all, zs_all, valids_all, num_surfaces):
    aps = []
    n_ang = len(ys_all[0])
    prv_ap = 10
    n_all_surf = num_surfaces + 2
    for i in range(n_all_surf):
        ap = 0
        for j in range(n_ang):
            ys, zs, valids = ys_all[i][j][0], zs_all[i][j][0], valids_all[i][j][0]
            valid_rays_on_i = ys[valids]
            if len(valid_rays_on_i) == 0:
                ap = max(ap, prv_ap)
            else:
                ap = max(ap, max(max(valid_rays_on_i), abs(min(valid_rays_on_i))))
        aps.append(ap)
        prv_ap = ap
    return aps


def add_aperture_stop_to_plot(
    fig, ax, aperture_stop_z, aperture_stop_aperture, lens_elements
):
    fig_w, fig_h = fig.get_size_inches()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    fig_w = xmax - xmin
    fig_h = ymax - ymin
    z = aperture_stop_z
    aperture = aperture_stop_aperture
    L = 0.02 * fig_w
    H = 0.08 * fig_h
    configs = {"zorder": 2 * len(lens_elements) + 20, "color": "black", "linewidth": 2}
    ax.plot([z, z], [H + aperture, aperture], **configs)
    ax.plot(
        [z - L / 2, z + L / 2],
        [aperture, aperture],
        **configs,
    )
    ax.plot([z, z], [-H - aperture, -aperture], **configs)
    ax.plot(
        [z - L / 2, z + L / 2],
        [-aperture, -aperture],
        **configs,
    )
    return fig, ax


def plot_lenses(
    cnst: model.LensSystemConstants,
    vars: model.LensSystemVariables,
    material_ids: jnp.ndarray,
    sensor_z: float = None,
    explicit_aperture=None,
    is_add_aperture_stop=True,
    num_rays=5,
    **kwargs,
):
    lens_elements, distances = lens_to_vis_format(
        cnst,
        vars,
        material_ids,
        sensor_z=sensor_z,
        explicit_aperture=explicit_aperture,
        rays_number=num_rays,
    )
    z_sensor = None
    if not sensor_z is None:
        z_sensor = sensor_z
    fig, ax = __plot_lenses(
        lens_elements,
        distances,
        z_sensor=z_sensor,
        **kwargs,
    )
    if is_add_aperture_stop:
        if not explicit_aperture is None:
            props: model.LensSystemComputedProperties = (
                optics.compute_optical_properties(cnst, vars)
            )
            ys_all, zs_all, valids_all = record_rays(cnst, vars, props, num_rays)
            aps = find_apertures(ys_all, zs_all, valids_all, cnst.num_surfaces)
            aperture_stop_z = props.vertices_z[cnst.aperture_stop_index]
            aperture_stop_aperture = aps[cnst.aperture_stop_index + 1]
        else:
            vertices_z = model.LensSystemVariables.resolve_verticies_z(vars.distances_z)
            aperture_stop_z = vertices_z[cnst.aperture_stop_index]
            aperture_stop_aperture = explicit_aperture
        fig, ax = add_aperture_stop_to_plot(
            fig, ax, aperture_stop_z, aperture_stop_aperture, lens_elements
        )
    return fig, ax


def plot_lenses_with_rays(
    cnst: model.LensSystemConstants,
    vars: model.LensSystemVariables,
    material_ids: jnp.ndarray,
    sensor_z: float,
    num_rays: int,
    field_factors: tuple[float, ...] = None,
    colors: str = "rgb",
    is_add_aperture_stop: bool = True,
    material_to_name=None,
    **kwargs,
):
    if num_rays < 0:
        raise ValueError(
            f"num_rays should be positive integer, but value {num_rays} is passed"
        )
    lens_elements, distances = lens_to_vis_format(
        cnst,
        vars,
        material_ids,
        sensor_z=sensor_z,
        explicit_aperture=None,
        rays_number=num_rays,
    )
    fig, ax = __plot_lenses(
        lens_elements,
        distances,
        z_sensor=sensor_z,
        material_to_name=material_to_name,
        **kwargs,
    )
    props: model.LensSystemComputedProperties = optics.compute_optical_properties(
        cnst, vars
    )
    ys_all, zs_all, valids_all = record_rays(
        cnst, vars, props, num_rays, field_factors=field_factors
    )
    for i in range(len(field_factors)):
        zorder = 2 * len(lens_elements) + 11 + i
        color = colors[i]
        for it1 in range(len(ys_all) - 1):
            it2 = it1 + 1
            y1, x1, valid1 = ys_all[it1][i][0], zs_all[it1][i][0], valids_all[it1][i][0]
            y2, x2, valid2 = ys_all[it2][i][0], zs_all[it2][i][0], valids_all[it2][i][0]
            for j in range(len(y1)):
                if valid1[j] and valid2[j]:
                    ax.plot([x1[j], x2[j]], [y1[j], y2[j]], c=color, zorder=zorder)
                elif valid1[j] and not valid2[j]:
                    continue
                elif not valid1[j] and valid2[j]:
                    raise RuntimeError("Invalid ray became valid")
                elif not valid1[j] and not valid2[j]:
                    continue

    if is_add_aperture_stop:
        aps = find_apertures(ys_all, zs_all, valids_all, cnst.num_surfaces)
        aperture_stop_z = props.vertices_z[cnst.aperture_stop_index]
        aperture_stop_aperture = aps[cnst.aperture_stop_index + 1]
        fig, ax = add_aperture_stop_to_plot(
            fig, ax, aperture_stop_z, aperture_stop_aperture, lens_elements
        )
    return fig, ax


def is_glass(ior_d: float):
    is_vacuum = math.isclose(ior_d, 1.0, abs_tol=1e-6)
    is_air = math.isclose(ior_d, 1.000293, abs_tol=1e-6)
    return not is_vacuum and not is_air


def is_separate_aperture_stop(
    cnst: model.LensSystemConstants,
    vars: model.LensSystemVariables,
):
    before_stop_ior_d = vars.iors[cnst.aperture_stop_index][0]
    after_stop_ior_d = vars.iors[cnst.aperture_stop_index + 1][0]
    return not is_glass(before_stop_ior_d) and not is_glass(after_stop_ior_d)


def lens_to_vis_format(
    cnst: model.LensSystemConstants,
    vars: model.LensSystemVariables,
    material_ids: jnp.ndarray,
    sensor_z: float,
    explicit_aperture=None,
    rays_number=0,
):
    if explicit_aperture is None:
        props: model.LensSystemComputedProperties = optics.compute_optical_properties(
            cnst, vars
        )
        ys, zs, valids = record_rays(cnst, vars, props, max(rays_number, 2))
        aps = find_apertures(ys, zs, valids, cnst.num_surfaces)
    else:
        aps = [explicit_aperture] * (cnst.num_surfaces + 2)
    aps.pop(0)
    rs = optics.get_paraxial_rs(
        vars.curvature_parameters,
        cnst.num_parameters_per_surface,
        cnst.surface_types,
    )
    separate_aperture_stop = is_separate_aperture_stop(cnst, vars)
    lens_elements = []
    distances = []
    # Create sequence of VisLensElements
    for it1 in range(cnst.num_surfaces - 1):
        it2 = it1 + 1
        ior_d = vars.iors[it2][0]
        if not is_glass(ior_d):
            # These two surfaces do not make up a lens element
            if separate_aperture_stop and it1 == cnst.aperture_stop_index:
                if len(distances) > 0:
                    distances[-1] += vars.distances_z[it1]
            else:
                distances.append(vars.distances_z[it1])
            continue
        if is_glass(vars.iors[it1][0]) and is_glass(vars.iors[it2][0]):
            # Detach glued lens surfaces into separate optical elements
            distances.append(0)
        element = VisualLensElement(
            material_ids[it2],
            cnst.ior_catalogs[it2],
            1 / rs[it1],
            1 / rs[it2],
            vars.distances_z[it1],
            max(aps[it1], aps[it2]),
        )
        lens_elements.append(element)
    if not sensor_z is None:
        imp_z = sensor_z
        distances.append(imp_z - sum(vars.distances_z))
    return lens_elements, distances
