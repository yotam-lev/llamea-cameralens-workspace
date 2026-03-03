import itertools
from functools import partial
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp

import lensgopt.optics.model as model
import lensgopt.optics.shapes as shapes


def normalize(v):
    """L2-normalize last axis, guarding against zero-length."""
    nrm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / jnp.where(nrm == 0, 1.0, nrm)


def aim_ray_y(distance_z: float, angle_z_deg: jnp.array) -> jnp.array:
    """
    Compute the y-offset for the chief ray such that when tilted by angle_z_deg,
    it intersects the optical axis at given distance in z.

    Args:
        distance:    Distance along z from intersection to reference plane.
        angle_z_deg: Tilt angle in degrees between ray and z-axis.

    Returns:
        float: Y-coordinate offset for the grid center.
    """
    rad = angle_z_deg * (jnp.pi / 180.0)
    # y = -z * tanθ = -distance * sinθ / cosθ
    return -distance_z * jnp.tan(rad)


class RayRMSSpotSize(NamedTuple):
    """
    Ray bundle for multi-angle, multi-wavelength grids. This is developed to compute RMS spot size
    """

    o: jnp.ndarray  # (..., 3)
    d: jnp.ndarray  # (..., 3), unit vectors
    mask: jnp.ndarray  # (...), bool, True = ray still valid

    @staticmethod
    def create_grid_rays_angle_field_for_rms_spot(
        n_rays_row: int,
        n_wl: int,
        r_ep: float,
        z_ep: float,
        angles_z_deg: jnp.ndarray,
    ) -> "RayRMSSpotSize":
        """
        Generate a uniform square grid of rays for multiple tilt angles and wavelengths.

        Args:
            n_rays_row   : number of rays per side of the square grid (int-like).
            n_wl         : number of wavelengths.
            r_ep         : entrance pupil radius (float-like).
            z_ep         : entrance pupil z-coordinate (float-like).
            angles_z_deg : 1D array of tilt angles (between the ray and z-axis) in degrees, shape (n_ang,).

        Returns:
            Ray: NamedTuple with:
                o    Origins array, shape (n_ang, n_wl, n*n, 3).
                d    Directions array, shape (n_ang, n_wl, n*n, 3), unit vectors.
                mask Boolean mask of shape (n_ang, n_wl, n*n), all True.
        """
        # Number of grid points per side
        n = n_rays_row
        # jax.debug.print('r_ep: {x:.13f}\n', x=r_ep, ordered=True)
        xs = jnp.linspace(-r_ep, r_ep, n)
        ys_base = jnp.linspace(-r_ep, r_ep, n)

        # Build meshgrid independent of angle: (n, n)
        xx, yy = jnp.meshgrid(xs, ys_base, indexing="xy")
        zz = jnp.zeros_like(xx)  # (n, n)

        # Number of angles
        n_ang = angles_z_deg.shape[0]

        chief_ray_y = aim_ray_y(z_ep, angles_z_deg)

        # Expand grid to per-angle by adding chief_ray_y along axis 0
        # chief_ray_y: (n_ang,)
        cy = chief_ray_y[:, None, None]  # (n_ang, 1, 1)
        xx_a = jnp.broadcast_to(xx[None, ...], (n_ang, n, n))
        yy_a = jnp.broadcast_to(yy[None, ...], (n_ang, n, n)) + cy
        zz_a = jnp.broadcast_to(zz[None, ...], (n_ang, n, n))
        grid_o = jnp.stack([xx_a, yy_a, zz_a], axis=-1)  # (n_ang, n, n, 3)

        # Broadcast origins to (n_ang, n_wl, n, n, 3)
        o_full = jnp.broadcast_to(grid_o[:, None, ...], (n_ang, n_wl, n, n, 3))

        # Compute direction vectors for each angle
        rad = angles_z_deg * (jnp.pi / 180.0)  # (n_ang,)
        sinz = jnp.sin(rad)  # (n_ang,)
        cosz = jnp.cos(rad)  # (n_ang,)
        dir_vecs = jnp.stack([jnp.zeros_like(rad), sinz, cosz], axis=1)  # (n_ang, 3)

        # Broadcast directions to (n_ang, n_wl, n, n, 3)
        d_full = jnp.broadcast_to(
            dir_vecs[:, None, None, None, :], (n_ang, n_wl, n, n, 3)
        )

        # Flatten the last two spatial dimensions (n, n) → (n*n)
        n_rays = n * n
        o_flat = o_full.reshape((n_ang, n_wl, n_rays, 3))
        d_flat = d_full.reshape((n_ang, n_wl, n_rays, 3))

        cy_filter = jnp.broadcast_to(cy, (n_ang, n_wl, n_rays))
        inside_EP = (
            o_flat[..., 0] ** 2 + (o_flat[..., 1] - cy_filter) ** 2 <= r_ep**2 + 1e-10
        )

        return RayRMSSpotSize(o=o_flat, d=d_flat, mask=inside_EP)

    @staticmethod
    def create_custom_rays(
        n_wl: int,
        angles_z_deg: jnp.ndarray,  # (n_rays,)
        origins: jnp.ndarray,  # (n_rays, 3)
    ) -> "RayRMSSpotSize":
        # number of rays and wavelengths
        n_rays = angles_z_deg.shape[0]

        # broadcast origins: (n_rays, 3) -> (n_rays, n_wl, 1, 3)
        o = jnp.broadcast_to(origins[:, None, None, :], (n_rays, n_wl, 1, 3))

        # compute per-ray direction vectors
        rad = angles_z_deg * (jnp.pi / 180.0)  # (n_rays,)
        sinz = jnp.sin(rad)  # (n_rays,)
        cosz = jnp.cos(rad)  # (n_rays,)
        dir_vec = jnp.stack([jnp.zeros_like(rad), sinz, cosz], axis=1)  # (n_rays, 3)

        # broadcast directions: (n_rays, 3) -> (n_rays, n_wl, 1, 3)
        d = jnp.broadcast_to(dir_vec[:, None, None, :], (n_rays, n_wl, 1, 3))

        # initial mask: all valid
        mask = jnp.ones((n_rays, n_wl, 1), dtype=bool)

        return RayRMSSpotSize(o=o, d=d, mask=mask)

    @staticmethod
    def create_chief_rays(
        n_wl: int, z_ep: float, angles_z_deg: jnp.ndarray
    ) -> "RayRMSSpotSize":
        """
        Create chief-ray origins and directions for each angle and wavelength,
        with shape matching the “ray” axis of a full ray bundle.

        Args:
            angles_z_deg : 1D array of tilt angles (degrees), shape (n_ang,).
            chief_ray_y  : 1D array of Y-offsets for each angle, shape (n_ang,).
            n_wl         : number of wavelengths.

        Returns:
            o_chief: array of chief origins, shape (n_ang, n_wl, 1, 3).
            d_chief: array of chief directions, shape (n_ang, n_wl, 1, 3).

        Explanation:
            - For each angle_index i:
                origin = [0, chief_ray_y[i], 0]
                direction = [0, sin(angle_i), cos(angle_i)]
        """
        # Number of angles
        n_ang = angles_z_deg.shape[0]

        # Convert angles to radians
        rad = angles_z_deg * (jnp.pi / 180.0)  # (n_ang,)
        sinz = jnp.sin(rad)  # (n_ang,)
        cosz = jnp.cos(rad)  # (n_ang,)

        chief_ray_y = aim_ray_y(z_ep, angles_z_deg)

        # Build (n_ang, 3) arrays for origin and direction of chief ray
        chief_o_per_angle = jnp.stack(
            [
                jnp.zeros_like(chief_ray_y),  # x = 0
                chief_ray_y,  # y = chief_ray_y[i]
                jnp.zeros_like(chief_ray_y),  # z = 0
            ],
            axis=1,
        )  # shape = (n_ang, 3)

        chief_d_per_angle = jnp.stack(
            [
                jnp.zeros_like(rad),  # x = 0
                sinz,  # y = sin(angle_i)
                cosz,  # z = cos(angle_i)
            ],
            axis=1,
        )  # shape = (n_ang, 3)

        # Expand to (n_ang, n_wl, 3)
        chief_o_expand = chief_o_per_angle[:, None, :]  # (n_ang, 1, 3)
        chief_o_expand = jnp.broadcast_to(chief_o_expand, (n_ang, n_wl, 3))

        chief_d_expand = chief_d_per_angle[:, None, :]  # (n_ang, 1, 3)
        chief_d_expand = jnp.broadcast_to(chief_d_expand, (n_ang, n_wl, 3))

        # Insert a “ray” axis of length 1 → shape (n_ang, n_wl, 1, 3)
        o_chief = chief_o_expand[..., None, :]  # (n_ang, n_wl, 1, 3)
        d_chief = chief_d_expand[..., None, :]  # (n_ang, n_wl, 1, 3)

        return RayRMSSpotSize(
            o=o_chief, d=d_chief, mask=jnp.ones((n_ang, n_wl, 1), dtype=bool)
        )

    @staticmethod
    def append_ray(
        rays: "RayRMSSpotSize", other_ray: "RayRMSSpotSize"
    ) -> "RayRMSSpotSize":
        """
        Concatenate a chief-ray (of shape (n_ang, n_wl, 1, 3)) onto an existing bundle.
        """
        # Concatenate along the “ray” axis (axis=2)
        o_new = jnp.concatenate(
            [rays.o, other_ray.o], axis=2
        )  # → (n_ang, n_wl, n_r+1, 3)
        d_new = jnp.concatenate(
            [rays.d, other_ray.d], axis=2
        )  # → (n_ang, n_wl, n_r+1, 3)
        mask_new = jnp.concatenate(
            [rays.mask, other_ray.mask], axis=2
        )  # → (n_ang, n_wl, n_r+1)

        return RayRMSSpotSize(o=o_new, d=d_new, mask=mask_new)

    @staticmethod
    def create_edge_thickness_rays(
        factor_angle: tuple[float, ...],
        factor_dEP: tuple[float, ...],
        entrance_pupil_dist: float,
        entrance_pupil_diam: float,
        lens_field: model.Field,
    ):
        """
        Generates 4D JAX arrays of ray origins and directions for edge thickness evaluation.

        Returns:
            origins: jnp.ndarray of shape (1, 1, N, 3)
            directions: jnp.ndarray of shape (1, 1, N, 3)
        """
        # Total number of rays: len(factor_angle) * len(factor_dEP)
        num_rays = len(factor_angle) * len(factor_dEP)

        origins = jnp.zeros((num_rays, 3))
        directions = jnp.zeros((num_rays, 3))

        idx = 0
        for fa, fd in itertools.product(factor_angle, factor_dEP):
            theta_rad = fa * lens_field.max_field * jnp.pi / 180.0
            d = fd * entrance_pupil_diam / 2
            origin = jnp.array([0.0, d - entrance_pupil_dist * jnp.tan(theta_rad), 0.0])
            direction = jnp.array([0.0, jnp.sin(theta_rad), jnp.cos(theta_rad)])
            origins = origins.at[idx].set(origin)
            directions = directions.at[idx].set(direction)
            idx += 1

        # Add two leading singleton dimensions for 4D output
        origins = origins[None, None, :, :]  # shape (1, 1, N, 3)
        directions = directions[None, None, :, :]  # shape (1, 1, N, 3)
        mask = jnp.ones((1, 1, num_rays), dtype=bool)
        return RayRMSSpotSize(o=origins, d=directions, mask=mask)

    def compute_rms2_spot_with_appended_chief_centroid(self) -> jnp.ndarray:
        """
        For each (angle, wavelength) combination, compute the RMS^2 spot radius
        using the last ray (chief ray) in the array as the center, excluding
        the chief ray from the variance count. When count == 0, we force RMS = 0.
        Finally, sum all RMS values over angles and wavelengths.

        Returns:
            total_rms: scalar sum of RMS spot sizes over all angles and wavelengths.
        """
        o = self.o  # (n_ang, n_wl, n_r, 3)
        mask = self.mask  # (n_ang, n_wl, n_r)

        # Extract x and y coordinates of all intersections
        x = o[..., 0]  # (n_ang, n_wl, n_r)
        y = o[..., 1]  # (n_ang, n_wl, n_r)

        # Use the last ray (chief) as the center for each (angle, wavelength)
        x_center = x[..., -1]  # (n_ang, n_wl)
        y_center = y[..., -1]  # (n_ang, n_wl)

        # Compute differences from the chief center
        dx = x - x_center[..., None]  # (n_ang, n_wl, n_r)
        dy = y - y_center[..., None]  # (n_ang, n_wl, n_r)

        # Count how many valid rays (including chief) for each (angle, wavelength)
        count = jnp.sum(mask, axis=-1)  # (n_ang, n_wl)

        # Exclude the chief ray from count
        count = count - 1  # (n_ang, n_wl)

        # Compute squared radius for each ray, zeroing out invalid ones
        rsq = (dx**2 + dy**2) * mask  # (n_ang, n_wl, n_r)

        # Sum squared radii over rays
        sum_rsq = jnp.sum(rsq, axis=-1)  # (n_ang, n_wl)

        # Compute mean square radius, but avoid division by zero:
        # if count > 0, do sum_rsq/count; otherwise 0.0
        count_safe = jnp.where(count > 0, count, 1.0)
        mean_rsq = jnp.where(count > 0, sum_rsq / count_safe, 0.0)  # (n_ang, n_wl)
        # jax.debug.print('{}\n', jnp.sqrt(mean_rsq), ordered=True)

        # Take square root
        # rms = jnp.sqrt(mean_rsq)  # (n_ang, n_wl)

        # Mean RMS^2 values across angles and wavelengths
        total_rms = jnp.sum(mean_rsq) / mean_rsq.shape[0] / mean_rsq.shape[1]  # scalar

        return total_rms


def paraxial_trace_backward(
    rs: jnp.ndarray,
    vertices: jnp.ndarray,
    paraxial_ray: jnp.ndarray,
    start_surface: int,
    stop_surface: int,
    iors: tuple[jnp.ndarray, ...],
):
    for i in range(start_surface, stop_surface, -1):
        d = vertices[i]
        d = d if i == 0 else d - vertices[i - 1]
        T_transfer = jnp.array([[1, -d], [0, 1]])
        paraxial_ray = T_transfer @ paraxial_ray
        if i > 0:
            n1 = iors[i][0]
            n2 = iors[i - 1][0]
            T_refract = jnp.array([[1, 0], [(n1 - n2) / n2 / rs[i - 1], n1 / n2]])
            paraxial_ray = T_refract @ paraxial_ray
    return paraxial_ray


def paraxial_trace_forward(
    rs: jnp.ndarray,
    vertices: jnp.ndarray,
    paraxial_ray: jnp.ndarray,
    start_surface: int,
    stop_surface: int,
    iors: tuple[jnp.ndarray, ...],
):
    for i in range(start_surface, stop_surface + 1):
        d = vertices[i]
        d = d if i == 0 else d - vertices[i - 1]
        T_transfer = jnp.array([[1, d], [0, 1]])
        paraxial_ray = T_transfer @ paraxial_ray
        n1 = iors[i][0]
        n2 = iors[i + 1][0]
        T_refract = jnp.array([[1, 0], [(n1 - n2) / n2 / rs[i], n1 / n2]])
        paraxial_ray = T_refract @ paraxial_ray
    return paraxial_ray


def find_entrance_pupil_dist(
    rs: jnp.ndarray,
    vertices_z: jnp.ndarray,
    aperture_stop_index: int,
    pangle: float,
    iors: tuple[jnp.ndarray, ...],
):
    ray = paraxial_trace_backward(
        rs=rs,
        vertices=vertices_z,
        paraxial_ray=jnp.array([0.0, pangle]),
        start_surface=aperture_stop_index,
        stop_surface=0,
        iors=iors,
    )
    return -ray[0] / ray[1]


def find_effective_focal_length(
    rs: jnp.ndarray, vertices_z: jnp.ndarray, y1: float, iors: tuple[jnp.array, ...]
):
    _, alpha = paraxial_trace_forward(
        rs=rs,
        vertices=vertices_z,
        paraxial_ray=jnp.array([y1, 0]),
        start_surface=0,
        stop_surface=len(rs) - 1,
        iors=iors,
    )
    return -y1 / alpha


def find_image_plane_z(
    rs: jnp.ndarray,
    vertices_z: jnp.ndarray,
    marginal_ray_y0: float,
    marginal_ray_u0: float,
    iors: tuple[jnp.array, ...],
):
    """
    Paraxial solve for sensor z-coordinate
    """
    y_p, alpha = paraxial_trace_forward(
        rs=rs,
        vertices=vertices_z,
        paraxial_ray=jnp.array([marginal_ray_y0, marginal_ray_u0]),
        start_surface=0,
        stop_surface=len(rs) - 1,
        iors=iors,
    )
    return -y_p / alpha + vertices_z[-1]


def find_entrance_pupil_diameter(aperture: model.Aperture, object_z: float):
    match aperture.type:
        case "ED":
            if object_z != -jnp.inf:
                raise NotImplementedError(
                    "Finite distances to the object are not supported yet"
                )
            else:
                return aperture.max_d
        case _:
            raise ValueError(
                f"Apperture type {aperture.type} is unknown or not implemented yet"
            )


def find_paraxial_marginal_ray(aperture: model.Aperture, object_z: float):
    match aperture.type:
        case "ED":
            if object_z != -jnp.inf:
                raise NotImplementedError(
                    "Finite distances to the object are not supported yet"
                )
            else:
                return jnp.array([aperture.max_d / 2, 0.0])
        case _:
            raise ValueError(
                f"Apperture type {aperture.type} is unknown or not implemented yet"
            )


def apply_to_surfaces(f, vertices_z, flat_params, cnt_per_surface, types):
    it = 0
    results = []
    for i, tp in enumerate(types):
        cls = shapes.get_surface_class_by_name(tp)
        params = flat_params[it : it + cnt_per_surface[i]]
        output = f(cls.create(vertices_z[i], params))
        results.append(output)
        it += cnt_per_surface[i]
    return results


def find_limiting_vertices_z(vertices_z, flat_params, cnt_per_surface, types):
    f = lambda s: s.find_opposite_vertex()
    lvs = apply_to_surfaces(f, vertices_z, flat_params, cnt_per_surface, types)
    return jnp.array(lvs)


def find_up_corneres_y(
    vertices_z, flat_params, cnt_per_surface, types, limiting_vertices
):
    return jnp.ones_like(limiting_vertices) * jnp.inf


def get_paraxial_rs(flat_params, cnt_per_surface, types):
    it = 0
    rs = [0] * len(cnt_per_surface)
    for i, tp in enumerate(types):
        cls = shapes.get_surface_class_by_name(tp)
        r = cls.get_paraxial_radius(flat_params[it : it + cnt_per_surface[i]])
        it += cnt_per_surface[i]
        rs[i] = r
    return jnp.array(rs)


@partial(jax.jit, static_argnums=(0))
def compute_optical_properties(
    cnst: model.LensSystemConstants, vars: model.LensSystemVariables
) -> model.LensSystemComputedProperties:
    vertices_z = model.LensSystemVariables.resolve_verticies_z(vars.distances_z)
    rs = get_paraxial_rs(
        vars.curvature_parameters, cnst.num_parameters_per_surface, cnst.surface_types
    )
    entrance_pupil_z = find_entrance_pupil_dist(
        rs, vertices_z, cnst.aperture_stop_index, cnst.MAX_PARAXIAL_ANGLE, vars.iors
    )
    entrance_pupil_diam = find_entrance_pupil_diameter(cnst.aperture, cnst.object_z)
    paraxial_marginal_ray = find_paraxial_marginal_ray(cnst.aperture, cnst.object_z)
    effective_focal_length = find_effective_focal_length(
        rs, vertices_z, cnst.EFFL_PARAXIAL_Y, vars.iors
    )
    if cnst.sensor_z_mode == "paraxial_solve":
        sensor_z = find_image_plane_z(
            rs,
            vertices_z,
            marginal_ray_y0=paraxial_marginal_ray[0],
            marginal_ray_u0=paraxial_marginal_ray[1],
            iors=vars.iors,
        )
    elif cnst.sensor_z_mode == "fixed":
        sensor_z = cnst.sensor_z
    else:
        sensor_z = vertices_z[-1]

    limiting_vertices_z = find_limiting_vertices_z(
        vertices_z,
        vars.curvature_parameters,
        cnst.num_parameters_per_surface,
        cnst.surface_types,
    )
    up_corners_y = find_up_corneres_y(
        vertices_z,
        vars.curvature_parameters,
        cnst.num_parameters_per_surface,
        cnst.surface_types,
        limiting_vertices_z,
    )
    return model.LensSystemComputedProperties(
        iors=vars.iors,
        entrance_pupil_z=entrance_pupil_z,
        entrance_pupil_diameter=entrance_pupil_diam,
        effective_focal_length=effective_focal_length,
        paraxial_marginal_ray=paraxial_marginal_ray,
        vertices_z=vertices_z,
        limiting_vertices_z=limiting_vertices_z,
        up_corners_y=up_corners_y,
        sensor_z=sensor_z,
    )


def refract_rays(
    ray: RayRMSSpotSize,
    P: jnp.ndarray,
    surf: shapes.RotSymSurface,
    n1: jnp.ndarray,
    n2: jnp.ndarray,
    up_corener_y: jnp.ndarray,
):
    """
    Computes the refracted directions for a bundle of rays at a given optical surface,
    using vector form of Snell's law for rotationally symmetric surfaces.

    Args:
        ray: RayRMSSpotSize
            The input rays, including directions and masks.
        P: jnp.ndarray
            The intersection points of rays with the surface, shape (..., 3).
        surf: RotSymSurface
            The optical surface the rays intersect with, which provides surface normals.
        n1: jnp.ndarray
            Refractive indices of the incident medium (shape: [num_wavelengths]).
        n2: jnp.ndarray
            Refractive indices of the transmission medium (shape: [num_wavelengths]).
        up_corener_y: jnp.ndarray
            [Note: this argument is currently unused.]

    Returns:
        valid_refract: jnp.ndarray
            A boolean mask indicating which rays have valid (forward-going) refractions.
        r_prime_vec_normalized: jnp.ndarray
            The normalized refracted ray directions (shape: same as ray.d).

    References:
        M. J. Kidger, "Fundamental Optical Design", SPIE Press, 2001.
        - Eq. (3.30) — Surface normal at point of intersection.
        - Eq. (3.31) — Cosine of angle of incidence.
        - Eq. (3.37) — Cosine of angle of refraction from Snell's law.
        - Eq. (3.39) — Vector form of Snell's law for computing refracted direction.
    """

    # Compute surface normal at each intersection point Eq. (3.30)
    a_vec = surf.normal(P[..., 0], P[..., 1], P[..., 2])

    # Compute cosine of angle of incidence Eq. (3.31) in vector form
    cos_I = jnp.sum(ray.d * a_vec, axis=-1)

    # Compute cosine of refraction angle using Snell’s law Eq. (3.37)
    cos2_I = jnp.square(cos_I)
    q = n1 / n2
    q_expanded = q[None, :, None]

    # Argument of square root for cos(θ_r) (Eq. 3.37)
    cos2_I_prime_arg = 1.0 - q_expanded**2 * (1.0 - cos2_I)

    # Validity mask for refraction
    eps = 1e-8
    valid_refract = cos2_I_prime_arg >= eps  # shape: (n_ang, n_wl, n_rays)
    safe_arg = jnp.where(valid_refract, cos2_I_prime_arg, 1.0)

    # Compute cos(θ_r) only for valid rays
    cos_I_prime = jnp.sqrt(safe_arg)

    # Eq. (3.39) in vector form based on Eq. (3.15)
    # r' = q * r + (cos(θ') - q * cos(θ)) * n
    qq = cos_I_prime - q_expanded * cos_I
    r_prime_vec = q[None, :, None, None] * ray.d + qq[..., None] * a_vec

    # Replace invalid directions with [0, 0, 1]
    neutral = jnp.array([0.0, 0.0, 1.0])
    r_prime_vec = jnp.where(valid_refract[..., None], r_prime_vec, neutral)

    return valid_refract, normalize(r_prime_vec)


def real_trace_forward(
    flat_params: jnp.ndarray,
    vertices: jnp.ndarray,
    cnt_per_surface: jnp.ndarray,
    types: tuple,
    iors: jnp.ndarray,
    limiting_vertices_z: jnp.ndarray,
    up_corners_y: jnp.ndarray,
    rays: RayRMSSpotSize,
    sensor: shapes.Spheric,
) -> jnp.ndarray:
    """
    Trace rays through each surface, updating origins & directions,
    and return the Ray at the last surface (i.e. the image-side intersection).

    References:
        Chapter 3 in
        M. J. Kidger, "Fundamental Optical Design", SPIE Press, 2001.
    """
    it = 0
    for i, tp in enumerate(types):
        cls = shapes.get_surface_class_by_name(tp)
        surf = cls.create(vertices[i], flat_params[it : it + cnt_per_surface[i]])
        it += cnt_per_surface[i]
        valid_intersect, o1 = surf.ray_surface_intersection(
            rays, limiting_vertices_z[i]
        )
        valid_refract, d1 = refract_rays(
            rays, o1, surf, iors[i], iors[i + 1], up_corners_y[i]
        )
        new_mask = rays.mask & valid_intersect & valid_refract
        rays = RayRMSSpotSize(o=o1, d=d1, mask=new_mask)
    _, o1 = sensor.ray_surface_intersection(rays, sensor.vertex_z)
    return RayRMSSpotSize(o=o1, d=rays.d, mask=rays.mask)


def real_trace_forward_with_penalties(
    flat_params: jnp.ndarray,
    vertices: jnp.ndarray,
    cnt_per_surface: jnp.ndarray,
    types: tuple,
    iors: jnp.ndarray,
    limiting_vertices_z: jnp.ndarray,
    up_corners_y: jnp.ndarray,
    rays: RayRMSSpotSize,
    sensor: shapes.Spheric,
) -> jnp.ndarray:
    """
    Trace rays through each surface, updating origins & directions,
    and return the Ray at the last surface (i.e. the image-side intersection).

    References:
        Chapter 3 in
        M. J. Kidger, "Fundamental Optical Design", SPIE Press, 2001.
    """
    it = 0
    penalty = jnp.array(0.0)
    for i, tp in enumerate(types):
        cls = shapes.get_surface_class_by_name(tp)
        surf = cls.create(vertices[i], flat_params[it : it + cnt_per_surface[i]])
        it += cnt_per_surface[i]
        valid_intersect, o1 = surf.ray_surface_intersection(
            rays, limiting_vertices_z[i]
        )
        valid_refract, d1 = refract_rays(
            rays, o1, surf, iors[i], iors[i + 1], up_corners_y[i]
        )
        new_mask = rays.mask & valid_intersect & valid_refract
        dropped = rays.mask & (~new_mask)
        penalty = penalty + jnp.sum(dropped)
        rays = RayRMSSpotSize(o=o1, d=d1, mask=new_mask)
    _, o1 = sensor.ray_surface_intersection(rays, sensor.vertex_z)
    return RayRMSSpotSize(o=o1, d=rays.d, mask=rays.mask), penalty


def real_trace_forward_between_surfaces(
    start_surface_num: int,
    stop_surface_num: int,
    flat_params: jnp.ndarray,
    vertices: jnp.ndarray,
    cnt_per_surface: jnp.ndarray,
    types: tuple,
    iors: jnp.ndarray,
    limiting_vertices_z: jnp.ndarray,
    up_corners_y: jnp.ndarray,
    rays: RayRMSSpotSize,
    sensor: shapes.Spheric,
) -> jnp.ndarray:
    """
    Trace rays through each surface, updating origins & directions,
    and return the Ray at the last surface (i.e. the image-side intersection).

    References:
        Chapter 3 in
        Kidger, M. J. (2001). Fundamental optical design. SPIE Optical Engineering Press.
    """
    it = 0
    for i in range(min(start_surface_num, len(types))):
        it += cnt_per_surface[i]
    for i in range(start_surface_num, min(stop_surface_num + 1, len(types))):
        tp = types[i]
        cls = shapes.get_surface_class_by_name(tp)
        surf = cls.create(vertices[i], flat_params[it : it + cnt_per_surface[i]])
        it += cnt_per_surface[i]
        valid_intersect, o1 = surf.ray_surface_intersection(
            rays, limiting_vertices_z[i]
        )
        valid_refract, d1 = refract_rays(
            rays, o1, surf, iors[i], iors[i + 1], up_corners_y[i]
        )
        new_mask = rays.mask & valid_intersect & valid_refract
        rays = RayRMSSpotSize(o=o1, d=d1, mask=new_mask)
        if i == stop_surface_num:
            return rays
    _, o1 = sensor.ray_surface_intersection(rays, sensor.vertex_z)
    return RayRMSSpotSize(o=o1, d=rays.d, mask=rays.mask)


def real_trace_forward_with_subscribers(
    flat_params: jnp.ndarray,  # Flattened surface parameters
    vertices: jnp.ndarray,  # (num_surfaces,) - surface vertices
    cnt_per_surface: jnp.ndarray,  # (num_surfaces,) - param count per surface
    types: Tuple[str, ...],  # (num_surfaces,) - surface type names
    iors: jnp.ndarray,  # (num_surfaces + 1,) - refractive indices
    limiting_vertices_z: jnp.ndarray,  # (num_surfaces,) - axial limits for intersection
    up_corners_y: jnp.ndarray,  # (num_surfaces,) - upper Y corners
    rays: RayRMSSpotSize,  # Input rays to trace
    sensor: shapes.Spheric,  # Sensor surface (final element)
    on_intersection: Callable[[int, RayRMSSpotSize, jnp.ndarray, jnp.ndarray], None],
    on_refraction: Callable[
        [int, RayRMSSpotSize, jnp.ndarray, jnp.ndarray, jnp.ndarray], None
    ],
) -> RayRMSSpotSize:
    """
    Traces the given rays through an optical system defined by a sequence of surfaces,
    recording each surface interaction via `on_intersection` and `on_refraction`.

    Args:
        flat_params: Concatenated surface parameters.
        vertices: Vertex coordinates for each surface.
        cnt_per_surface: Number of parameters per surface.
        types: Names of surface types (e.g., 'Spheric', 'Aspheric').
        iors: Refractive indices for each region between surfaces.
        limiting_vertices_z: Z-bounds used to validate intersection.
        up_corners_y: Upper corner y-values used for Fresnel zone check.
        rays: Initial rays to trace.
        sensor: Final image-side surface (e.g., detector).
        on_intersection: Callback triggered after each surface intersection.
        on_refraction: Callback triggered after each surface refraction.

    Returns:
        RayRMSSpotSize: The final ray bundle intersected with the sensor.
    """
    param_offset = 0
    for i, surface_type in enumerate(types):
        surface_class = shapes.get_surface_class_by_name(surface_type)
        surface = surface_class.create(
            vertices[i], flat_params[param_offset : param_offset + cnt_per_surface[i]]
        )
        param_offset += cnt_per_surface[i]

        valid_intersect, intersection_pts = surface.ray_surface_intersection(
            rays, limiting_vertices_z[i]
        )
        on_intersection(i, rays, valid_intersect, intersection_pts)

        valid_refract, refracted_dirs = refract_rays(
            rays, intersection_pts, surface, iors[i], iors[i + 1], up_corners_y[i]
        )
        valid_total = rays.mask & valid_intersect & valid_refract
        on_refraction(i, rays, intersection_pts, valid_total, refracted_dirs)

        rays = RayRMSSpotSize(o=intersection_pts, d=refracted_dirs, mask=valid_total)

    # Final intersection with the image-side sensor
    _, final_intersection = sensor.ray_surface_intersection(rays, sensor.vertex_z)
    on_intersection(len(types), rays, rays.mask, final_intersection)

    return RayRMSSpotSize(o=final_intersection, d=rays.d, mask=rays.mask)


def edge_thickness_constraint(
    cnst: model.LensSystemConstants,
    curvature_parameters: jnp.ndarray,
    vertices_z: jnp.ndarray,
    iors: jnp.ndarray,
    props: model.LensSystemComputedProperties,
    sensor_plane: shapes.Spheric,
):
    edge_thickness_rays = RayRMSSpotSize.create_edge_thickness_rays(
        cnst.edge_thickness_field_factors,
        cnst.edge_thickness_entrance_pupil_factors,
        props.entrance_pupil_z,
        props.entrance_pupil_diameter,
        cnst.lens_field,
    )
    zs = []
    real_trace_forward_with_subscribers(
        flat_params=curvature_parameters,
        vertices=vertices_z,
        cnt_per_surface=cnst.num_parameters_per_surface,
        types=cnst.surface_types,
        iors=iors[..., 0][:, None],
        limiting_vertices_z=props.limiting_vertices_z,
        up_corners_y=props.up_corners_y,
        rays=edge_thickness_rays,
        sensor=sensor_plane,
        on_intersection=lambda surf_num, rays, valid_intersection, o: zs.append(
            o[..., 2].flatten()
        ),
        on_refraction=lambda surf_num, rays, o1, valid_refraction, d1: None,
    )
    zs_with_sensor = jnp.array(zs)
    deltas = zs_with_sensor[1:] - zs_with_sensor[:-1]  # shape (num_surfaces, N)
    min_deltas = jnp.min(deltas, axis=1)  # shape (num_surfaces,)
    delta_violation = jnp.minimum(0.0, min_deltas - cnst.target_edge_thickness)
    penalty = jnp.sum(delta_violation**2)
    return penalty


def edge_thickness_constraint_inline(
    cnst: model.LensSystemConstants,
    curvature_parameters: jnp.ndarray,
    vertices_z: jnp.ndarray,
    iors_: jnp.ndarray,
    props: model.LensSystemComputedProperties,
    sensor_plane: shapes.Spheric,
) -> jnp.ndarray:
    """
    Penalizes surfaces that produce too thin edge thickness (z distance between rays).

    Returns:
        A scalar penalty: sum of squared violations (negative deviations).
    """

    rays = RayRMSSpotSize.create_edge_thickness_rays(
        cnst.edge_thickness_field_factors,
        cnst.edge_thickness_entrance_pupil_factors,
        props.entrance_pupil_z,
        props.entrance_pupil_diameter,
        cnst.lens_field,
    )

    total_penalty = 0.0
    param_offset = 0
    z_it0 = None
    iors = iors_[..., 0][:, None]

    for i in range(cnst.num_surfaces):
        surface_type = cnst.surface_types[i]
        surface_class = shapes.get_surface_class_by_name(surface_type)
        num_params = cnst.num_parameters_per_surface[i]
        surface_params = curvature_parameters[param_offset : param_offset + num_params]
        surface = surface_class.create(vertices_z[i], surface_params)
        param_offset += num_params

        valid_intersect, o1 = surface.ray_surface_intersection(
            rays, props.limiting_vertices_z[i]
        )
        valid_refract, d1 = refract_rays(
            rays, o1, surface, iors[i], iors[i + 1], props.up_corners_y[i]
        )

        valid_mask = rays.mask & valid_intersect & valid_refract
        rays = RayRMSSpotSize(o=o1, d=d1, mask=valid_mask)

        z_it1 = o1[..., 2].flatten()  # shape (N,)

        if z_it0 is not None:
            # Only compute deltas for valid rays
            z_diff = z_it1 - z_it0
            z_diff = jnp.where(
                valid_mask.flatten(), z_diff, jnp.inf
            )  # mask invalid rays
            min_delta = jnp.min(z_diff)
            penalty = jnp.square(
                jnp.minimum(0.0, min_delta - cnst.target_edge_thickness)
            )
            total_penalty += penalty

        z_it0 = z_it1

    # Final intersection with sensor
    _, o_sensor = sensor_plane.ray_surface_intersection(rays, sensor_plane.vertex_z)
    z_it1 = o_sensor[..., 2].flatten()

    if z_it0 is not None:
        z_diff = z_it1 - z_it0
        z_diff = jnp.where(rays.mask.flatten(), z_diff, jnp.inf)
        min_delta = jnp.min(z_diff)
        penalty = jnp.square(jnp.minimum(0.0, min_delta - cnst.target_edge_thickness))
        total_penalty += penalty

    return total_penalty


def constraint_axis_thickness(
    target_axis_thickness: float, verticies_z: jnp.ndarray
) -> jnp.ndarray:
    d = verticies_z[1:] - verticies_z[:-1] - target_axis_thickness
    penalties = jnp.where(d < 0, d**2, 0.0)
    return jnp.sum(penalties)


def constraint_free_working_distance(
    target_free_working_distance: float,
    verticies_z: jnp.ndarray,
    sensor_plain_z: jnp.ndarray,
    sensor_z_mode: str,
) -> jnp.ndarray:

    if sensor_z_mode == "optimize":
        d = verticies_z[-1] - verticies_z[-2] - target_free_working_distance
    else:
        d = sensor_plain_z - verticies_z[-1] - target_free_working_distance
    return jnp.where(d < 0, d**2, 0.0)


def constraint_effective_focal_length(target_effl: float, effl: jnp.ndarray):
    return (effl - target_effl) ** 2
