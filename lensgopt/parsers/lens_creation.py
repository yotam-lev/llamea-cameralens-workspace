from pathlib import Path
import lensgopt.materials.refractive_index_catalogs as refractive_index_catalogs
import lensgopt.optics.meta as meta
import lensgopt.optics.model as model
import lensgopt.optics.shapes as shapes
import lensgopt.parsers.zmax_fmt as zmax_fmt

import jax
import jax.numpy as jnp


def create_double_gauss(sensor_z_mode):
    jax.config.update("jax_enable_x64", True)
    HERE = Path(__file__).resolve().parent
    lenszmax = zmax_fmt.parse_lens_zmax_format(
        HERE / "lenses" / "double_gauss_schott.txt"
    )

    wls = tuple([meta.LAMBDA_D, meta.LAMBDA_C, meta.LAMBDA_F])
    ior_catalogs, iors = refractive_index_catalogs.resolve_catalogs_and_iors(
        lenszmax.material_names, wls
    )
    material_ids = refractive_index_catalogs.resolve_material_ids(
        lenszmax.material_names, ior_catalogs
    )
    num_parameters_per_surface, surface_types, curvature_parameters, distances_z = (
        shapes.flatten_surfaces(lenszmax.surfaces)
    )

    if sensor_z_mode == "optimize":
        distances_z = jnp.concatenate([distances_z, jnp.array([lenszmax.sensor_distance])])

    cnst = model.LensSystemConstants(
        lens_field=lenszmax.lens_field,
        aperture=lenszmax.lens_aperture,
        num_surfaces=len(lenszmax.surfaces),
        num_parameters_per_surface=num_parameters_per_surface,
        surface_types=surface_types,
        ior_catalogs=ior_catalogs,
        aperture_stop_index=lenszmax.aperture_stop_index,
        target_effective_focal_length=99.5,
        target_edge_thickness=0.25,
        target_axis_thickness=0.10,
        target_free_working_distance=3.0,
        wavelengths=wls,
        field_factors=tuple([0.0, 0.7, 1.0]),
        object_z=-lenszmax.object_distance,
        sensor_z=lenszmax.surfaces[-1].vertex_z + lenszmax.sensor_distance,
        sensor_z_mode=sensor_z_mode,
        is_iors_grad=False,
        n_rays_row=31,
    )

    vars = model.LensSystemVariables(
        curvature_parameters=curvature_parameters, distances_z=distances_z, iors=iors
    )
    return cnst, vars, material_ids
