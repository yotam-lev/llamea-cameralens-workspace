from dataclasses import dataclass
from functools import partial
from typing import Callable, ClassVar, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

import lensgopt.materials.refractive_index_catalogs as ior_catalogs
import lensgopt.optics.model as model
import lensgopt.optics.optics as optics
import lensgopt.optics.shapes as shapes


def _split_sections(
    vector: jnp.ndarray, sizes: Sequence[int]
) -> Tuple[jnp.ndarray, ...]:
    splits = []
    start = 0
    for size in sizes:
        splits.append(vector[start : start + size])
        start += size
    return tuple(splits)


def _apply_updates(
    template: jnp.ndarray,
    idx: jnp.ndarray,
    updates: Optional[jnp.ndarray],
) -> jnp.ndarray:
    if updates is None or idx.size == 0:
        return template
    return template.at[idx].set(updates)


def _round_material_ids(material_ids: jnp.ndarray) -> jnp.ndarray:
    return jnp.round(material_ids).astype(int)


def _gather(template: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    if idx.size == 0:
        return jnp.zeros((0,), dtype=template.dtype)
    return template[idx]


def _concat_parts(parts: Sequence[jnp.ndarray]) -> jnp.ndarray:
    filtered = [part for part in parts if part.size]
    if not filtered:
        dtype = parts[0].dtype if parts else jnp.float32
        return jnp.zeros((0,), dtype=dtype)
    if len(filtered) == 1:
        return filtered[0]
    return jnp.concatenate(filtered)


def _resolve_iors(
    material_ids: jnp.ndarray, catalogs: Sequence[ior_catalogs.RefractiveIndexCatalog]
) -> jnp.ndarray:
    return ior_catalogs.resolve_iors(material_ids.astype(int), catalogs)


class _LossFactoryBase:
    grad_argnums: ClassVar = 0
    jacobian_argnums: ClassVar = 0
    hessian_argnums: ClassVar = 0
    vmap_in_axes: ClassVar = (0,)

    def _evaluate(self, *args) -> jnp.ndarray:
        raise NotImplementedError

    def _evaluate_full(self, *args) -> jnp.ndarray:
        raise NotImplementedError

    def make_jit_loss(self):
        return jax.jit(self._evaluate)

    def make_jit_loss_full(self):
        return jax.jit(self._evaluate_full)

    def make_grad(self):
        return jax.grad(self.make_jit_loss(), argnums=self.grad_argnums)

    def make_value_and_grad(self):
        return jax.value_and_grad(self.make_jit_loss(), argnums=self.grad_argnums)

    def make_jacobian(self):
        return jax.jacrev(self.make_jit_loss_full(), argnums=self.jacobian_argnums)

    def make_hessian(self):
        return jax.hessian(self.make_jit_loss(), argnums=self.hessian_argnums)

    def make_vmap_loss(self):
        return jax.jit(jax.vmap(self._evaluate, in_axes=self.vmap_in_axes))

    def make_loss_suite(self):
        """Return the standard suite of loss utilities for optimisation pipelines."""
        loss_full = self.make_jit_loss_full()
        loss = self.make_jit_loss()
        grad = self.make_grad()
        hess = self.make_hessian()
        loss_vmap = self.make_vmap_loss()
        return loss_full, loss, grad, hess, loss_vmap


@partial(jax.jit, static_argnums=(0))
def loss(
    cnst: model.LensSystemConstants,
    flat_params: jnp.ndarray,
    dists: jnp.ndarray,
    iors_: jnp.ndarray,
):
    if cnst.is_iors_grad:
        iors = iors_
    else:
        iors = jax.lax.stop_gradient(iors_)
    vars = model.LensSystemVariables(flat_params, dists, iors)
    props: model.LensSystemComputedProperties = optics.compute_optical_properties(
        cnst, vars
    )
    rays = optics.RayRMSSpotSize.create_grid_rays_angle_field_for_rms_spot(
        n_rays_row=cnst.n_rays_row,
        n_wl=len(cnst.wavelengths),
        r_ep=props.entrance_pupil_diameter / 2,
        z_ep=props.entrance_pupil_z,
        angles_z_deg=jnp.array(cnst.field_factors) * cnst.lens_field.max_field,
    )
    chr = optics.RayRMSSpotSize.create_chief_rays(
        len(cnst.wavelengths),
        z_ep=props.entrance_pupil_z,
        angles_z_deg=jnp.array(cnst.field_factors) * cnst.lens_field.max_field,
    )
    rays = optics.RayRMSSpotSize.append_ray(rays, chr)
    vertices_z = model.LensSystemVariables.resolve_verticies_z(dists)
    sensor_plane = shapes.Spheric(props.sensor_z, jnp.inf)
    final_rays, penalty = optics.real_trace_forward_with_penalties(
        flat_params,
        vertices_z,
        cnst.num_parameters_per_surface,
        cnst.surface_types,
        iors,
        props.limiting_vertices_z,
        props.up_corners_y,
        rays,
        sensor_plane,
    )
    rms_term = final_rays.compute_rms2_spot_with_appended_chief_centroid()
    edge_thickness_penalty = optics.edge_thickness_constraint_inline(
        cnst, flat_params, vertices_z, iors, props, sensor_plane
    )
    on_axis_thickness_penalty = optics.constraint_axis_thickness(
        cnst.target_axis_thickness, vertices_z
    )
    free_working_distance_penalty = optics.constraint_free_working_distance(
        cnst.target_free_working_distance,
        vertices_z,
        sensor_plane.vertex_z,
        cnst.sensor_z_mode,
    )
    effl_penalty = optics.constraint_effective_focal_length(
        cnst.target_effective_focal_length, props.effective_focal_length
    )
    return jnp.array(
        [
            rms_term,
            10 * penalty,
            edge_thickness_penalty,
            on_axis_thickness_penalty,
            free_working_distance_penalty,
            effl_penalty,
        ]
    )


@dataclass(frozen=True)
class LossFactoryWithIORs(_LossFactoryBase):
    loss_full: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    curvatures_template: jnp.ndarray
    distances_template: jnp.ndarray
    iors_template: jnp.ndarray
    curvatures_optimized_idx: jnp.ndarray
    distances_optimized_idx: jnp.ndarray
    materials_optimized_idx: jnp.ndarray
    vmap_in_axes: ClassVar = (0, 0)

    def _reconstruct_params(
        self, x: jnp.ndarray, iors_x: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        curvatures_x, distances_x = _split_sections(
            x,
            (
                self.curvatures_optimized_idx.size,
                self.distances_optimized_idx.size,
            ),
        )
        curvatures = _apply_updates(
            self.curvatures_template, self.curvatures_optimized_idx, curvatures_x
        )
        distances = _apply_updates(
            self.distances_template, self.distances_optimized_idx, distances_x
        )
        iors = _apply_updates(self.iors_template, self.materials_optimized_idx, iors_x)
        return curvatures, distances, iors

    def _evaluate(self, x: jnp.ndarray, iors_x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(self.loss_full(*self._reconstruct_params(x, iors_x)))

    def _evaluate_full(self, x: jnp.ndarray, iors_x: jnp.ndarray) -> jnp.ndarray:
        return self.loss_full(*self._reconstruct_params(x, iors_x))

    def make_grad(self, wrt_iors: bool = True):
        jax_evaluate = self.make_jit_loss()
        if wrt_iors:
            return jax.grad(jax_evaluate, argnums=(0, 1))
        return jax.grad(jax_evaluate, argnums=0)

    def make_value_and_grad(self, wrt_iors: bool = True):
        jax_evaluate = self.make_jit_loss()
        if wrt_iors:
            return jax.value_and_grad(jax_evaluate, argnums=(0, 1))
        return jax.value_and_grad(jax_evaluate, argnums=0)

    def make_jacobian(self, wrt_iors: bool = True):
        jax_loss_full = self.make_jit_loss_full()
        if wrt_iors:
            return jax.jacrev(
                jax_loss_full,
                argnums=(0, 1),
            )
        return jax.jacrev(
            jax_loss_full,
            argnums=0,
        )

    def make_hessian(self):
        # no iors
        jax_evaluate = self.make_jit_loss()
        return jax.hessian(jax_evaluate, argnums=0)

    def init_from_templates(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return (
            _concat_parts(
                [
                    _gather(self.curvatures_template, self.curvatures_optimized_idx),
                    _gather(self.distances_template, self.distances_optimized_idx),
                ]
            ),
            _gather(self.iors_template, self.materials_optimized_idx),
        )

    def unpack_full(self, x, iors_x) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self._reconstruct_params(x, iors_x)

    def pack_from_full(
        self, curvatures, distances, iors
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return (
            _concat_parts(
                [
                    _gather(curvatures, self.curvatures_optimized_idx),
                    _gather(distances, self.distances_optimized_idx),
                ]
            ),
            _gather(iors, self.materials_optimized_idx),
        )


@dataclass(frozen=True)
class LossFactoryWithMaterials(_LossFactoryBase):
    loss_full: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    curvatures_template: jnp.ndarray
    distances_template: jnp.ndarray
    material_ids_template: jnp.ndarray
    curvatures_optimized_idx: jnp.ndarray
    distances_optimized_idx: jnp.ndarray
    materials_optimized_idx: jnp.ndarray
    catalogs: Sequence[ior_catalogs.RefractiveIndexCatalog]
    vmap_in_axes: ClassVar = (0, 0)

    @property
    def sz_packed_curvatures(self):
        return self.curvatures_optimized_idx.size

    @property
    def sz_packed_distances(self):
        return self.distances_optimized_idx.size

    @property
    def sz_packed_materials(self):
        return self.materials_optimized_idx.size

    def _resolve_iors(self, material_ids: jnp.ndarray) -> jnp.ndarray:
        return _resolve_iors(_round_material_ids(material_ids), self.catalogs)

    def _reconstruct_params(
        self, x: jnp.ndarray, material_ids: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        curvatures_x, distances_x = _split_sections(
            x,
            (
                self.curvatures_optimized_idx.size,
                self.distances_optimized_idx.size,
            ),
        )
        curvatures = _apply_updates(
            self.curvatures_template, self.curvatures_optimized_idx, curvatures_x
        )
        distances = _apply_updates(
            self.distances_template, self.distances_optimized_idx, distances_x
        )
        all_material_ids = _apply_updates(
            self.material_ids_template,
            self.materials_optimized_idx,
            _round_material_ids(material_ids),
        )
        iors = self._resolve_iors(all_material_ids)
        return curvatures, distances, iors

    def _evaluate(self, x: jnp.ndarray, material_ids: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(self.loss_full(*self._reconstruct_params(x, material_ids)))

    def _evaluate_full(self, x: jnp.ndarray, material_ids: jnp.ndarray) -> jnp.ndarray:
        return self.loss_full(*self._reconstruct_params(x, material_ids))

    def _evaluate_from_flat_parameters(self, parameters: jnp.ndarray) -> jnp.ndarray:
        total_x_dim = (
            self.curvatures_optimized_idx.size + self.distances_optimized_idx.size
        )
        x_c = parameters[: self.curvatures_optimized_idx.size]
        x_d = parameters[self.curvatures_optimized_idx.size : total_x_dim]
        x = _concat_parts([1.0 / x_c, x_d])
        material_ids = parameters[total_x_dim:]
        return self._evaluate(x, material_ids)

    def make_hessian(self):
        # no iors
        jax_evaluate = self.make_jit_loss()
        return jax.hessian(jax_evaluate, argnums=0)

    def make_jit_loss_flat(self):
        return jax.jit(self._evaluate_from_flat_parameters)

    def make_grad(self):
        return super().make_grad()

    def make_value_and_grad(self):
        return super().make_value_and_grad()

    def make_jacobian(self):
        return super().make_jacobian()

    def make_vmap_loss(self):
        return super().make_vmap_loss()

    def init_from_templates(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return (
            _concat_parts(
                [
                    _gather(self.curvatures_template, self.curvatures_optimized_idx),
                    _gather(self.distances_template, self.distances_optimized_idx),
                ]
            ),
            _gather(self.material_ids_template, self.materials_optimized_idx),
        )

    def init_flat_from_templates(self) -> jnp.ndarray:
        return self.pack_flat_from_full(
            self.curvatures_template,
            self.distances_template,
            self.material_ids_template,
        )

    def unpack_full_with_iors(
        self, x, material_ids
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self._reconstruct_params(x, material_ids)

    def unpack_full_with_materials(
        self, x, material_ids
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        all_material_ids = self.material_ids_template.at[
            self.materials_optimized_idx
        ].set(jnp.round(material_ids).astype(int))
        c, d, _ = self._reconstruct_params(x, material_ids)
        return c, d, all_material_ids

    def unpack_flat_full_with_materials(
        self, parameters
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        total_x_dim = (
            self.curvatures_optimized_idx.size + self.distances_optimized_idx.size
        )
        x_c = parameters[: self.curvatures_optimized_idx.size]
        x_d = parameters[self.curvatures_optimized_idx.size : total_x_dim]
        x = _concat_parts([1.0 / x_c, x_d])
        material_ids = parameters[total_x_dim:]
        return self.unpack_full_with_materials(x, material_ids)

    def pack_from_full(
        self, curvatures, distances, material_ids
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return (
            _concat_parts(
                [
                    _gather(curvatures, self.curvatures_optimized_idx),
                    _gather(distances, self.distances_optimized_idx),
                ]
            ),
            _gather(jnp.array(material_ids), self.materials_optimized_idx),
        )

    def pack_flat_from_full(self, curvatures, distances, material_ids) -> jnp.ndarray:
        return _concat_parts(
            [
                1.0 / _gather(curvatures, self.curvatures_optimized_idx),
                _gather(distances, self.distances_optimized_idx),
                _gather(jnp.array(material_ids), self.materials_optimized_idx),
            ]
        )

    def pack_iors_from_packed_material_ids(self, material_ids: jnp.ndarray):
        all_material_ids = _apply_updates(
            self.material_ids_template,
            self.materials_optimized_idx,
            _round_material_ids(material_ids),
        )
        iors = self._resolve_iors(all_material_ids)
        return iors[self.materials_optimized_idx]


@dataclass(frozen=True)
class LossFactoryWithFlat:
    loss_factory_with_catalogs: LossFactoryWithMaterials

    def _extract_x_materials(self, parameters):
        total_x_dim = (
            self.loss_factory_with_catalogs.curvatures_optimized_idx.size
            + self.loss_factory_with_catalogs.distances_optimized_idx.size
        )
        x = parameters[:total_x_dim]
        material_ids = parameters[total_x_dim:]
        return x, material_ids

    def _evaluate_from_flat_parameters(self, parameters: jnp.ndarray) -> jnp.ndarray:
        x, material_ids = self._extract_x_materials(parameters)
        return self.loss_factory_with_catalogs._evaluate(x, material_ids)

    def make_jit_loss(self):
        return jax.jit(self._evaluate_from_flat_parameters)

    def init_from_templates(self) -> jnp.ndarray:
        factory = self.loss_factory_with_catalogs
        return _concat_parts(
            [
                _gather(factory.curvatures_template, factory.curvatures_optimized_idx),
                _gather(factory.distances_template, factory.distances_optimized_idx),
                _gather(factory.material_ids_template, factory.materials_optimized_idx),
            ]
        )

    def unpack_full(self, parameters) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x, material_ids = self._extract_x_materials(parameters)
        return self.loss_factory_with_catalogs._reconstruct_params(x, material_ids)

    def pack_from_full(self, curvatures, distances, material_ids) -> jnp.ndarray:
        factory = self.loss_factory_with_catalogs
        return _concat_parts(
            [
                _gather(curvatures, factory.curvatures_optimized_idx),
                _gather(distances, factory.distances_optimized_idx),
                _gather(material_ids, factory.materials_optimized_idx),
            ]
        )


@dataclass(frozen=True)
class CurvatureLossFactory(_LossFactoryBase):
    loss_full: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    curvatures_template: jnp.ndarray
    distances_template: jnp.ndarray
    iors_template: jnp.ndarray
    curvatures_optimized_idx: jnp.ndarray

    def _reconstruct_params(
        self, curvatures_x: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        curvatures = _apply_updates(
            self.curvatures_template, self.curvatures_optimized_idx, curvatures_x
        )
        distances = self.distances_template
        return curvatures, distances, self.iors_template

    def _evaluate(self, curvatures_x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(self.loss_full(*self._reconstruct_params(curvatures_x)))

    def _evaluate_full(self, curvatures_x: jnp.ndarray) -> jnp.ndarray:
        return self.loss_full(*self._reconstruct_params(curvatures_x))

    def init_from_templates(self) -> jnp.ndarray:
        return _gather(self.curvatures_template, self.curvatures_optimized_idx)

    def unpack_full(
        self, curvatures_x: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self._reconstruct_params(curvatures_x)

    def pack_from_full(self, curvatures, distances, iors) -> jnp.ndarray:
        return _gather(curvatures, self.curvatures_optimized_idx)


@dataclass(frozen=True)
class InverseCurvatureLossFactory(CurvatureLossFactory):
    def _reconstruct_params(
        self, curvatures_x: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        curvatures = _apply_updates(
            self.curvatures_template, self.curvatures_optimized_idx, 1 / curvatures_x
        )
        distances = self.distances_template
        return curvatures, distances, self.iors_template

    def init_from_templates(self) -> jnp.ndarray:
        return 1 / _gather(self.curvatures_template, self.curvatures_optimized_idx)

    def pack_from_full(self, curvatures, distances, iors) -> jnp.ndarray:
        return 1 / _gather(curvatures, self.curvatures_optimized_idx)


@dataclass(frozen=True)
class ThicknessLossFactory(_LossFactoryBase):
    loss_full: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    curvatures_template: jnp.ndarray
    distances_template: jnp.ndarray
    iors_template: jnp.ndarray
    distances_optimized_idx: jnp.ndarray

    def _reconstruct_params(
        self, distances_x: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        curvatures = self.curvatures_template
        distances = _apply_updates(
            self.distances_template, self.distances_optimized_idx, distances_x
        )
        return curvatures, distances, self.iors_template

    def _evaluate(self, distances_x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(self.loss_full(*self._reconstruct_params(distances_x)))

    def _evaluate_full(self, distances_x: jnp.ndarray) -> jnp.ndarray:
        return self.loss_full(*self._reconstruct_params(distances_x))

    def init_from_templates(self) -> jnp.ndarray:
        return _gather(self.distances_template, self.distances_optimized_idx)

    def unpack_full(
        self, distances_x: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self._reconstruct_params(distances_x)

    def pack_from_full(self, curvatures, distances, iors) -> jnp.ndarray:
        return _gather(distances, self.distances_optimized_idx)


@dataclass(frozen=True)
class GlassLossFactory(_LossFactoryBase):
    loss_full: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    curvatures_template: jnp.ndarray
    distances_template: jnp.ndarray
    material_ids_template: jnp.ndarray
    materials_optimized_idx: jnp.ndarray
    catalogs: Sequence[ior_catalogs.RefractiveIndexCatalog]

    def _reconstruct_params(
        self, material_ids: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        curvatures = self.curvatures_template
        distances = self.distances_template
        all_material_ids = _apply_updates(
            self.material_ids_template,
            self.materials_optimized_idx,
            _round_material_ids(material_ids),
        )
        iors = _resolve_iors(all_material_ids, self.catalogs)
        return curvatures, distances, iors

    def _evaluate(self, material_ids: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(self.loss_full(*self._reconstruct_params(material_ids)))

    def _evaluate_full(self, material_ids: jnp.ndarray) -> jnp.ndarray:
        return self.loss_full(*self._reconstruct_params(material_ids))

    def init_from_templates(self) -> jnp.ndarray:
        return _gather(self.material_ids_template, self.materials_optimized_idx)

    def unpack_full(
        self, material_ids: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self._reconstruct_params(material_ids)

    def pack_from_full(self, curvatures, distances, material_ids) -> jnp.ndarray:
        return _gather(material_ids, self.materials_optimized_idx)

    def pack_iors_from_packed_material_ids(
        self, material_ids: jnp.ndarray
    ) -> jnp.ndarray:
        all_material_ids = _apply_updates(
            self.material_ids_template,
            self.materials_optimized_idx,
            _round_material_ids(material_ids),
        )
        iors = _resolve_iors(all_material_ids, self.catalogs)
        return _gather(iors, self.materials_optimized_idx)


@dataclass(frozen=True)
class CurvatureThicknessLossFactory(_LossFactoryBase):
    loss_full: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    curvatures_template: jnp.ndarray
    distances_template: jnp.ndarray
    iors_template: jnp.ndarray
    curvatures_optimized_idx: jnp.ndarray
    distances_optimized_idx: jnp.ndarray

    def _reconstruct_params(
        self, x: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        curvatures_x, distances_x = _split_sections(
            x,
            (
                self.curvatures_optimized_idx.size,
                self.distances_optimized_idx.size,
            ),
        )
        curvatures = _apply_updates(
            self.curvatures_template, self.curvatures_optimized_idx, curvatures_x
        )
        distances = _apply_updates(
            self.distances_template, self.distances_optimized_idx, distances_x
        )
        return curvatures, distances, self.iors_template

    def _evaluate(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(self.loss_full(*self._reconstruct_params(x)))

    def _evaluate_full(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.loss_full(*self._reconstruct_params(x))

    def init_from_templates(self) -> jnp.ndarray:
        return _concat_parts(
            [
                _gather(self.curvatures_template, self.curvatures_optimized_idx),
                _gather(self.distances_template, self.distances_optimized_idx),
            ]
        )

    def unpack_full(
        self, x: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self._reconstruct_params(x)

    def pack_from_full(self, curvatures, distances, iors) -> jnp.ndarray:
        return _concat_parts(
            [
                _gather(curvatures, self.curvatures_optimized_idx),
                _gather(distances, self.distances_optimized_idx),
            ]
        )


@dataclass(frozen=True)
class InverseCurvatureThicknessLossFactory(CurvatureThicknessLossFactory):
    def _reconstruct_params(
        self, x: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        curvatures_x, distances_x = _split_sections(
            x,
            (
                self.curvatures_optimized_idx.size,
                self.distances_optimized_idx.size,
            ),
        )
        curvatures = _apply_updates(
            self.curvatures_template, self.curvatures_optimized_idx, 1 / curvatures_x
        )
        distances = _apply_updates(
            self.distances_template, self.distances_optimized_idx, distances_x
        )
        return curvatures, distances, self.iors_template

    def init_from_templates(self) -> jnp.ndarray:
        return _concat_parts(
            [
                1 / _gather(self.curvatures_template, self.curvatures_optimized_idx),
                _gather(self.distances_template, self.distances_optimized_idx),
            ]
        )

    def pack_from_full(self, curvatures, distances, iors) -> jnp.ndarray:
        return _concat_parts(
            [
                1 / _gather(curvatures, self.curvatures_optimized_idx),
                _gather(distances, self.distances_optimized_idx),
            ]
        )


@dataclass(frozen=True)
class CurvatureGlassLossFactory(_LossFactoryBase):
    loss_full: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    curvatures_template: jnp.ndarray
    distances_template: jnp.ndarray
    material_ids_template: jnp.ndarray
    curvatures_optimized_idx: jnp.ndarray
    materials_optimized_idx: jnp.ndarray
    catalogs: Sequence[ior_catalogs.RefractiveIndexCatalog]
    vmap_in_axes: ClassVar = (0, 0)

    def _reconstruct_params(
        self, curvatures_x: jnp.ndarray, material_ids: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        curvatures = _apply_updates(
            self.curvatures_template, self.curvatures_optimized_idx, curvatures_x
        )
        distances = self.distances_template
        all_material_ids = _apply_updates(
            self.material_ids_template,
            self.materials_optimized_idx,
            _round_material_ids(material_ids),
        )
        iors = _resolve_iors(all_material_ids, self.catalogs)
        return curvatures, distances, iors

    def _evaluate(
        self, curvatures_x: jnp.ndarray, material_ids: jnp.ndarray
    ) -> jnp.ndarray:
        return jnp.sum(
            self.loss_full(*self._reconstruct_params(curvatures_x, material_ids))
        )

    def _evaluate_full(
        self, curvatures_x: jnp.ndarray, material_ids: jnp.ndarray
    ) -> jnp.ndarray:
        return self.loss_full(*self._reconstruct_params(curvatures_x, material_ids))

    def init_from_templates(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return (
            _gather(self.curvatures_template, self.curvatures_optimized_idx),
            _gather(self.material_ids_template, self.materials_optimized_idx),
        )

    def unpack_full(
        self, curvatures_x: jnp.ndarray, material_ids: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self._reconstruct_params(curvatures_x, material_ids)

    def pack_from_full(
        self, curvatures, distances, material_ids
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return (
            _gather(curvatures, self.curvatures_optimized_idx),
            _gather(material_ids, self.materials_optimized_idx),
        )

    def pack_iors_from_packed_material_ids(
        self, material_ids: jnp.ndarray
    ) -> jnp.ndarray:
        all_material_ids = _apply_updates(
            self.material_ids_template,
            self.materials_optimized_idx,
            _round_material_ids(material_ids),
        )
        iors = _resolve_iors(all_material_ids, self.catalogs)
        return _gather(iors, self.materials_optimized_idx)


@dataclass(frozen=True)
class ThicknessGlassLossFactory(_LossFactoryBase):
    loss_full: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    curvatures_template: jnp.ndarray
    distances_template: jnp.ndarray
    material_ids_template: jnp.ndarray
    distances_optimized_idx: jnp.ndarray
    materials_optimized_idx: jnp.ndarray
    catalogs: Sequence[ior_catalogs.RefractiveIndexCatalog]
    vmap_in_axes: ClassVar = (0, 0)

    def _reconstruct_params(
        self, distances_x: jnp.ndarray, material_ids: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        curvatures = self.curvatures_template
        distances = _apply_updates(
            self.distances_template, self.distances_optimized_idx, distances_x
        )
        all_material_ids = _apply_updates(
            self.material_ids_template,
            self.materials_optimized_idx,
            _round_material_ids(material_ids),
        )
        iors = _resolve_iors(all_material_ids, self.catalogs)
        return curvatures, distances, iors

    def _evaluate(
        self, distances_x: jnp.ndarray, material_ids: jnp.ndarray
    ) -> jnp.ndarray:
        return jnp.sum(
            self.loss_full(*self._reconstruct_params(distances_x, material_ids))
        )

    def _evaluate_full(
        self, distances_x: jnp.ndarray, material_ids: jnp.ndarray
    ) -> jnp.ndarray:
        return self.loss_full(*self._reconstruct_params(distances_x, material_ids))

    def init_from_templates(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return (
            _gather(self.distances_template, self.distances_optimized_idx),
            _gather(self.material_ids_template, self.materials_optimized_idx),
        )

    def unpack_full(
        self, distances_x: jnp.ndarray, material_ids: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self._reconstruct_params(distances_x, material_ids)

    def pack_from_full(
        self, curvatures, distances, material_ids
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return (
            _gather(distances, self.distances_optimized_idx),
            _gather(material_ids, self.materials_optimized_idx),
        )

    def pack_iors_from_packed_material_ids(
        self, material_ids: jnp.ndarray
    ) -> jnp.ndarray:
        all_material_ids = _apply_updates(
            self.material_ids_template,
            self.materials_optimized_idx,
            _round_material_ids(material_ids),
        )
        iors = _resolve_iors(all_material_ids, self.catalogs)
        return _gather(iors, self.materials_optimized_idx)


class CurvatureThicknessGlassLossFactory(LossFactoryWithMaterials):
    """Alias with descriptive naming for backward compatibility."""

    pass


def create_lens_descriptor_bounds(
    curvatures_optimized_ids: jnp.ndarray,
    materials_optimized_ids: jnp.ndarray,
    catalogs_full: tuple[ior_catalogs.RefractiveIndexCatalog],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    sz = curvatures_optimized_ids.size + materials_optimized_ids.size
    lower = [0] * sz
    upper = [1] * sz
    lower[0] = 1
    for i in range(curvatures_optimized_ids.size, sz):
        packed_material_idx = i - curvatures_optimized_ids.size
        upper[i] = len(catalogs_full[materials_optimized_ids[packed_material_idx]]) - 1
    return jnp.array(lower), jnp.array(upper)


def unpack_materail_ids(
    material_ids_template: jnp.ndarray,
    materials_optimized_idx: jnp.ndarray,
    packed_material_ids: jnp.ndarray,
):
    return material_ids_template.at[materials_optimized_idx].set(
        jnp.round(packed_material_ids).astype(int)
    )


def compute_curvatures_bb_by_descriptor(
    descriptor: tuple[int, ...],
    len_optimized_curvatures: int,
):
    lo, hi = [], []
    for curvature_descriptor in descriptor[:len_optimized_curvatures]:
        if curvature_descriptor == 0:
            lo.append(-1000.0)
            hi.append(-4.0)
        else:
            lo.append(4.0)
            hi.append(1000.0)
    return jnp.array(lo), jnp.array(hi)


@dataclass(frozen=True)
class LossFactoryFixedDescriptor:
    loss_full: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    curvatures_template: jnp.ndarray
    distances_template: jnp.ndarray
    curvatures_optimized_idx: jnp.ndarray
    distances_optimized_idx: jnp.ndarray
    iors_full: jnp.ndarray
    materials_full: jnp.ndarray
    low_curvature_packed: jnp.ndarray
    up_curvature_packed: jnp.ndarray
    weight_bb_penalty: jnp.ndarray = 2.0

    @staticmethod
    def create(
        cnst: model.LensSystemConstants,
        curvatures_template: jnp.ndarray,
        distances_template: jnp.ndarray,
        material_ids_template: jnp.ndarray,
        curvatures_optimized_idx: jnp.ndarray,
        distances_optimized_idx: jnp.ndarray,
        materials_optimized_idx: jnp.ndarray,
        descriptor: tuple[int, ...],
    ):
        packed_material_ids = descriptor[curvatures_optimized_idx.size :]
        full_material_ids = material_ids_template.at[materials_optimized_idx].set(
            jnp.round(packed_material_ids).astype(int)
        )
        iors_full = ior_catalogs.resolve_iors(full_material_ids, cnst.ior_catalogs)
        lo, hi = compute_curvatures_bb_by_descriptor(
            descriptor,
            curvatures_optimized_idx.size,
        )
        return LossFactoryFixedDescriptor(
            loss_full=lambda curv_, dist_, iors_: loss(cnst, curv_, dist_, iors_),
            curvatures_template=curvatures_template,
            distances_template=distances_template,
            curvatures_optimized_idx=curvatures_optimized_idx,
            distances_optimized_idx=distances_optimized_idx,
            iors_full=iors_full,
            materials_full=full_material_ids,
            low_curvature_packed=lo,
            up_curvature_packed=hi,
            weight_bb_penalty=2.0,
        )

    def _reconstruct_params(
        self, x: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        curvatures_x, distances_x = _split_sections(
            x,
            (
                self.curvatures_optimized_idx.size,
                self.distances_optimized_idx.size,
            ),
        )
        curvatures = _apply_updates(
            self.curvatures_template, self.curvatures_optimized_idx, curvatures_x
        )
        distances = _apply_updates(
            self.distances_template, self.distances_optimized_idx, distances_x
        )
        return curvatures, distances, self.iors_full

    def _bb_penalty(self, x):
        curvature_packed = x[: self.curvatures_optimized_idx.size]
        under = jnp.clip(self.low_curvature_packed - curvature_packed, a_min=0.0)
        over = jnp.clip(curvature_packed - self.up_curvature_packed, a_min=0.0)
        return self.weight_bb_penalty * jnp.sum(under**2 + over**2)

    def _evaluate(self, x):
        return jnp.sum(self.loss_full(*self._reconstruct_params(x)))

    def _evaluate_with_bb_penalty(self, x):
        return self._evaluate(x) + self._bb_penalty(x)

    def make_jit_loss(self):
        return jax.jit(self._evaluate)

    def make_jit_loss_bb(self):
        return jax.jit(self._evaluate_with_bb_penalty)

    def make_jit_loss_full(self):
        return jax.jit(lambda x_: self.loss_full(*self._reconstruct_params(x_)))

    def init_from_templates(self) -> jnp.ndarray:
        return _concat_parts(
            [
                _gather(self.curvatures_template, self.curvatures_optimized_idx),
                _gather(self.distances_template, self.distances_optimized_idx),
            ]
        )

    def unpack_full(self, x) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self._reconstruct_params(x)

    def pack_from_full(self, curvatures, distances) -> jnp.ndarray:
        return _concat_parts(
            [
                _gather(curvatures, self.curvatures_optimized_idx),
                _gather(distances, self.distances_optimized_idx),
            ]
        )
