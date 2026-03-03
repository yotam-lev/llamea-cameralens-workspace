import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class RotSymSurface(ABC):
    vertex_z: jnp.ndarray

    @abstractmethod
    def ray_surface_intersection(self, ray, surface):
        pass

    @abstractmethod
    def normal(self, x, y, z):
        pass

    @abstractmethod
    def flatten_param(self) -> jnp.array:
        pass

    @abstractmethod
    def find_opposite_vertex(self) -> jnp.ndarray:
        pass

    @classmethod
    @abstractmethod
    def get_curvature_param_count(cls) -> int:
        pass

    @classmethod
    @abstractmethod
    def get_paraxial_radius(cls, flattened_params) -> float:
        pass

    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)


@dataclass(frozen=True)
class Spheric(RotSymSurface):
    r: jnp.ndarray

    @classmethod
    def create(cls, vertex_z: float, r: jnp.ndarray):
        return cls(vertex_z=vertex_z, r=r[0])

    def ray_surface_intersection(self, ray, limiting_vertex_z: jnp.ndarray):
        """
        Compute intersection of rays with this spherical (or planar) surface.
        Returns a boolean mask and the intersection points [...,3].
        """

        eps: float = 1e-8

        def planar_case(_):
            alpha = (-ray.o[..., 2] + self.vertex_z) / ray.d[..., 2]
            sol = ray.o + alpha[..., None] * ray.d
            valid = ~jnp.isnan(alpha)
            return valid, sol

        def sphere_case(_):
            d = self.vertex_z
            center = jnp.array([0.0, 0.0, d + self.r])
            e = ray.o - center

            inner_ed = jnp.sum(e * ray.d, axis=-1)
            ne2 = jnp.sum(e * e, axis=-1)
            r2 = self.r**2
            D = inner_ed**2 - (ne2 - r2)

            valid = D > 1e-8  # rays with real intersection
            sqrtD = jnp.sqrt(jnp.where(valid, D, 1.0))  # safe dummy value for AD

            # Compute alpha for converging (r > 0) or diverging (r < 0) surfaces
            alpha_raw = jnp.where(
                self.r > 0, -inner_ed - sqrtD, -inner_ed + sqrtD
            )  # omit division by nd2 = 1

            alpha = jnp.where(valid, alpha_raw, 0.0)  # mask invalid rays
            sol = ray.o + alpha[..., None] * ray.d
            sol = jnp.where(
                valid[..., None], sol, 0.0
            )  # mask invalid intersection points

            # Check z-bound validity
            z = sol[..., 2]
            lo = jnp.minimum(d, limiting_vertex_z) - eps
            hi = jnp.maximum(d, limiting_vertex_z) + eps
            valid &= (z >= lo) & (z <= hi)  # combine with existing mask

            sol = jnp.where(valid[..., None], sol, 0.0)  # reapply z-bound masking
            return valid, sol

        is_plane = jnp.isinf(self.r)
        valid, sol = jax.lax.cond(is_plane, planar_case, sphere_case, operand=None)

        return valid, sol

    def normal(self, x0, y0, z0):
        # We guarantee that (x,y,z) belongs to the sphere to speed up the computations
        # Kidger, Eq. (3.30)
        return jnp.stack(
            [
                -x0 / self.r,
                -y0 / self.r,
                1 + (self.vertex_z - z0) / self.r,
            ],
            axis=-1,
        )

    @classmethod
    def get_curvature_param_count(cls) -> int:
        return 1

    @classmethod
    def get_paraxial_radius(cls, flattened_params: jnp.ndarray) -> float:
        return flattened_params[0]

    def flatten_param(self) -> jnp.array:
        return jnp.array([self.r])

    def find_opposite_vertex(self) -> jnp.ndarray:
        return jnp.where(jnp.isinf(self.r), self.vertex_z, self.vertex_z + self.r)


def get_surface_class_by_name(class_name: str):
    module = sys.modules[__name__]
    try:
        cls = getattr(module, class_name)
    except AttributeError:
        raise ValueError(f"No such class {class_name!r}")
    return cls


def get_curvature_param_count_via_interface(class_name: str) -> int:
    cls = get_surface_class_by_name(class_name)
    return cls.get_curvature_param_count()


def create_surface_by_name(
    class_name: str, vertice_z: jnp.ndarray, params: jnp.ndarray
) -> RotSymSurface:
    cls = get_surface_class_by_name(class_name)
    return cls.create(vertice_z, params)


def flatten_surfaces(surfaces: tuple[RotSymSurface, ...]):
    num_parameters_per_surface = []
    surface_types = []
    curvature_parameters = []
    distances_z = []
    prv_z = 0.0
    for s in surfaces:
        surface_types.append(s.__class__.__name__)
        params = s.flatten_param()
        num_parameters_per_surface.append(len(params))
        curvature_parameters.append(params)
        distances_z.append(s.vertex_z - prv_z)
        prv_z = s.vertex_z
    distances_z.pop(0)
    return (
        tuple(num_parameters_per_surface),
        tuple(surface_types),
        jnp.concat(curvature_parameters, axis=0),
        jnp.array(distances_z),
    )
