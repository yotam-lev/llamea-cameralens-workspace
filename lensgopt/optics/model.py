from typing import NamedTuple

import jax.numpy as jnp

import lensgopt.materials.refractive_index_catalogs as ior_catalogs
import lensgopt.optics.meta as meta


class Aperture(NamedTuple):
    """
    Defines the aperture stop configuration for an optical system.

    The aperture controls the cone of light that can pass through the system,
    influencing illumination, depth of field, and aberrations.

    """

    type: str = "ED"  # options: 'ED', 'float', 'objectNA', 'fNum', 'parFNum'
    max_d: float = None  # [mm]


class Field(NamedTuple):
    """
    Defines the object field configuration for an optical system.

    The field specification determines how rays originate from the object
    space, which can be angular or spatial, affecting image height and field curvature.

    """

    type: str = "angle"  # options: 'angle', 'objectHeight'
    max_field: float = None  # angle in degrees, height in mm


class LensSystemConstants(NamedTuple):
    """
    Holds all fixed design parameters and target specifications for an optical lens system.

    This dataclass encapsulates:
      - Definitions of the field and aperture.
      - Number and types of surfaces in the lens train.
      - Material catalogs for each surface.
      - Target (desired) values for performance metrics.
      - Wavelengths and field weighting factors.
      - Paraxial thresholds and mode for computing sensor position.

    """

    # Field and aperture definitions
    lens_field: Field
    aperture: Aperture

    # Number of surfaces and parameters per surface
    num_surfaces: int
    num_parameters_per_surface: tuple  # shape = (num_surfaces,)
    surface_types: tuple[str, ...]  # shape = (num_surfaces,) (Spheric, Aspheric, ...)

    # Refractive index catalogs assigned to each surface
    ior_catalogs: tuple[ior_catalogs.RefractiveIndexCatalog, ...]
    aperture_stop_index: int

    # Target performance values
    target_effective_focal_length: float  # [mm]
    target_edge_thickness: float  # [mm]
    target_axis_thickness: float  # [mm]
    target_free_working_distance: float  # [mm]

    # Wavelengths (first element must be the D-line)
    wavelengths: tuple = tuple(
        [
            meta.LAMBDA_D,  # Must always be first (Fraunhofer D-line)
            meta.LAMBDA_C,
            meta.LAMBDA_F,
        ]
    )

    # Weight factors for different field points
    field_factors: tuple = tuple([0.0, 0.7, 1.0])
    edge_thickness_field_factors: tuple = tuple([-1, 0, 1])
    edge_thickness_entrance_pupil_factors: tuple = tuple([-1, 1])

    # Paraxial threshold constants
    MAX_PARAXIAL_ANGLE: float = 1e-3
    EFFL_PARAXIAL_Y: float = 1e-6

    # Object and sensor z coordinates
    object_z: float = float("-inf")
    sensor_z: float = 0.0

    # Mode for computing sensor_z: 'fixed', 'optimize' or 'paraxial_solve'
    sensor_z_mode: str = "paraxial_solve"

    # Mode for computing gradients w.r.t. iors
    is_iors_grad: bool = False

    # Number of rays in one row of square pattern
    n_rays_row: int = 31

    def __post_init__(self):
        # Ensure the first wavelength is meta.LAMBDA_D (D-line)
        if len(self.wavelengths) < 1 or self.wavelengths[0] != meta.LAMBDA_D:
            raise ValueError("The first element of wavelengths must be meta.LAMBDA_D")

        # Validate array lengths match the declared number of surfaces
        if len(self.num_parameters_per_surface) != self.num_surfaces:
            raise ValueError("num_parameters_per_surface must have length num_surfaces")
        if len(self.surface_types) != self.num_surfaces:
            raise ValueError("surface_type_id must have length num_surfaces")
        if len(self.ior_catalogs) != self.num_surfaces + 1:
            raise ValueError("len(catalogs) must equal num_surfaces + 1")

        # Additional validation for target values
        if self.target_effective_focal_length <= 0:
            raise ValueError("target_effective_focal_length must be > 0")
        if self.target_edge_thickness < 0:
            raise ValueError("target_edge_thickness must be >= 0")

        # aperture_stop_index must be within valid range
        if not (0 <= self.aperture_stop_index < self.num_surfaces):
            raise ValueError("aperture_stop_index must be in [0, num_surfaces)")

        # sensor_mode must be either 'optimize' or 'paraxial_solve'
        if self.sensor_z_mode not in ("fixed", "optimize", "paraxial_solve"):
            raise ValueError(
                "sensor_mode must be 'fixed', 'optimize' or 'paraxial_solve'"
            )


class LensSystemVariables(NamedTuple):
    curvature_parameters: jnp.ndarray
    distances_z: jnp.ndarray
    iors: jnp.ndarray

    @staticmethod
    def resolve_verticies_z(distances_z):
        zero = jnp.zeros((1,))
        prefs = jnp.cumsum(distances_z)
        return jnp.concatenate([zero, prefs], axis=0)


class LensSystemComputedProperties(NamedTuple):
    iors: jnp.ndarray
    entrance_pupil_z: float
    entrance_pupil_diameter: float
    effective_focal_length: float
    paraxial_marginal_ray: jnp.ndarray  # [y, alpha]
    vertices_z: jnp.ndarray
    limiting_vertices_z: jnp.ndarray
    up_corners_y: jnp.ndarray
    sensor_z: float
