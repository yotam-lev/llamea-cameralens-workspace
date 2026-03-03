import json
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

import lensgopt.optics.model as model
import lensgopt.optics.shapes as shapes


class LensFromZMAXFormat(NamedTuple):
    lens_aperture: model.Aperture
    lens_field: model.Field
    surfaces: tuple[shapes.RotSymSurface, ...]
    thicknesses: tuple[float, ...]
    object_distance: float
    sensor_distance: float
    aperture_stop_index: int
    material_names: tuple[str, ...]


def parse_field_line(line: str) -> model.Field:
    prefix = "Field:"
    if not line.startswith(prefix):
        raise ValueError(f"Line must start with '{prefix}'")
    # Extract substring after "Field:"
    json_part = line[len(prefix) :].strip()
    try:
        data: Any = json.loads(json_part)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")

    # Validate presence of required keys
    if "type" not in data or "MAX_FIELD" not in data:
        raise ValueError("JSON must contain keys 'type' and 'MAX_FIELD'")

    f_type = data["type"]
    max_f = data["MAX_FIELD"]

    # Ensure types are correct
    if not isinstance(f_type, str):
        raise ValueError(f"'type' must be a string, got {type(f_type).__name__}")
    try:
        max_f = float(max_f)
    except (TypeError, ValueError):
        raise ValueError(
            f"'MAX_FIELD' must be convertible to float, got {data['MAX_FIELD']}"
        )

    return model.Field(type=f_type, max_field=max_f)


def parse_aperture_line(line: str) -> model.Aperture:
    prefix = "Aperture:"
    if not line.startswith(prefix):
        raise ValueError(f"Line must start with '{prefix}'")
    # Extract substring after "Aperture:"
    json_part = line[len(prefix) :].strip()
    try:
        data: Any = json.loads(json_part)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")

    # Validate presence of required keys
    if "type" not in data or "MAX_D" not in data:
        raise ValueError("JSON must contain keys 'type' and 'MAX_D'")

    ap_type = data["type"]
    max_d = data["MAX_D"]

    # Ensure types are correct
    if not isinstance(ap_type, str):
        raise ValueError(f"'type' must be a string, got {type(ap_type).__name__}")
    try:
        max_d = float(max_d)
    except (TypeError, ValueError):
        raise ValueError(f"'MAX_D' must be convertible to float, got {data['MAX_D']}")

    return model.Aperture(type=ap_type, max_d=max_d)


def parse_surface(
    line_no, surface_type, vertex_z, flat_str_params
) -> shapes.RotSymSurface:
    if surface_type == "S" or surface_type == "A":
        try:
            roc_val = float(flat_str_params[0])
        except ValueError:
            raise ValueError(
                f"Invalid spheric curvature on line {line_no+1}: '{flat_str_params}'"
            )
        return shapes.Spheric.create(vertex_z, jnp.array([roc_val]))
    raise ValueError(f"Surface type {surface_type} is not supported")


def parse_lens_zmax_format(
    filename: str,
) -> LensFromZMAXFormat:
    """
    Parse a Zemax-exported “ZMAX”-style lens file into a LensSystem.

    The file format is expected to be:
        Line 0: (ignored comment/title)
        Line 1: Field: {"type": "...", "MAX_FIELD": ...}
        Line 2: Aperture: {"type": "...", "MAX_D": ...}
        Line 3: column headers (ignored)
        Lines 4+: surface definitions, one per line, with columns:
            type    distance    roc    material

    Surface “type” may be:
      - 'O' = object plane  (distance resets to zero)
      - 'S' = normal spherical surface
      - 'A' = aperture stop surface
      - 'I' = image/sensor plane

    Args:
        filename: Path to the Zemax-format text file.

    Returns:
        LensFromZMAXFormat: Immutable class with parsed field, aperture,
            surfaces, materials, catalogs, curvatures and distances.

    Raises:
        ValueError: if required lines are missing or malformed.
        IOError: if the file cannot be read.
    """
    lens_aperture: model.Aperture = None
    lens_field: model.Field = None
    surfaces: tuple[shapes.RotSymSurface, ...] = []
    thicknesses: tuple[float, ...] = []
    object_distance: float = None
    sensor_distance: float = None
    aperture_stop_index: int = None
    material_names: tuple[str, ...] = []

    # Running total of distance along z
    z_total = 0.0

    with open(filename, "r") as f:
        # Read first three lines explicitly
        try:
            _title_line = next(f)  # line 0: ignored
            field_line = next(f)  # line 1: contains JSON for Field
            aperture_line = next(f)  # line 2: contains JSON for Aperture
            _header_line = next(f)  # line 3: column headers, ignore
        except StopIteration:
            raise ValueError(
                f"File '{filename}' must have at least 4 lines for title, Field, Aperture, header."
            )

        # Parse Field
        try:
            lens_field = parse_field_line(field_line.strip())
        except Exception as e:
            raise ValueError(f"Error parsing Field on line 2: {e}")

        # Parse Aperture
        try:
            lens_aperture = parse_aperture_line(aperture_line.strip())
        except Exception as e:
            raise ValueError(f"Error parsing Aperture on line 3: {e}")

        is_first_surface = False
        # Now process each subsequent surface line
        for line_no, raw in enumerate(f, start=4):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue  # skip blank/comment lines

            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Malformed surface line {line_no+1}: '{line}'")

            surf_type = parts[0]
            try:
                distance = float(parts[1])
            except ValueError:
                raise ValueError(f"Invalid distance on line {line_no+1}: '{parts[1]}'")

            material_str = parts[-1]

            # Increment total z by distance unless this is an image plane ('I')
            if surf_type == "O":
                # 'O' = object plane: reset running z_total to zero
                z_total = 0.0
                is_first_surface = True
                material_names.append(material_str)
                continue

            if is_first_surface and np.isinf(distance):
                z_total = 0.0
            else:
                z_total += distance

            if is_first_surface:
                object_distance = distance
            else:
                thicknesses.append(distance)

            is_first_surface = False

            if surf_type == "I":
                sensor_distance = distance
                continue

            if surf_type == "A":
                aperture_stop_index = len(surfaces)

            surface = parse_surface(line_no, surf_type, z_total, parts[2:-1])
            surfaces.append(surface)
            material_names.append(material_str)

    # Final sanity checks
    if lens_field is None:
        raise ValueError("Missing Field specification in input file.")
    if lens_aperture is None:
        raise ValueError("Missing Aperture specification in input file.")
    if not surfaces:
        raise ValueError("No surfaces parsed from input file.")

    return LensFromZMAXFormat(
        lens_aperture=lens_aperture,
        lens_field=lens_field,
        surfaces=surfaces,
        thicknesses=thicknesses,
        object_distance=object_distance,
        sensor_distance=sensor_distance,
        aperture_stop_index=aperture_stop_index,
        material_names=material_names,
    )
