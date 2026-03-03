import math
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Sequence

import jax.numpy as jnp
import numpy as np
import pandas as pd

import lensgopt.optics.meta as meta


class RefractiveIndexCatalog(ABC):

    def __init__(self, wavelengths: tuple):
        self.wavelengths = jnp.array(wavelengths)
        if len(self.wavelengths) < 1 or not math.isclose(
            self.wavelengths[0], meta.LAMBDA_D, abs_tol=1e-4
        ):
            raise ValueError(
                f"The first element of wavelengths must be the standard D-line in nm, but it is {self.wavelengths[0]:.10f} [nm]"
            )

    @abstractmethod
    def precompute_iors(self):
        """
        Precompute refractive indices for all materials at specified wavelengths.

        Returns:
            dict[str, jnp.ndarray]: Mapping the wavelengths number to a
                1D JAX array of refractive indices (dtype float64) for all materials.
        """
        pass

    @abstractmethod
    def ior_by_id(self, material_id: int):
        """
        Retrieve refractive indices for a single material by its ID across wavelengths.

        Ensures precomputed values exist for the requested wavelengths and then
        extracts the refractive index for the given material index.

        Args:
            material_id (int): Index of the material in the catalog (order from self.names).

        Returns:
            jnp.ndarray: 1D array (dtype float64) of refractive indices at each wavelength.
        """
        pass

    @abstractmethod
    def material_id_by_material_name(self, material_name: str):
        """
        Get the material ID (index) for a given material name, case-insensitively.

        Args:
            name (str): Name of the material (e.g., 'vacuum', 'AIR').

        Returns:
            int: Index of the material in the catalog.

        Raises:
            KeyError: If the material name is not found in the catalog.
        """
        pass


class SellmeierGlassCatalog(RefractiveIndexCatalog):
    """
    Abstract base class representing an optical glass catalog.

    Provides a common interface for loading, querying, and retrieving glass properties.
    """

    def __init__(self, wavelengths: tuple, df: pd.DataFrame):
        """
        Initialize the catalog with a DataFrame, validating required columns.

        """
        super().__init__(wavelengths=wavelengths)

        self.target_columns = ["Glass", "nd", "vd", "B1", "B2", "B3", "C1", "C2", "C3"]
        if not set(self.target_columns).issubset(df.columns):
            raise ValueError(
                f"Target columns {self.target_columns} are not in the DataFrame columns: {df.columns}"
            )
        self.df = df.sort_values(by=["nd"]).reset_index(drop=True)

    def _ior_sellmeier(
        self,
        lambda_nm: float,
        B1: float,
        B2: float,
        B3: float,
        C1: float,
        C2: float,
        C3: float,
    ) -> float:
        """
        Compute refractive index for a single glass at one wavelength using Sellmeier equation.

        Args:
            lambda_nm (float): Wavelength in nanometers.
            B1, B2, B3, C1, C2, C3 (float): Sellmeier coefficients.

        Returns:
            float: Refractive index at the given wavelength.
        """
        l2 = (lambda_nm / 1000.0) ** 2
        return np.sqrt(1.0 + l2 * (B1 / (l2 - C1) + B2 / (l2 - C2) + B3 / (l2 - C3)))

    def precompute_iors(self) -> dict[str, jnp.ndarray]:
        # Retrieve Sellmeier constants for all materials: shape (n_materials, 6)
        constants = self.get_all_sellmeier_constants()
        # Split into B coefficients and C coefficients
        B = constants[:, :3]  # shape (n_materials, 3)
        C = constants[:, 3:]  # shape (n_materials, 3)

        # Square of normalized wavelength: (λ/1000)^2, shape (n_wavelengths,)
        l2 = (jnp.array(self.wavelengths) / 1000.0) ** 2  # jnp.ndarray

        # Broadcast to calculate denominator for each wavelength-material pair
        # denom: shape (n_wavelengths, n_materials, 3)
        denom = l2[:, None, None] - C[None, :, :]

        # Compute Sellmeier terms for each pair, then sum over coefficients
        # terms: B / denom, shape (n_wavelengths, n_materials, 3)
        terms = B[None, :, :] / denom
        # Sum across the 3 Sellmeier terms: shape (n_wavelengths, n_materials)
        sum_terms = jnp.sum(terms, axis=-1)

        # Compute refractive index matrix
        # iors_mat[w, m] = sqrt(1 + l2[w] * sum_terms[w, m])
        iors_mat = jnp.sqrt(1.0 + l2[:, None] * sum_terms)

        # For material store its iors by wavelenghts
        self.precomputed_iors = iors_mat.T
        return self.precomputed_iors

    def ior_by_id(self, material_id: int) -> jnp.ndarray:
        # Ensure precomputed indices exist and match requested wavelengths
        # if not hasattr(self, "precomputed_iors") or set(
        #     self.precomputed_iors.keys()
        # ) != {f"{float(w):.5f}" for w in wavelengths}:
        if not hasattr(self, "precomputed_iors"):
            # Recompute if missing or stale
            self.precompute_iors()

        # Extract index values for each wavelength key
        return self.precomputed_iors[material_id]

    def material_id_by_material_name(self, name: str):
        id_ = self.get_index_by_glass_name(name)
        if id_ is None:
            raise KeyError(
                f"Material name '{name}' not found in {self.get_catalog_name()} catalog."
            )
        return id_

    def has_glass(self, name: str) -> bool:
        """
        Check if a glass with the given name exists in the catalog.

        Args:
            name (str): Name of the glass to look up.

        Returns:
            bool: True if the glass is present, False otherwise.
        """
        return name in self.df["Glass"].values

    def get_index_by_glass_name(self, name: str) -> int | None:
        """
        Get the index of a glass by its name.

        Args:
            name (str): Name of the glass.

        Returns:
            int | None: Index of the glass if found; otherwise None.
        """
        out = self.df.loc[self.df["Glass"] == name]
        return int(out.index[0]) if not out.empty else None

    def get_glass_name_by_index(self, index: int) -> str:
        """
        Retrieve the glass name at the specified index.

        Args:
            index (int): Row index in the catalog DataFrame.

        Returns:
            str: Name of the glass.
        """
        return self.df.loc[int(index), "Glass"]

    def get_sellmeier_constants_by_index(self, index: int) -> np.ndarray:
        """
        Get the Sellmeier constants (B1 to C3) for a glass by its index.

        Args:
            index (int): Row index in the catalog DataFrame.

        Returns:
            np.ndarray: Array of Sellmeier constants [B1, B2, B3, C1, C2, C3].
        """
        return self.df.loc[index, "B1":"C3"].values

    def get_all_sellmeier_constants(self) -> np.ndarray:
        """
        Get the Sellmeier constants (B1 to C3) for all the glass in the catalog.

        Returns:
            np.ndarray: Array of Sellmeier constants [B1, B2, B3, C1, C2, C3].
        """
        return self.df.loc[0 : len(self.df) - 1, "B1":"C3"].values

    def get_catalog_name(self) -> str:
        """
        Return the human-readable name of the catalog.

        Returns:
            str: Catalog name.
        """
        return ""

    def __len__(self) -> int:
        """
        Return the number of glasses in the catalog.

        Returns:
            int: Number of rows in the catalog DataFrame.
        """
        return len(self.df)


class Schott(SellmeierGlassCatalog):
    """
    Concrete class loading the Schott optical glass catalog.

    The catalog is loaded from a local Excel file or fallback online source.

    Locally stored copy was most recently updated on 16 September 2024 and downloaded on 6 March 2025.
    """

    def __init__(self, wavelengths: tuple):
        """
        Initialize the Schott catalog by reading and preprocessing the data.
        """
        df = Schott.load_catalog()
        super().__init__(wavelengths, df)

    def get_catalog_name(self) -> str:
        """
        Return the catalog name identifier.

        Returns:
            str: "Schott".
        """
        return "Schott"

    @staticmethod
    def __read_excel(filename) -> pd.DataFrame:
        """
        Read the Schott catalog Excel file into a DataFrame.

        Args:
            filename (str | Path): Path or URL to the Excel file.

        Returns:
            pd.DataFrame: Raw DataFrame extracted from the "Preferred glasses" sheet.
        """
        return pd.read_excel(
            filename, sheet_name=r"Preferred glasses", header=3, skipfooter=1
        )

    @staticmethod
    def load_catalog() -> pd.DataFrame:
        """
        Load and filter the Schott catalog, trying local file first then online fallback.

        Returns:
            pd.DataFrame: Cleaned catalog with only necessary columns.

        Raises:
            FileNotFoundError: If neither local nor online sources can be loaded.
        """
        df = None
        try:
            HERE = Path(__file__).resolve().parent
            filename = (
                HERE
                / "databases"
                / "schott-optical-glass-overview-excel-format-en.xlsx"
            )
            df = Schott.__read_excel(filename)
        except Exception:
            print("Glass catalog in xlsx format is not loaded locally.")
        if df is None:
            try:
                url = (
                    "https://mss-p-009-delivery.stylelabs.cloud/api/public/content/"
                    "7f5a58f7fa754445abab44cafc76acb3?v=a102056c&download=true"
                )
                df = Schott.__read_excel(url)
            except Exception:
                print("Glass catalog is not downloaded from online source.")
        if df is None:
            raise FileNotFoundError("Glass catalog not found locally or online.")
        cols = ["Glass", "nd", "vd"] + list(
            df.columns[
                df.columns.tolist().index("B1") : df.columns.tolist().index("C3") + 1
            ]
        )
        return df.loc[:, cols]

    def material_name_by_id(self, material_id: int) -> str:
        return self.get_glass_name_by_index(material_id)


class CauchyMaterialCatalog(RefractiveIndexCatalog):
    """
    A catalog that uses the Cauchy approximation (n(λ) = A + B/λ²)
    to compute refractive indices given nD (at λD) and Abbe number V.

    Here:
        B = (nD - 1) / [V·(1/λF² - 1/λC²)],
        A = nD - B/λD².
    """

    def __init__(self, wavelengths: tuple):
        super().__init__(wavelengths=wavelengths)
        self._A_list: List[float] = []
        self._B_list: List[float] = []
        self._names: List[str] = []
        self._name_to_id: Dict[str, int] = {}

    def add_material(self, nD: float, V: float, name=None) -> str:
        """
        Add a new material defined by its refractive index nD (at λD) and Abbe number V.

        Cauchy coefficients are computed as:
            B = (nD - 1) / [ V·(1/λF² - 1/λC²) ]
            A = nD - B/λD²

        Args:
            nD (float): Refractive index at D-line (λD = meta.LAMBDA_D).
            V  (float): Abbe number.
            name (str|None): Optional user-provided name. If None, use str(id).

        Returns:
            str: The assigned material name (string).
        """
        # Fraunhofer wavelengths in nm
        λD = float(meta.LAMBDA_D)
        λF = float(meta.LAMBDA_F)
        λC = float(meta.LAMBDA_C)

        # Compute B from nD and V
        numerator = nD - 1.0
        denom = V * (1.0 / (λF * λF) - 1.0 / (λC * λC))
        B = numerator / denom

        # Compute A so that n(λD) = nD = A + B/λD²
        A = nD - (B / (λD * λD))

        # Assign new ID and name
        new_id = len(self._A_list)
        assigned_name = name if (name is not None) else str(new_id)

        # Store coefficients and name
        self._A_list.append(A)
        self._B_list.append(B)
        self._names.append(assigned_name)
        self._name_to_id[assigned_name] = new_id

        return assigned_name

    def precompute_iors(self):
        """
        Precompute refractive indices for all materials at each wavelength λ
        using n(λ) = A + B/λ². Stores in self.precomputed_iors mapping.
        """
        n_mat = len(self._A_list)
        A_arr = jnp.broadcast_to(
            jnp.array(self._A_list)[..., None],
            (len(self._A_list), len(self.wavelengths)),
        )  # shape (n_mat, n_wl)
        B_arr = jnp.broadcast_to(
            jnp.array(self._B_list)[..., None],
            (len(self._B_list), len(self.wavelengths)),
        )  # shape (n_mat, n_wl)

        # Clear previous
        self.precomputed_iors = {}

        self.precomputed_iors = A_arr + B_arr / self.wavelengths**2

        return self.precomputed_iors

    def ior_by_id(self, material_id: int) -> jnp.ndarray:
        """
        Return refractive index for a single material at each wavelength.
        If precomputed_iors is missing or keys mismatch, recompute.

        Args:
            material_id (int): Index of the material.

        Returns:
            jnp.ndarray: 1D array of refractive indices (length = len(wavelengths)).
        """
        if not hasattr(self, "precomputed_iors"):
            self.precompute_iors()

        # Gather for each wavelength
        return self.precomputed_iors[material_id]

    def material_id_by_material_name(self, material_name: str) -> int:
        """
        Return the integer ID for a given material name.

        Args:
            material_name (str): Name of added material.

        Raises:
            KeyError: if name not found.

        Returns:
            int: Material ID.
        """
        if material_name.lower() not in self._name_to_id:
            raise KeyError(f"Material '{material_name}' not found.")
        return self._name_to_id[material_name.lower()]

    def material_name_by_id(self, material_id: int) -> str:
        for k, v in self._name_to_id.items():
            if v == material_id:
                return k
        raise KeyError(f"No material with id {material_id}")

    def get_catalog_name(self) -> str:
        """
        Return the catalog name identifier.
        """
        return ""

    def __len__(self) -> int:
        """
        Return the number of glasses in the catalog.

        Returns:
            int: Number of rows in the catalog DataFrame.
        """
        return 1


class Gaze(CauchyMaterialCatalog):
    """
    Catalog of simple gaze/environment materials with constant refractive indices.

    Materials included:
        - vacuum: nD = 1.0
        - air:    nD = 1.000293
    """

    def __init__(self, wavelengths: tuple):
        """
        Initialize the gaze catalog, preserving material order and settings.
        """
        # Define base materials with [nd, Abbe_number]
        super().__init__(wavelengths=wavelengths)
        self.materials = {
            "vacuum": [1.0, np.inf],
            "air": [1.000293, np.inf],
        }
        for material_name, nD_V in self.materials.items():
            nD, V = nD_V
            super().add_material(nD, V, name=material_name)

    def get_catalog_name(self) -> str:
        """
        Return the catalog name identifier.
        """
        return ""


def resolve_material_ids(
    material_names: Sequence[str], catalogs: Sequence[RefractiveIndexCatalog]
):
    material_ids = []
    for i, name_ in enumerate(material_names):
        if "/" in name_:
            material_id = 0
            material_ids.append(material_id)
            continue
        elif ":" in name_:
            name, _ = name_.split(":")
        else:
            name = name_
        material_id = catalogs[i].material_id_by_material_name(name)
        material_ids.append(material_id)
    return jnp.array(material_ids)


def resolve_iors(material_ids: jnp.ndarray, catalogs: Sequence[RefractiveIndexCatalog]):
    iors = []
    for i, material_id in enumerate(material_ids):
        ior_values = catalogs[i].ior_by_id(material_id)
        iors.append(ior_values)
    return jnp.array(iors)


def resolve_catalogs_and_iors(material_names, wls):
    ctlgs = []
    iors = []
    for material in material_names:
        if "/" in material:
            nD, V = map(float, material.split("/"))
            ctlg = CauchyMaterialCatalog(wls)
            name = ctlg.add_material(nD, V)
        elif ":" in material:
            name, catalog_name = material.split(":")
            module = sys.modules[__name__]
            try:
                cls = getattr(module, catalog_name)
            except AttributeError:
                raise ValueError(f"No such catalog {catalog_name!r}")
            ctlg = cls(wls)
        else:
            ctlg = Gaze(wls)
            name = material.lower()
        id = ctlg.material_id_by_material_name(name)
        ior = ctlg.ior_by_id(id)
        ctlgs.append(ctlg)
        iors.append(ior)
    return tuple(ctlgs), jnp.array(iors)
