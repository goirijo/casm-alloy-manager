import casmutils as cu
import json
import os
import numpy as np


class ProjectFilesystem(object):
    """Interface for retreiving paths related to a casm project,
    such as CONTCAR files for calculations for a particular supercell
    and configuration."""

    def __init__(self, root, name=None, calctype="default"):
        """TODO: to be defined.

        Parameters
        ----------
        root : path to casm project
        name : name for casm project, optional
        calctype : name for calculation type, optional


        """
        self._root = root

        if name is None:
            name = os.path.basename(root)

        self._name = name

        self.calctype = calctype

    @property
    def name(self):
        return self._name

    @property
    def root(self):
        return self._root

    def prim(self):
        """Return path to prim.json file
        Returns
        -------
        path

        """
        return os.path.join(self.root, "prim.json")

    def configuration(self, configname):
        """Return path to the specified configuration, given
        as "SCELX_A_B_C_D_E_F/Z"

        Parameters
        ----------
        configname : str

        Returns
        -------
        path

        """
        scel, config = configname.split("/")
        return os.path.join(self.root, "training_data", scel, config)

    def pos(self, configname):
        """Return path to POS (ideal POSCAR) of
        the specified configuration

        Parameters
        ----------
        configname : str

        Returns
        -------
        path

        """
        return os.path.join(self.configuration(configname), "POS")

    def calc_properties(self, configname):
        """Returns path to the properties.calc.json file of the
        specified configuration, where the configname is given in the
        form "SCELX_A_B_C_D_E_F/Z"

        Parameters
        ----------
        configname : str

        Returns
        -------
        path

        """
        scel, config = configname.split("/")
        return os.path.join(
            self.configuration(configname), "calctype.{}".format(self.calctype),
            "properties.calc.json")


def make_lattice(properties):
    """Construct cu.xtal.Lattice object from
    a calculated properties dictionary, as created
    by casm after DFT calculations.

    Parameters
    ----------
    properties : dict

    Returns
    -------
    cu.xtal.Lattice

    """
    lat_mat = np.array(properties["relaxed_lattice"])
    return cu.xtal.Lattice(*lat_mat)


def make_basis(properties):
    """Construct list(cu.xtal.Site) from a calculated properties
    dictionary, as created by casm after DFT calculations.

    Parameters
    ----------
    properties : dict

    Returns
    -------
    list(cu.xtal.Site)

    """
    lat = make_lattice(properties)
    types = properties["atom_type"]
    types_count = properties["atoms_per_type"]
    frac_basis = properties["relaxed_basis"]

    types_roll = [
        x for tt in [c * [t] for c, t in zip(types_count, types)] for x in tt
    ]

    return [
        cu.xtal.Site(cu.xtal.Coordinate.from_fractional(frac, lat), t)
        for frac, t in zip(frac_basis, types_roll)
    ]


def make_structure(properties):
    """Construct a cu.xtal.Structre object from
    a claculated properties dictionary, as created
    by casm after DFT calculation.

    Parameters
    ----------
    properties : dict

    Returns
    -------
    cu.xtal.Structure

    """
    lat = make_lattice(properties)
    basis = make_basis(properties)
    return cu.xtal.Structure(lat, basis)


def read_structure(properties_file):
    """Given a properties.calc.json file, read
    the data and create a cu.xtal.Structure from it

    Parameters
    ----------
    properties_file : path

    Returns
    -------
    cu.xtal.Structure

    """
    with open(properties_file) as json_data:
        d = json.load(json_data)
    return make_structure(d)


def make_allowed_species(prim):
    """Construct list(list(str)) where each row is a list
    of occupants for one site, as specified in the prim

    Parameters
    ----------
    prim : dict

    Returns
    -------
    list(list(str))

    """
    return [site["occupant_dof"] for site in prim["basis"]]


def make_prim(prim):
    """Construct cu.xtal.Structure using the first occupants
    specified in the prim file

    Parameters
    ----------
    prim : dict

    Returns
    -------
    cu.xtal.Structure

    """
    lat = cu.xtal.Lattice(*prim["lattice_vectors"])
    species = [site["occupant_dof"][0] for site in prim["basis"]]

    if prim["coordinate_mode"][0] in "fF":
        coords = [
            cu.xtal.Coordinate.from_fractional(site["coordinate"], lat)
            for site in prim["basis"]
        ]

    else:
        coords = [
            cu.xtal.Coordinate(site["coordinate"]) for site in prim["basis"]
        ]

    return cu.xtal.Structure(lat,[cu.xtal.Site(c,s) for c,s in zip(coords,species)])


def make_prim_mapper(prim):
    """Construct a cu.mapping.structure.StructureMapper
    that will map to the given primitive structure, using
    the specified degrees of freedom as occupants

    Parameters
    ----------
    prim : dict

    Returns
    -------
    cu.mapping.structure.StructureMapper

    """
    prim_struc=make_prim(prim)
    allowed_species=make_allowed_species(prim)

    return cu.mapping.structure.StructureMapper(prim_struc,allowed_species=allowed_species)


def main():

if __name__ == "__main__":
    main()
