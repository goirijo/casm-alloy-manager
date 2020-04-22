import casmutils as cu
import json
import os
import glob
import numpy as np


def returned_path_must_exist(returns_path):
    """Decorator that throws exception if the
    returned value of returns_path does not
    exist

    Parameters
    ----------
    returns_path : function, returns path to file

    Returns
    -------
    function

    """

    def except_if_doesnt_exist(*args, **kwargs):
        path = returns_path(*args, **kwargs)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path

    return except_if_doesnt_exist


def return_None_if_doesnt_exist(returns_path):
    """Decorator that returns None if
    returned value of returns_path does not
    exist

    Parameters
    ----------
    returns_path : function, returns path to file

    Returns
    -------
    function

    """

    def none_if_doesnt_exist(*args, **kwargs):
        path = returns_path(*args, **kwargs)
        if not os.path.exists(path):
            return None
        return path

    return none_if_doesnt_exist


def read_json(json_file):
    with open(json_file) as json_data:
        d = json.load(json_data)
    return d


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

    @returned_path_must_exist
    def prim(self):
        """Return path to prim.json file
        Returns
        -------
        path

        """
        return os.path.join(self.root, "prim.json")

    @returned_path_must_exist
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

    @returned_path_must_exist
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

    @return_None_if_doesnt_exist
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

    def confignames(self):
        """Returns list of all the confignames that were found in the
        project filesystem

        Returns
        -------
        list(path)

        """
        paths = [
            p for p in glob.glob(
                os.path.join(self.root, "training_data", "SCEL*_*_*_*_*_*_*",
                             "*")) if os.path.isdir(p)
        ]

        prefix = os.path.commonpath(paths)
        return [os.path.relpath(p, start=prefix) for p in paths]


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
    d = read_json(properties_file)
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

    return cu.xtal.Structure(
        lat, [cu.xtal.Site(c, s) for c, s in zip(coords, species)])


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
    prim_struc = make_prim(prim)
    allowed_species = make_allowed_species(prim)

    #TODO: use_crystal_symmetry=True ?
    return cu.mapping.structure.StructureMapper(
        prim_struc, allowed_species=allowed_species, use_crystal_symmetry=True)


def load_relaxed_structures(fs, confignames=None):
    """Create cu.xtal.Structure objects from the relaxed
    properties files for each of the specified configurations
    within the specified casm project. If the configuration
    exists, but has not been relaxed yet, inserts None.

    Parameters
    ----------
    fs : ProjectFilesystem
    confignames : list(str), optional (all configurations if not specified)

    Returns
    -------
    list(cu.xtal.Structure)

    """
    if confignames is None:
        confignames = fs.confignames()

    props_files = [fs.calc_properties(c) for c in confignames]
    return [
        make_structure(read_json(p)) if p is not None else None
        for p in props_files
    ]


def load_prim_mapper(fs):
    """Given a project filesystem, construct a mapper
    object that uses the prim of the project as a
    reference structure.

    Parameters
    ----------
    fs : ProjectFilesystem

    Returns
    -------
    cu.mapping.structure.StructureMapper

    """
    return make_prim_mapper(read_json(fs.prim()))


def rank_relaxed_structure_mapping_scores(mappers, relaxed_structures):
    """Pass each relaxed structure to all of the mapping objects
    and rank how well they each scored. If elements in the relaxed
    structures list are None, they are assigned -1 rank.

    Parameters
    ----------
    mappers : list(cu.xtal.mapping.structure.StructureMapper)
    relaxed_structures : list(cu.xtal.Structure)

    Returns
    -------
    list(list(int)) where 0 is the best map

    """
    scores = np.array([[m(p)[0].cost
                        for m in mappers]
                       if p is not None else len(mappers) * [np.nan]
                       for p in relaxed_structures])
    ranks = np.argsort(scores)
    mask=np.isnan(scores)
    ranks[mask]=-1
    return ranks


def main():
    #Specify root directories for each casm project
    fcc_fs = ProjectFilesystem("./tests/NiAl", name="NiAl-FCC")
    b2_fs = ProjectFilesystem("./tests/NiAl-B2", name="NiAl-B2")

    #Store each project filesystem in a list
    fss = [fcc_fs, b2_fs]
    #Load every configuration available in the FCC project
    fcc_confignames=fcc_fs.confignames()
    #Specify which relaxed structures to load (all of them in this case)
    fcc_relaxed_strucs = load_relaxed_structures(fcc_fs,fcc_confignames)
    #For each project, create a mapper that uses its prim as the reference structure
    mappers = [load_prim_mapper(fs) for fs in fss]

    #Rank how well all of the relaxed FCC structures we loaded earlier
    #map to each of the project prims
    ranks = rank_relaxed_structure_mapping_scores(mappers, fcc_relaxed_strucs)

    #print results
    print([fs.name for fs in fss])
    for c, r in zip(fcc_confignames, ranks):
        print(c, r)

if __name__ == "__main__":
    main()
