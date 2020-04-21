import unittest
import casmutils as cu
# from .prim_bin import ProjectFilesystem
import prim_bin
import json
import os

ProjectFilesystem = prim_bin.ProjectFilesystem
test_input_root = "./tests"


class ProjectFilesystemTest(unittest.TestCase):

    def setUp(self):
        self.fcc_fs = ProjectFilesystem(
            os.path.join(test_input_root, "NiAl"), "NiAl-FCC")

        self.b2_fs = ProjectFilesystem(
            os.path.join(test_input_root, "NiAl-B2"), calctype="default")

    def test_root(self):
        self.assertTrue(os.path.exists(self.fcc_fs.root))
        self.assertTrue(os.path.exists(self.b2_fs.root))

    def test_name(self):
        self.assertEqual(self.fcc_fs.name, "NiAl-FCC")
        self.assertEqual(self.b2_fs.name, "NiAl-B2")

    def test_calctype(self):
        self.assertEqual(self.fcc_fs.calctype, "default")
        self.assertEqual(self.b2_fs.calctype, "default")

    def test_prim(self):
        self.assertTrue(os.path.exists(self.fcc_fs.prim()))

    def test_pos(self):
        self.assertTrue(os.path.exists(self.fcc_fs.pos("SCEL5_1_1_5_0_0_0/0")))

    def test_calc_properties(self):
        fcc_prim_calc_properties = self.fcc_fs.calc_properties(
            "SCEL1_1_1_1_0_0_0/0")

        self.assertTrue(
            "NiAl/training_data/SCEL1_1_1_1_0_0_0/0/calctype.default/properties.calc.json"
            in fcc_prim_calc_properties)


class PrimBinTest(unittest.TestCase):

    def setUp(self):
        self.b2_fs = ProjectFilesystem(os.path.join(test_input_root, "NiAl-B2"))
        with open(self.b2_fs.prim()) as prim_data:
            self.b2_prim = json.load(prim_data)

        self.fcc_fs = ProjectFilesystem(os.path.join(test_input_root, "NiAl"))
        self.fcc_configname="SCEL5_1_1_5_0_0_0/0"
        self.fcc_config_props_file = self.fcc_fs.calc_properties(self.fcc_configname)

        with open(self.fcc_config_props_file) as json_data:
            self.fcc_config_props = json.load(json_data)


        self.fcc_config_final_struc = cu.xtal.Structure.from_poscar(
            os.path.join(
                self.fcc_fs.configuration(self.fcc_configname),
                "calctype.default", "run.final", "CONTCAR"))


    def test_make_lattice(self):
        lat = prim_bin.make_lattice(self.fcc_config_props)
        self.assertEqual(self.fcc_config_final_struc.lattice(), lat)

    def test_make_basis(self):
        basis= prim_bin.make_basis(self.fcc_config_props)
        self.assertEqual(self.fcc_config_final_struc.basis_sites(), basis)

    def test_read_structure(self):
        struc=prim_bin.read_structure(self.fcc_config_props_file)
        final_struc=self.fcc_config_final_struc

        self.assertEqual(struc.lattice(),final_struc.lattice())
        self.assertEqual(struc.basis_sites(),final_struc.basis_sites())

    def test_make_allowed_species(self):
        self.assertEqual([["Ni","Va"],["Al","Ni"]],prim_bin.make_allowed_species(self.b2_prim))

    def test_make_prim(self):
        prim_config=cu.xtal.Structure.from_poscar(self.b2_fs.pos("SCEL1_1_1_1_0_0_0/0"))
        prim_prim=prim_bin.make_prim(self.b2_prim)

        self.assertEqual(prim_config.lattice(),prim_config.lattice())
        self.assertEqual(prim_prim.basis_sites(),prim_prim.basis_sites())

    def test_make_prim_mapper(self):
        map_to_b2=prim_bin.make_prim_mapper(self.b2_prim)
        configs=["SCEL1_1_1_1_0_0_0/0","SCEL1_1_1_1_0_0_0/1","SCEL1_1_1_1_0_0_0/2","SCEL1_1_1_1_0_0_0/3"]
        ideal_strucs=[cu.xtal.Structure.from_poscar(self.b2_fs.pos(p)) for p in configs]

        reports=[map_to_b2(s) for s in ideal_strucs]
        for rs in reports:
            for r in rs:
                self.assertAlmostEqual(r.cost,0)

        bad_rs=map_to_b2(cu.xtal.Structure.from_poscar(self.fcc_fs.pos("SCEL1_1_1_1_0_0_0/0")))
        for r in bad_rs:
            self.assertNotAlmostEqual(r.cost,0)



def main():
    unittest.main()


if __name__ == "__main__":
    main()
