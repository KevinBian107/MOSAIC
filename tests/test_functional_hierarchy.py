"""Tests for functional hierarchy builder.

Tests the FunctionalHierarchyBuilder and TwoLevelHierarchy classes
for correct hierarchy construction and reconstruction.
"""

import pytest
import torch
from torch_geometric.data import Data

from src.tokenizers.coarsening.functional_hierarchy import FunctionalHierarchyBuilder
from src.tokenizers.structures import (
    CommunityCommunityEdge,
    FunctionalCommunity,
    TwoLevelHierarchy,
)

# Optional RDKit import
try:
    from src.data.molecular import smiles_to_graph

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ===========================================================================
# Test Fixtures
# ===========================================================================


def edges_equal(g1: Data, g2: Data) -> bool:
    """Check if two graphs have the same edges (ignoring order and direction)."""

    def edge_set(g: Data) -> frozenset[tuple[int, int]]:
        if g.edge_index.numel() == 0:
            return frozenset()
        ei = g.edge_index.numpy()
        return frozenset(
            (min(int(ei[0, i]), int(ei[1, i])), max(int(ei[0, i]), int(ei[1, i])))
            for i in range(ei.shape[1])
            if ei[0, i] != ei[1, i]
        )

    return edge_set(g1) == edge_set(g2)


@pytest.fixture
def simple_graph() -> Data:
    """A simple graph with 4 nodes: 0-1-2-3."""
    edges = [(0, 1), (1, 2), (2, 3)]
    edge_list = [(s, d) for s, d in edges] + [(d, s) for s, d in edges]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    return Data(edge_index=edge_index, num_nodes=4)


@pytest.fixture
def triangle_graph() -> Data:
    """A simple triangle (3-clique)."""
    edges = [(0, 1), (1, 2), (2, 0)]
    edge_list = [(s, d) for s, d in edges] + [(d, s) for s, d in edges]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    return Data(edge_index=edge_index, num_nodes=3)


@pytest.fixture
def two_triangles() -> Data:
    """Two triangles connected by one edge."""
    edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)]
    edge_list = [(s, d) for s, d in edges] + [(d, s) for s, d in edges]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    return Data(edge_index=edge_index, num_nodes=6)


# ===========================================================================
# Test FunctionalCommunity
# ===========================================================================


class TestFunctionalCommunity:
    """Tests for FunctionalCommunity dataclass."""

    def test_community_creation(self) -> None:
        """Test creating a FunctionalCommunity."""
        community = FunctionalCommunity(
            community_id=0,
            community_type="ring",
            group_name="benzene",
            atom_indices=[0, 1, 2, 3, 4, 5],
            internal_edges=[(0, 1), (1, 0)],
        )
        assert community.community_id == 0
        assert community.community_type == "ring"
        assert community.num_atoms == 6
        assert community.num_edges == 2

    def test_singleton_community(self) -> None:
        """Test singleton community."""
        community = FunctionalCommunity(
            community_id=0,
            community_type="singleton",
            group_name="singleton",
            atom_indices=[0],
            internal_edges=[],
        )
        assert community.num_atoms == 1
        assert community.num_edges == 0


# ===========================================================================
# Test CommunityCommunityEdge
# ===========================================================================


class TestCommunityCommunityEdge:
    """Tests for CommunityCommunityEdge dataclass."""

    def test_edge_creation(self) -> None:
        """Test creating a CommunityCommunityEdge."""
        edge = CommunityCommunityEdge(
            source_community=0,
            target_community=1,
            source_atom=5,
            target_atom=6,
            bond_type=1,
        )
        assert edge.source_community == 0
        assert edge.target_community == 1
        assert edge.source_atom == 5
        assert edge.target_atom == 6
        assert edge.bond_type == 1


# ===========================================================================
# Test TwoLevelHierarchy
# ===========================================================================


class TestTwoLevelHierarchy:
    """Tests for TwoLevelHierarchy dataclass."""

    def test_hierarchy_creation(self) -> None:
        """Test creating a TwoLevelHierarchy."""
        comm0 = FunctionalCommunity(
            community_id=0,
            community_type="singleton",
            group_name="singleton",
            atom_indices=[0, 1],
            internal_edges=[(0, 1), (1, 0)],
        )
        comm1 = FunctionalCommunity(
            community_id=1,
            community_type="singleton",
            group_name="singleton",
            atom_indices=[2, 3],
            internal_edges=[(2, 3), (3, 2)],
        )
        super_edge = CommunityCommunityEdge(
            source_community=0,
            target_community=1,
            source_atom=1,
            target_atom=2,
        )

        hierarchy = TwoLevelHierarchy(
            communities=[comm0, comm1],
            super_edges=[super_edge],
            atom_to_community=[0, 0, 1, 1],
            num_atoms=4,
        )

        assert hierarchy.num_communities == 2
        assert hierarchy.num_super_edges == 1
        assert hierarchy.num_atoms == 4

    def test_get_community(self) -> None:
        """Test getting a community by ID."""
        comm0 = FunctionalCommunity(
            community_id=0,
            community_type="singleton",
            group_name="singleton",
            atom_indices=[0],
            internal_edges=[],
        )
        comm1 = FunctionalCommunity(
            community_id=1,
            community_type="singleton",
            group_name="singleton",
            atom_indices=[1],
            internal_edges=[],
        )

        hierarchy = TwoLevelHierarchy(
            communities=[comm0, comm1],
            super_edges=[],
            atom_to_community=[0, 1],
            num_atoms=2,
        )

        assert hierarchy.get_community(0) == comm0
        assert hierarchy.get_community(1) == comm1

        with pytest.raises(KeyError):
            hierarchy.get_community(5)

    def test_reconstruct_simple(self) -> None:
        """Test reconstructing a simple graph."""
        comm0 = FunctionalCommunity(
            community_id=0,
            community_type="singleton",
            group_name="singleton",
            atom_indices=[0, 1],
            internal_edges=[(0, 1), (1, 0)],
        )
        comm1 = FunctionalCommunity(
            community_id=1,
            community_type="singleton",
            group_name="singleton",
            atom_indices=[2],
            internal_edges=[],
        )
        super_edge = CommunityCommunityEdge(
            source_community=0,
            target_community=1,
            source_atom=1,
            target_atom=2,
        )

        hierarchy = TwoLevelHierarchy(
            communities=[comm0, comm1],
            super_edges=[super_edge],
            atom_to_community=[0, 0, 1],
            num_atoms=3,
        )

        reconstructed = hierarchy.reconstruct()

        assert reconstructed.num_nodes == 3
        assert reconstructed.edge_index.shape[1] > 0

    def test_get_level_info(self) -> None:
        """Test getting level info."""
        comm0 = FunctionalCommunity(
            community_id=0,
            community_type="ring",
            group_name="benzene",
            atom_indices=[0, 1, 2],
            internal_edges=[],
        )
        comm1 = FunctionalCommunity(
            community_id=1,
            community_type="singleton",
            group_name="singleton",
            atom_indices=[3],
            internal_edges=[],
        )

        hierarchy = TwoLevelHierarchy(
            communities=[comm0, comm1],
            super_edges=[],
            atom_to_community=[0, 0, 0, 1],
            num_atoms=4,
        )

        info = hierarchy.get_level_info()
        assert info["num_communities"] == 2
        assert info["num_atoms"] == 4
        assert "ring" in info["community_types"]
        assert "singleton" in info["community_types"]


# ===========================================================================
# Test FunctionalHierarchyBuilder
# ===========================================================================


class TestFunctionalHierarchyBuilder:
    """Tests for FunctionalHierarchyBuilder class."""

    def test_builder_creation(self) -> None:
        """Test creating a builder."""
        builder = FunctionalHierarchyBuilder()
        assert builder.include_rings is True

    def test_build_simple_graph(self, simple_graph: Data) -> None:
        """Test building hierarchy for simple graph without SMILES."""
        builder = FunctionalHierarchyBuilder()
        hierarchy = builder.build(simple_graph)

        # Without SMILES, all atoms become singletons
        assert hierarchy.num_atoms == 4
        assert hierarchy.num_communities == 4

        # Should have super-edges connecting singletons
        assert hierarchy.num_super_edges > 0

    def test_all_atoms_assigned(self, simple_graph: Data) -> None:
        """Test that all atoms are assigned to exactly one community."""
        builder = FunctionalHierarchyBuilder()
        hierarchy = builder.build(simple_graph)

        # Check atom_to_community mapping
        assert len(hierarchy.atom_to_community) == simple_graph.num_nodes

        # All values should be valid community IDs
        community_ids = {c.community_id for c in hierarchy.communities}
        for comm_id in hierarchy.atom_to_community:
            assert comm_id in community_ids

        # All atoms should appear in exactly one community
        atom_counts = {}
        for comm in hierarchy.communities:
            for atom in comm.atom_indices:
                atom_counts[atom] = atom_counts.get(atom, 0) + 1

        for atom in range(simple_graph.num_nodes):
            assert atom_counts.get(atom, 0) == 1, f"Atom {atom} not in exactly one community"

    def test_roundtrip_simple_graph(self, simple_graph: Data) -> None:
        """Test roundtrip for simple graph."""
        builder = FunctionalHierarchyBuilder()
        hierarchy = builder.build(simple_graph)
        reconstructed = hierarchy.reconstruct()

        assert edges_equal(simple_graph, reconstructed)

    def test_roundtrip_triangle(self, triangle_graph: Data) -> None:
        """Test roundtrip for triangle graph."""
        builder = FunctionalHierarchyBuilder()
        hierarchy = builder.build(triangle_graph)
        reconstructed = hierarchy.reconstruct()

        assert edges_equal(triangle_graph, reconstructed)

    def test_roundtrip_two_triangles(self, two_triangles: Data) -> None:
        """Test roundtrip for two triangles."""
        builder = FunctionalHierarchyBuilder()
        hierarchy = builder.build(two_triangles)
        reconstructed = hierarchy.reconstruct()

        assert edges_equal(two_triangles, reconstructed)


# ===========================================================================
# Test with Molecular Data (requires RDKit)
# ===========================================================================


@pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not available")
class TestMolecularHierarchy:
    """Tests for hierarchy building with molecular data."""

    def test_benzene_hierarchy(self) -> None:
        """Test hierarchy for benzene."""
        data = smiles_to_graph("c1ccccc1")
        assert data is not None

        builder = FunctionalHierarchyBuilder()
        hierarchy = builder.build(data)

        # Should detect benzene as a ring community
        ring_communities = [
            c for c in hierarchy.communities if c.community_type == "ring"
        ]
        assert len(ring_communities) >= 1

    def test_phenol_hierarchy(self) -> None:
        """Test hierarchy for phenol (Oc1ccccc1)."""
        data = smiles_to_graph("Oc1ccccc1")
        assert data is not None

        builder = FunctionalHierarchyBuilder()
        hierarchy = builder.build(data)

        # Should have ring community
        ring_communities = [
            c for c in hierarchy.communities if c.community_type == "ring"
        ]
        assert len(ring_communities) >= 1

        # Total atoms should match
        total_atoms = sum(c.num_atoms for c in hierarchy.communities)
        assert total_atoms == data.num_nodes

    def test_molecular_roundtrip(self) -> None:
        """Test roundtrip for molecular graphs."""
        molecules = [
            "CCO",  # Ethanol
            "c1ccccc1",  # Benzene
            "Oc1ccccc1",  # Phenol
            "CC(=O)O",  # Acetic acid
        ]

        builder = FunctionalHierarchyBuilder()

        for smiles in molecules:
            data = smiles_to_graph(smiles)
            if data is None:
                continue

            hierarchy = builder.build(data)
            reconstructed = hierarchy.reconstruct()

            assert edges_equal(data, reconstructed), f"Roundtrip failed for {smiles}"

    @pytest.mark.parametrize("smiles,name", [
        ("CC(=O)OC1=CC=CC=C1C(=O)O", "aspirin"),
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "caffeine"),
        ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "ibuprofen"),
    ])
    def test_drug_molecules(self, smiles: str, name: str) -> None:
        """Test hierarchy for drug molecules."""
        data = smiles_to_graph(smiles)
        assert data is not None, f"Failed to parse {name}"

        builder = FunctionalHierarchyBuilder()
        hierarchy = builder.build(data)

        # Should have multiple communities
        assert hierarchy.num_communities >= 1

        # Roundtrip should preserve edges
        reconstructed = hierarchy.reconstruct()
        assert edges_equal(data, reconstructed), f"Roundtrip failed for {name}"

    def test_no_overlapping_communities(self) -> None:
        """Test that community atoms don't overlap."""
        data = smiles_to_graph("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")  # Caffeine
        assert data is not None

        builder = FunctionalHierarchyBuilder()
        hierarchy = builder.build(data)

        seen_atoms: set[int] = set()
        for comm in hierarchy.communities:
            for atom in comm.atom_indices:
                assert atom not in seen_atoms, f"Atom {atom} in multiple communities"
                seen_atoms.add(atom)


# ===========================================================================
# Test Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases in hierarchy building."""

    def test_empty_graph(self) -> None:
        """Test hierarchy for empty graph."""
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        data = Data(edge_index=edge_index, num_nodes=0)

        builder = FunctionalHierarchyBuilder()
        hierarchy = builder.build(data)

        assert hierarchy.num_atoms == 0
        assert hierarchy.num_communities == 0

    def test_single_node(self) -> None:
        """Test hierarchy for single node graph."""
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        data = Data(edge_index=edge_index, num_nodes=1)

        builder = FunctionalHierarchyBuilder()
        hierarchy = builder.build(data)

        assert hierarchy.num_atoms == 1
        assert hierarchy.num_communities == 1
        assert hierarchy.communities[0].community_type == "singleton"

    def test_disconnected_graph(self) -> None:
        """Test hierarchy for disconnected graph."""
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
        data = Data(edge_index=edge_index, num_nodes=4)

        builder = FunctionalHierarchyBuilder()
        hierarchy = builder.build(data)

        # Should have 4 singletons (no SMILES for detection)
        assert hierarchy.num_atoms == 4

        # Roundtrip should work
        reconstructed = hierarchy.reconstruct()
        assert edges_equal(data, reconstructed)
