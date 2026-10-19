#include <algorithm>
#include <cmath>
#include <ranges>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <Base/GlobalSettings.h>
#include <Base/Math.h>

#include <Data/Descs.h>

#include <Engine/Interface/GeometryBuffers.h>
#include <Engine/Interface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class GeometryTests : public IntegrationTestFramework
{
public:
    GeometryTests()
        : IntegrationTestFramework()
    {}

protected:
    ObjectVertexData getVertexNextTo(std::vector<ObjectVertexData> const& vertices, RealVector2D const& pos) const
    {
        return *std::ranges::min_element(vertices, {}, [&](ObjectVertexData const& vertex) {
            auto dx = vertex.pos[0] - pos.x;
            auto dy = vertex.pos[1] - pos.y;
            return dx * dx + dy * dy;
        });
    }

    template <typename Getter>
    float sumShadowGrid(std::vector<ShadowGridNodeData> const& nodes, Getter getter) const
    {
        auto result = 0.0f;
        for (auto const& node : nodes) {
            result += static_cast<float>(getter(node));
        }
        return result / ShadowGridNodeData::FixedPointScale;
    }
};

TEST_F(GeometryTests, copyBuffers_emptySim)
{
    _simulationFacade->clear();
    auto geometryBuffers = _HostGeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToRenderer(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(0u, numObjects.objects);
    EXPECT_EQ(0u, numObjects.fluidParticles);
    EXPECT_EQ(0u, numObjects.lineIndices);
    EXPECT_EQ(0u, numObjects.triangleIndices);
}

TEST_F(GeometryTests, copyBuffers_objects)
{
    auto data = ContentDesc().addCreature({
        ObjectDesc().id(1).pos({100.0f, 100.0f}),
        ObjectDesc().id(2).pos({101.0f, 100.0f}),
        ObjectDesc().id(3).pos({102.0f, 100.0f}),
    });
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _HostGeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToRenderer(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(3u, numObjects.objects);
    EXPECT_EQ(0u, numObjects.fluidParticles);

    // Verify buffer entries
    auto cellData = geometryBuffers->getCellData();
    EXPECT_EQ(3u, cellData.size());
}

TEST_F(GeometryTests, copyBuffers_cullsObjectsOutsideVisibleRect)
{
    auto data = ContentDesc().addCreature({
        ObjectDesc().id(1).pos({5.0f, 5.0f}),
        ObjectDesc().id(2).pos({500.0f, 500.0f}),
    });
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _HostGeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {10, 10}};

    _simulationFacade->tryCopyBuffersFromCudaToRenderer(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(1u, numObjects.objects);

    auto cellData = geometryBuffers->getCellData();
    ASSERT_EQ(1u, cellData.size());
    EXPECT_FLOAT_EQ(5.0f, cellData.at(0).pos[0]);
    EXPECT_FLOAT_EQ(5.0f, cellData.at(0).pos[1]);
}

TEST_F(GeometryTests, copyBuffers_fluidParticles)
{
    auto data = ContentDesc().energies({
        EnergyDesc().id(1).pos({100.0f, 100.0f}).energy(10.0f),
        EnergyDesc().id(2).pos({101.0f, 100.0f}).energy(10.0f),
        EnergyDesc().id(3).pos({102.0f, 100.0f}).energy(10.0f),
        EnergyDesc().id(4).pos({103.0f, 100.0f}).energy(10.0f),
    });
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _HostGeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToRenderer(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(4u, numObjects.fluidParticles);

    // Verify buffer entries
    auto particleData = geometryBuffers->getFluidParticleData();
    EXPECT_EQ(4u, particleData.size());
}

TEST_F(GeometryTests, copyBuffers_cullsFluidParticlesOutsideVisibleRect)
{
    auto data = ContentDesc().energies({
        EnergyDesc().id(1).pos({5.0f, 5.0f}).energy(10.0f),
        EnergyDesc().id(2).pos({500.0f, 500.0f}).energy(10.0f),
    });
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _HostGeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {10, 10}};

    _simulationFacade->tryCopyBuffersFromCudaToRenderer(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(1u, numObjects.fluidParticles);

    auto particleData = geometryBuffers->getFluidParticleData();
    ASSERT_EQ(1u, particleData.size());
    EXPECT_FLOAT_EQ(5.0f, particleData.at(0).pos[0]);
    EXPECT_FLOAT_EQ(5.0f, particleData.at(0).pos[1]);
}

TEST_F(GeometryTests, copyBuffers_energyParticleVelocity)
{
    auto data = ContentDesc().energies({
        EnergyDesc().id(1).pos({100.0f, 100.0f}).vel({0.5f, -0.25f}).energy(10.0f),
    });
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _HostGeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToRenderer(geometryBuffers, visibleWorldRect);

    auto particleData = geometryBuffers->getFluidParticleData();
    ASSERT_EQ(1u, particleData.size());
    EXPECT_FLOAT_EQ(0.5f, particleData.at(0).vel[0]);
    EXPECT_FLOAT_EQ(-0.25f, particleData.at(0).vel[1]);
}

TEST_F(GeometryTests, copyBuffers_cellsWithConnections)
{
    auto data = ContentDesc().addCreature({
        ObjectDesc().id(1).pos({100.0f, 100.0f}),
        ObjectDesc().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    auto geometryBuffers = _HostGeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToRenderer(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(2u, numObjects.objects);
    EXPECT_EQ(2u, numObjects.lineIndices);

    // Verify buffer entries
    auto lines = geometryBuffers->getLineIndices();
    EXPECT_EQ(2u, lines.size());
}

TEST_F(GeometryTests, copyBuffers_cullsConnectionsOutsideVisibleRect)
{
    auto data = ContentDesc().addCreature({
        ObjectDesc().id(1).pos({5.0f, 5.0f}),
        ObjectDesc().id(2).pos({6.0f, 5.0f}),
        ObjectDesc().id(3).pos({500.0f, 500.0f}),
        ObjectDesc().id(4).pos({501.0f, 500.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(3, 4);
    _simulationFacade->setSimulationData(data);

    auto geometryBuffers = _HostGeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {10, 10}};

    _simulationFacade->tryCopyBuffersFromCudaToRenderer(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(2u, numObjects.objects);
    EXPECT_EQ(2u, numObjects.lineIndices);

    auto lines = geometryBuffers->getLineIndices();
    EXPECT_EQ(2u, lines.size());
}

TEST_F(GeometryTests, copyBuffers_triangle)
{
    auto data = ContentDesc().addCreature({
        ObjectDesc().id(1).pos({100.0f, 100.0f}),
        ObjectDesc().id(2).pos({101.0f, 100.0f}),
        ObjectDesc().id(3).pos({100.5f, 100.866f}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 1);
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _HostGeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToRenderer(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(3u, numObjects.objects);
    EXPECT_EQ(6u, numObjects.lineIndices);
    EXPECT_EQ(6u, numObjects.triangleIndices);

    // Verify buffer entries
    auto lines = geometryBuffers->getLineIndices();
    EXPECT_EQ(6u, lines.size());

    auto triangles = geometryBuffers->getTriangleIndices();
    EXPECT_EQ(6u, triangles.size());
}

TEST_F(GeometryTests, copyBuffers_quad)
{
    auto data = ContentDesc().addCreature({
        ObjectDesc().id(1).pos({100.0f, 100.0f}),
        ObjectDesc().id(2).pos({101.0f, 100.0f}),
        ObjectDesc().id(3).pos({101.0f, 101.0f}),
        ObjectDesc().id(4).pos({100.0f, 101.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);
    data.addConnection(4, 1);
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _HostGeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToRenderer(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(4u, numObjects.objects);
    EXPECT_EQ(8u, numObjects.lineIndices);
    EXPECT_EQ(12u, numObjects.triangleIndices);

    // Verify buffer entries
    auto lines = geometryBuffers->getLineIndices();
    EXPECT_EQ(8u, lines.size());

    auto triangles = geometryBuffers->getTriangleIndices();
    EXPECT_EQ(12u, triangles.size());
}

TEST_F(GeometryTests, copyBuffers_smoothsStepInSolidOutline)
{
    // Two rows of solids with a single solid on top, where the outline steps out and back in
    auto constexpr RowDistance = 0.866f;
    auto data = ContentDesc().objects({
        ObjectDesc().id(1).pos({100.0f, 100.0f}).type(SolidDesc()),
        ObjectDesc().id(2).pos({101.0f, 100.0f}).type(SolidDesc()),
        ObjectDesc().id(3).pos({102.0f, 100.0f}).type(SolidDesc()),
        ObjectDesc().id(4).pos({103.0f, 100.0f}).type(SolidDesc()),
        ObjectDesc().id(5).pos({104.0f, 100.0f}).type(SolidDesc()),
        ObjectDesc().id(6).pos({100.5f, 100.0f - RowDistance}).type(SolidDesc()),
        ObjectDesc().id(7).pos({101.5f, 100.0f - RowDistance}).type(SolidDesc()),
        ObjectDesc().id(8).pos({102.5f, 100.0f - RowDistance}).type(SolidDesc()),
        ObjectDesc().id(9).pos({103.5f, 100.0f - RowDistance}).type(SolidDesc()),
        ObjectDesc().id(10).pos({102.0f, 100.0f - 2 * RowDistance}).type(SolidDesc()),
    });
    for (auto const& [id1, id2] : std::vector<std::pair<uint64_t, uint64_t>>{
             {1, 2}, {2, 3}, {3, 4}, {4, 5}, {6, 7}, {7, 8}, {8, 9}, {6, 1}, {6, 2}, {7, 2}, {7, 3}, {8, 3}, {8, 4}, {9, 4}, {9, 5}, {10, 7}, {10, 8}}) {
        data.addConnection(id1, id2);
    }
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _HostGeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToRenderer(geometryBuffers, visibleWorldRect);

    auto vertices = geometryBuffers->getCellData();
    ASSERT_EQ(10u, vertices.size());
    auto topSolid = getVertexNextTo(vertices, {102.0f, 100.0f - 2 * RowDistance});
    EXPECT_FLOAT_EQ(102.0f, topSolid.pos[0]);
    EXPECT_GT(topSolid.pos[1], 100.0f - 1.6f * RowDistance);
    auto solidOnStraightOutline = getVertexNextTo(vertices, {102.0f, 100.0f});
    EXPECT_FLOAT_EQ(102.0f, solidOnStraightOutline.pos[0]);
    EXPECT_FLOAT_EQ(100.0f, solidOnStraightOutline.pos[1]);
}

TEST_F(GeometryTests, copyBuffers_keepsDistantSolidCloseToItsPosition)
{
    // Two rows of solids with a single solid far above, which is connected to two solids of the upper row
    auto constexpr RowDistance = 0.866f;
    auto constexpr DistantSolidPos = RealVector2D{102.0f, 100.0f - RowDistance - 2.5f};
    auto data = ContentDesc().objects({
        ObjectDesc().id(1).pos({100.0f, 100.0f}).type(SolidDesc()),
        ObjectDesc().id(2).pos({101.0f, 100.0f}).type(SolidDesc()),
        ObjectDesc().id(3).pos({102.0f, 100.0f}).type(SolidDesc()),
        ObjectDesc().id(4).pos({103.0f, 100.0f}).type(SolidDesc()),
        ObjectDesc().id(5).pos({104.0f, 100.0f}).type(SolidDesc()),
        ObjectDesc().id(6).pos({100.5f, 100.0f - RowDistance}).type(SolidDesc()),
        ObjectDesc().id(7).pos({101.5f, 100.0f - RowDistance}).type(SolidDesc()),
        ObjectDesc().id(8).pos({102.5f, 100.0f - RowDistance}).type(SolidDesc()),
        ObjectDesc().id(9).pos({103.5f, 100.0f - RowDistance}).type(SolidDesc()),
        ObjectDesc().id(10).pos(DistantSolidPos).type(SolidDesc()),
    });
    for (auto const& [id1, id2] : std::vector<std::pair<uint64_t, uint64_t>>{
             {1, 2}, {2, 3}, {3, 4}, {4, 5}, {6, 7}, {7, 8}, {8, 9}, {6, 1}, {6, 2}, {7, 2}, {7, 3}, {8, 3}, {8, 4}, {9, 4}, {9, 5}, {10, 7}, {10, 8}}) {
        data.addConnection(id1, id2);
    }
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _HostGeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToRenderer(geometryBuffers, visibleWorldRect);

    auto vertices = geometryBuffers->getCellData();
    ASSERT_EQ(10u, vertices.size());
    auto distantSolid = getVertexNextTo(vertices, DistantSolidPos);
    EXPECT_LE(std::hypot(distantSolid.pos[0] - DistantSolidPos.x, distantSolid.pos[1] - DistantSolidPos.y), 0.51f);
}

TEST_F(GeometryTests, copyBuffers_keepsOutlineOfRoundSolid)
{
    // Hexagon of solids around a central solid
    auto data = ContentDesc().objects({ObjectDesc().id(1).pos({100.0f, 100.0f}).type(SolidDesc())});
    for (auto const& index : std::views::iota(0, 6)) {
        auto angle = toFloat(index) * Const::Pi / 3;
        data._objects.emplace_back(ObjectDesc().id(index + 2).pos({100.0f + std::cos(angle), 100.0f + std::sin(angle)}).type(SolidDesc()));
    }
    for (auto const& index : std::views::iota(0, 6)) {
        data.addConnection(1, index + 2);
        data.addConnection(index + 2, (index + 1) % 6 + 2);
    }
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _HostGeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToRenderer(geometryBuffers, visibleWorldRect);

    auto vertices = geometryBuffers->getCellData();
    ASSERT_EQ(7u, vertices.size());
    for (auto const& object : data._objects) {
        auto vertex = getVertexNextTo(vertices, object._pos);
        EXPECT_NEAR(object._pos.x, vertex.pos[0], 1e-4f);
        EXPECT_NEAR(object._pos.y, vertex.pos[1], 1e-4f);
    }
}

TEST_F(GeometryTests, copyBuffers_mixedCellsAndParticles)
{
    auto data = ContentDesc()
                    .addCreature({
                        ObjectDesc().id(1).pos({100.0f, 100.0f}),
                        ObjectDesc().id(2).pos({101.0f, 100.0f}),
                    })
                    .energies({
                        EnergyDesc().id(3).pos({200.0f, 200.0f}).energy(10.0f),
                        EnergyDesc().id(4).pos({201.0f, 200.0f}).energy(10.0f),
                        EnergyDesc().id(5).pos({202.0f, 200.0f}).energy(10.0f),
                    });
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _HostGeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToRenderer(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(2u, numObjects.objects);
    EXPECT_EQ(3u, numObjects.fluidParticles);

    // Verify buffer entries
    auto particleData = geometryBuffers->getFluidParticleData();
    EXPECT_EQ(3u, particleData.size());

    auto cellData = geometryBuffers->getCellData();
    EXPECT_EQ(2u, cellData.size());
}

TEST_F(GeometryTests, copyBuffers_marksFluids)
{
    auto data = ContentDesc().energies({EnergyDesc().id(1).pos({200.0f, 200.0f}).energy(10.0f)});
    data._objects.emplace_back(ObjectDesc().id(2).pos({100.0f, 100.0f}).type(FluidDesc()));
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _HostGeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToRenderer(geometryBuffers, visibleWorldRect);

    // The z coordinate is 1 for fluids and 0 for energy particles
    auto particleData = geometryBuffers->getFluidParticleData();
    ASSERT_EQ(2u, particleData.size());
    for (auto const& particle : particleData) {
        auto isFluid = particle.pos[0] < 150.0f;
        EXPECT_FLOAT_EQ(isFluid ? 1.0f : 0.0f, particle.pos[2]);
    }
}

TEST_F(GeometryTests, copyBuffers_activityShowsPositivePartOfFirstSignal)
{
    auto data = ContentDesc().addCreature({
        ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().neuralActivity(NeuralActivityDesc().signals({1.0f, 0, 0, 0, 0, 0, 0, 0}))),
        ObjectDesc().id(2).pos({200.0f, 100.0f}).type(CellDesc().neuralActivity(NeuralActivityDesc().signals({-2.0f, 2.0f, 0, 0, 0, 0, 0, 0}))),
    });
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _HostGeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToRenderer(geometryBuffers, visibleWorldRect);

    auto cellData = geometryBuffers->getCellData();
    ASSERT_EQ(2u, cellData.size());
    EXPECT_NEAR(0.5f, getVertexNextTo(cellData, {100.0f, 100.0f}).activity, 1.0f / 255);
    EXPECT_FLOAT_EQ(0.0f, getVertexNextTo(cellData, {200.0f, 100.0f}).activity);
}

TEST_F(GeometryTests, copyBuffers_activityIsFullWhileCellIsAttacked)
{
    auto data = ContentDesc().addCreature({
        ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().event(CellEvent_Attacked).eventCounter(10)),
        ObjectDesc().id(2).pos({200.0f, 100.0f}).type(CellDesc().event(CellEvent_Attacked).eventCounter(0)),
    });
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _HostGeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToRenderer(geometryBuffers, visibleWorldRect);

    auto cellData = geometryBuffers->getCellData();
    ASSERT_EQ(2u, cellData.size());
    EXPECT_FLOAT_EQ(1.0f, getVertexNextTo(cellData, {100.0f, 100.0f}).activity);
    EXPECT_FLOAT_EQ(0.0f, getVertexNextTo(cellData, {200.0f, 100.0f}).activity);
}

TEST_F(GeometryTests, copyBuffers_activitySmoothsAlternatingSignals)
{
    // By default, connected cells swap their signals with every cell function pass
    auto data = ContentDesc()
                    .addCreature({
                        ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().neuralActivity(NeuralActivityDesc().signals({2.0f, 0, 0, 0, 0, 0, 0, 0}))),
                        ObjectDesc().id(2).pos({101.0f, 100.0f}).type(CellDesc().neuralActivity(NeuralActivityDesc().signals({0, 0, 0, 0, 0, 0, 0, 0}))),
                    })
                    .addConnection(1, 2);
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _HostGeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    for (int i = 0; i < 30; ++i) {
        _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);
        _simulationFacade->tryCopyBuffersFromCudaToRenderer(geometryBuffers, visibleWorldRect);
    }

    auto cellData = geometryBuffers->getCellData();
    ASSERT_EQ(2u, cellData.size());
    for (auto const& cell : cellData) {
        EXPECT_GT(cell.activity, 0.3f);
        EXPECT_LT(cell.activity, 0.7f);
    }
}

TEST_F(GeometryTests, copyBuffers_creature)
{
    auto data = ContentDesc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
            ObjectDesc().id(3).pos({102.0f, 100.0f}),
        },
        CreatureDesc().id(1));
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _HostGeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToRenderer(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(3u, numObjects.objects);
    EXPECT_EQ(4u, numObjects.lineIndices);
    EXPECT_EQ(0u, numObjects.triangleIndices);

    // Verify buffer entries
    auto cellData = geometryBuffers->getCellData();
    EXPECT_EQ(3u, cellData.size());

    auto lines = geometryBuffers->getLineIndices();
    EXPECT_EQ(4u, lines.size());
}

TEST_F(GeometryTests, copyBuffers_shadowGridCollectsFluidsOutsideVisibleRect)
{
    auto data = ContentDesc().objects({
        ObjectDesc().id(1).pos({5.0f, 5.0f}).type(FluidDesc()),
        ObjectDesc().id(2).pos({50.5f, 20.25f}).type(FluidDesc()),
    });
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _HostGeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {10, 10}};
    ShadowGridLayout shadowGridLayout{.origin = {0, 0}, .nodeDistance = 1.0f, .size = {100, 100}};

    _simulationFacade->tryCopyBuffersFromCudaToRenderer(geometryBuffers, visibleWorldRect, shadowGridLayout);

    EXPECT_EQ(10000u, geometryBuffers->getNumObjects().shadowGridNodes);
    auto nodes = geometryBuffers->getShadowGridData();
    ASSERT_EQ(10000u, nodes.size());
    EXPECT_NEAR(1.0f, sumShadowGrid(nodes, [](auto const& node) { return node.numFluids; }), 0.01f);

    // The fluid at (50.5, 20.25) is distributed bilinearly over the four nodes around it
    auto getFluids = [&](int x, int y) { return static_cast<float>(nodes.at(y * 100 + x).numFluids) / ShadowGridNodeData::FixedPointScale; };
    EXPECT_NEAR(0.375f, getFluids(50, 20), 0.01f);
    EXPECT_NEAR(0.375f, getFluids(51, 20), 0.01f);
    EXPECT_NEAR(0.125f, getFluids(50, 21), 0.01f);
    EXPECT_NEAR(0.125f, getFluids(51, 21), 0.01f);
}

TEST_F(GeometryTests, copyBuffers_shadowGridCoversCellsWithTheirStrands)
{
    auto data = ContentDesc().addCreature({
        ObjectDesc().id(1).pos({50.0f, 50.0f}),
        ObjectDesc().id(2).pos({51.0f, 50.0f}),
    });
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _HostGeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {10, 10}};
    ShadowGridLayout shadowGridLayout{.origin = {0, 0}, .nodeDistance = 1.0f, .size = {100, 100}};

    _simulationFacade->tryCopyBuffersFromCudaToRenderer(geometryBuffers, visibleWorldRect, shadowGridLayout);

    // Each cell covers its disk and half of the strand of length 1
    auto nodes = geometryBuffers->getShadowGridData();
    auto tissueArea = sumShadowGrid(nodes, [](auto const& node) { return node.tissueArea; });
    auto cellRadius = RayTracedObjectSizes::CellRadius;
    EXPECT_NEAR(2.0f * (Const::Pi * cellRadius * cellRadius + RayTracedObjectSizes::StrandHalfWidth), tissueArea, 0.01f);
    EXPECT_FLOAT_EQ(0.0f, sumShadowGrid(nodes, [](auto const& node) { return node.numFluids; }));
    EXPECT_FLOAT_EQ(0.0f, sumShadowGrid(nodes, [](auto const& node) { return node.solidArea; }));

    // The strongest component of the hue is 1
    auto maxHueArea = std::max({
        sumShadowGrid(nodes, [](auto const& node) { return node.tissueHueArea[0]; }),
        sumShadowGrid(nodes, [](auto const& node) { return node.tissueHueArea[1]; }),
        sumShadowGrid(nodes, [](auto const& node) { return node.tissueHueArea[2]; }),
    });
    EXPECT_NEAR(tissueArea, maxHueArea, 0.01f);
}

TEST_F(GeometryTests, copyBuffers_shadowGridCollectsPeriodicCopies)
{
    auto data = ContentDesc().objects({ObjectDesc().id(1).pos({995.5f, 100.25f}).type(FluidDesc())});
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _HostGeometryBuffers::create();
    RealRect visibleWorldRect{{100, 100}, {110, 110}};
    ShadowGridLayout shadowGridLayout{.origin = {-20, 90}, .nodeDistance = 1.0f, .size = {60, 30}};

    _simulationFacade->tryCopyBuffersFromCudaToRenderer(geometryBuffers, visibleWorldRect, shadowGridLayout);

    // The fluid at the right border of the world is collected left of the world at (-4.5, 100.25)
    auto nodes = geometryBuffers->getShadowGridData();
    EXPECT_NEAR(1.0f, sumShadowGrid(nodes, [](auto const& node) { return node.numFluids; }), 0.01f);
    EXPECT_NEAR(0.375f, static_cast<float>(nodes.at(10 * 60 + 15).numFluids) / ShadowGridNodeData::FixedPointScale, 0.01f);
}

TEST_F(GeometryTests, copyBuffers_shadowGridCoversSolids)
{
    auto data = ContentDesc().objects({ObjectDesc().id(1).pos({50.0f, 50.0f}).type(SolidDesc())});
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _HostGeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {10, 10}};
    ShadowGridLayout shadowGridLayout{.origin = {0, 0}, .nodeDistance = 1.0f, .size = {100, 100}};

    _simulationFacade->tryCopyBuffersFromCudaToRenderer(geometryBuffers, visibleWorldRect, shadowGridLayout);

    auto nodes = geometryBuffers->getShadowGridData();
    auto solidRadius = RayTracedObjectSizes::SolidRadius;
    EXPECT_NEAR(Const::Pi * solidRadius * solidRadius, sumShadowGrid(nodes, [](auto const& node) { return node.solidArea; }), 0.01f);
    EXPECT_FLOAT_EQ(0.0f, sumShadowGrid(nodes, [](auto const& node) { return node.tissueArea; }));
}

TEST_F(GeometryTests, copyBuffers_shadowGridWithoutNodes)
{
    auto data = ContentDesc().objects({ObjectDesc().id(1).pos({50.0f, 50.0f}).type(FluidDesc())});
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _HostGeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {10, 10}};

    _simulationFacade->tryCopyBuffersFromCudaToRenderer(geometryBuffers, visibleWorldRect);

    EXPECT_EQ(0u, geometryBuffers->getNumObjects().shadowGridNodes);
    EXPECT_TRUE(geometryBuffers->getShadowGridData().empty());
}
