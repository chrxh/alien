#include <gtest/gtest.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <Base/GlobalSettings.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/GeometryBuffers.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class GeometryTests : public IntegrationTestFramework
{
public:
    GeometryTests()
        : IntegrationTestFramework()
    {
        glfwInit();
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        _window = glfwCreateWindow(100, 100, "Test", nullptr, nullptr);
        if (_window) {
            glfwMakeContextCurrent(_window);
            gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress));
        }
    }

    ~GeometryTests()
    {
        if (_window) {
            glfwDestroyWindow(_window);
        }
        glfwTerminate();
    }

protected:
    GLFWwindow* _window = nullptr;
};

TEST_F(GeometryTests, copyBuffers_emptySim)
{
    _simulationFacade->clear();
    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(0u, numObjects.cells);
    EXPECT_EQ(0u, numObjects.energyParticles);
    EXPECT_EQ(0u, numObjects.lineIndices);
    EXPECT_EQ(0u, numObjects.triangleIndices);
}

TEST_F(GeometryTests, copyBuffers_singleCell)
{
    auto data = Description().cells({CellDescription().id(1).pos({100.0f, 100.0f})});
    _simulationFacade->setSimulationData(data);

    // Select the cell to enable selected object data extraction
    _simulationFacade->setSelection({99.0f, 99.0f}, {101.0f, 101.0f});

    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(1u, numObjects.cells);
    EXPECT_EQ(0u, numObjects.energyParticles);

    // Verify buffer entries via downloadSelectedObjectData
    auto selectedObjects = geometryBuffers->downloadSelectedObjectData();
    EXPECT_EQ(1u, selectedObjects.size());
}

TEST_F(GeometryTests, copyBuffers_multipleCells)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}),
        CellDescription().id(2).pos({101.0f, 100.0f}),
        CellDescription().id(3).pos({102.0f, 100.0f}),
    });
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(3u, numObjects.cells);
}

TEST_F(GeometryTests, copyBuffers_singleParticle)
{
    auto data = Description().particles({ParticleDescription().id(1).pos({100.0f, 100.0f}).energy(10.0f)});
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(1u, numObjects.energyParticles);
}

TEST_F(GeometryTests, copyBuffers_multipleParticles)
{
    auto data = Description().particles({
        ParticleDescription().id(1).pos({100.0f, 100.0f}).energy(10.0f),
        ParticleDescription().id(2).pos({101.0f, 100.0f}).energy(10.0f),
        ParticleDescription().id(3).pos({102.0f, 100.0f}).energy(10.0f),
        ParticleDescription().id(4).pos({103.0f, 100.0f}).energy(10.0f),
    });
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(4u, numObjects.energyParticles);
}

TEST_F(GeometryTests, copyBuffers_cellsWithConnections)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    // Select both cells to enable connection data extraction
    _simulationFacade->setSelection({99.0f, 99.0f}, {102.0f, 101.0f});

    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(2u, numObjects.cells);
    EXPECT_EQ(2u, numObjects.lineIndices);

    // Inspect connection data via downloadSelectedConnectionData
    auto connectionArrows = geometryBuffers->downloadSelectedConnectionData();
    EXPECT_EQ(2u, connectionArrows.size());  // 2 vertices per connection line
}

TEST_F(GeometryTests, copyBuffers_triangularCluster)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}),
        CellDescription().id(2).pos({101.0f, 100.0f}),
        CellDescription().id(3).pos({100.5f, 100.866f}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 1);
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(3u, numObjects.cells);
    EXPECT_EQ(6u, numObjects.lineIndices);
    EXPECT_EQ(6u, numObjects.triangleIndices);
}

TEST_F(GeometryTests, copyBuffers_mixedCellsAndParticles)
{
    auto data = Description()
                    .cells({
                        CellDescription().id(1).pos({100.0f, 100.0f}),
                        CellDescription().id(2).pos({101.0f, 100.0f}),
                    })
                    .particles({
                        ParticleDescription().id(3).pos({200.0f, 200.0f}).energy(10.0f),
                        ParticleDescription().id(4).pos({201.0f, 200.0f}).energy(10.0f),
                        ParticleDescription().id(5).pos({202.0f, 200.0f}).energy(10.0f),
                    });
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(2u, numObjects.cells);
    EXPECT_EQ(3u, numObjects.energyParticles);
}

TEST_F(GeometryTests, copyBuffers_creature)
{
    auto data = Description().addCreature(CreatureDescription().id(1).cells({
        CellDescription().id(1).pos({100.0f, 100.0f}),
        CellDescription().id(2).pos({101.0f, 100.0f}),
        CellDescription().id(3).pos({102.0f, 100.0f}),
    }));
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(3u, numObjects.cells);
    EXPECT_EQ(4u, numObjects.lineIndices);
}

// Signal restriction tests for cudaExtractSelectedObjectData

TEST_F(GeometryTests, selectedObjectData_noRestriction_inactive)
{
    auto cell = CellDescription()
                    .id(1)
                    .pos({100.0f, 100.0f})
                    .signalRestriction(SignalRestrictionDescription().mode(SignalRestrictionMode_Inactive).baseAngle(45.0f).openingAngle(90.0f));

    auto data = Description().cells({
        cell,
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    // Select cell 1 using position-based selection
    _simulationFacade->setSelection({99.0f, 99.0f}, {100.5f, 101.0f});

    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};
    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto selectedObjects = geometryBuffers->downloadSelectedObjectData();
    ASSERT_EQ(1u, selectedObjects.size());
    EXPECT_EQ(0, selectedObjects[0].hasSignalRestriction);  // Inactive mode = no restriction
}

TEST_F(GeometryTests, selectedObjectData_hasRestriction_active)
{
    auto cell = CellDescription()
                    .id(1)
                    .pos({100.0f, 100.0f})
                    .signalRestriction(SignalRestrictionDescription().mode(SignalRestrictionMode_Active).baseAngle(45.0f).openingAngle(90.0f));

    auto data = Description().cells({
        cell,
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    // Select cell 1 using position-based selection
    _simulationFacade->setSelection({99.0f, 99.0f}, {100.5f, 101.0f});

    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};
    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto selectedObjects = geometryBuffers->downloadSelectedObjectData();
    ASSERT_EQ(1u, selectedObjects.size());
    EXPECT_EQ(1, selectedObjects[0].hasSignalRestriction);  // Active mode = has restriction
}

TEST_F(GeometryTests, selectedObjectData_hasRestriction_conditional)
{
    auto cell = CellDescription()
                    .id(1)
                    .pos({100.0f, 100.0f})
                    .signalRestriction(SignalRestrictionDescription().mode(SignalRestrictionMode_Conditional).baseAngle(45.0f).openingAngle(90.0f));

    auto data = Description().cells({
        cell,
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    // Select cell 1 using position-based selection
    _simulationFacade->setSelection({99.0f, 99.0f}, {100.5f, 101.0f});

    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};
    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto selectedObjects = geometryBuffers->downloadSelectedObjectData();
    ASSERT_EQ(1u, selectedObjects.size());
    EXPECT_EQ(1, selectedObjects[0].hasSignalRestriction);  // Conditional mode = has restriction
}

// Signal restriction tests for cudaExtractSelectedConnectionData

TEST_F(GeometryTests, connectionData_noRestriction_inactive_bothDirections)
{
    auto cell1 = CellDescription()
                     .id(1)
                     .pos({100.0f, 100.0f})
                     .signalRestriction(SignalRestrictionDescription().mode(SignalRestrictionMode_Inactive));

    auto cell2 = CellDescription()
                     .id(2)
                     .pos({101.0f, 100.0f})
                     .signalRestriction(SignalRestrictionDescription().mode(SignalRestrictionMode_Inactive));

    auto data = Description().cells({cell1, cell2});
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    // Select both cells using position-based selection
    _simulationFacade->setSelection({99.0f, 99.0f}, {102.0f, 101.0f});

    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};
    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto connectionArrows = geometryBuffers->downloadSelectedConnectionData();
    ASSERT_EQ(2u, connectionArrows.size());  // 2 vertices per connection line
    // arrowFlags: bit 0 = arrow to cell1, bit 1 = arrow to cell2
    // Both cells have no restriction, so signals can flow both ways (flags = 3)
    EXPECT_EQ(3, connectionArrows[0].arrowFlags);
    EXPECT_EQ(3, connectionArrows[1].arrowFlags);
}

TEST_F(GeometryTests, connectionData_withRestriction_active_restrictedDirection)
{
    // Use baseAngle = 90 and openingAngle = 90 to point away from connection
    // Connection angle is 0 (first connection), so range [45+180, 135+180] = [225, 315] doesn't include 0
    auto cell1 = CellDescription()
                     .id(1)
                     .pos({100.0f, 100.0f})
                     .signalRestriction(SignalRestrictionDescription().mode(SignalRestrictionMode_Active).baseAngle(90.0f).openingAngle(90.0f));

    auto cell2 = CellDescription()
                     .id(2)
                     .pos({101.0f, 100.0f})
                     .signalRestriction(SignalRestrictionDescription().mode(SignalRestrictionMode_Inactive));

    auto data = Description().cells({cell1, cell2});
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    // Select both cells using position-based selection
    _simulationFacade->setSelection({99.0f, 99.0f}, {102.0f, 101.0f});

    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};
    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto connectionArrows = geometryBuffers->downloadSelectedConnectionData();
    ASSERT_EQ(2u, connectionArrows.size());
    // Cell1 has restriction that blocks signal to cell2 (connection angle 0 is outside range [225,315])
    // Cell2 has no restriction, so signal can flow to cell1
    // Expected: arrow to cell1 (bit 0 = 1), no arrow to cell2 (bit 1 = 0) => flags = 1
    EXPECT_EQ(1, connectionArrows[0].arrowFlags);
    EXPECT_EQ(1, connectionArrows[1].arrowFlags);
}

TEST_F(GeometryTests, connectionData_withRestriction_conditional_restrictedDirection)
{
    // Use baseAngle = 90 and openingAngle = 90 to point away from connection
    auto cell1 = CellDescription()
                     .id(1)
                     .pos({100.0f, 100.0f})
                     .signalRestriction(SignalRestrictionDescription().mode(SignalRestrictionMode_Conditional).baseAngle(90.0f).openingAngle(90.0f));

    auto cell2 = CellDescription()
                     .id(2)
                     .pos({101.0f, 100.0f})
                     .signalRestriction(SignalRestrictionDescription().mode(SignalRestrictionMode_Inactive));

    auto data = Description().cells({cell1, cell2});
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    // Select both cells using position-based selection
    _simulationFacade->setSelection({99.0f, 99.0f}, {102.0f, 101.0f});

    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};
    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto connectionArrows = geometryBuffers->downloadSelectedConnectionData();
    ASSERT_EQ(2u, connectionArrows.size());
    // Conditional mode should render the same as Active mode for arrow directions
    // Cell1 has restriction that blocks signal to cell2
    // Cell2 has no restriction, so signal can flow to cell1
    // Expected: arrow to cell1 (bit 0 = 1), no arrow to cell2 (bit 1 = 0) => flags = 1
    EXPECT_EQ(1, connectionArrows[0].arrowFlags);
    EXPECT_EQ(1, connectionArrows[1].arrowFlags);
}
