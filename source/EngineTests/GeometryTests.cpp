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
    EXPECT_EQ(0u, numObjects.objects);
    EXPECT_EQ(0u, numObjects.energies);
    EXPECT_EQ(0u, numObjects.lineIndices);
    EXPECT_EQ(0u, numObjects.triangleIndices);
}

TEST_F(GeometryTests, copyBuffers_objects)
{
    auto data = Description().objects({
        ObjectDescription().id(1).pos({100.0f, 100.0f}),
        ObjectDescription().id(2).pos({101.0f, 100.0f}),
        ObjectDescription().id(3).pos({102.0f, 100.0f}),
    });
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(3u, numObjects.objects);
    EXPECT_EQ(0u, numObjects.energies);

    // Verify buffer entries
    auto cellData = geometryBuffers->getCellData();
    EXPECT_EQ(3u, cellData.size());
}

TEST_F(GeometryTests, copyBuffers_energyParticles)
{
    auto data = Description().energies({
        EnergyDescription().id(1).pos({100.0f, 100.0f}).energy(10.0f),
        EnergyDescription().id(2).pos({101.0f, 100.0f}).energy(10.0f),
        EnergyDescription().id(3).pos({102.0f, 100.0f}).energy(10.0f),
        EnergyDescription().id(4).pos({103.0f, 100.0f}).energy(10.0f),
    });
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(4u, numObjects.energies);

    // Verify buffer entries
    auto particleData = geometryBuffers->getEnergyParticleData();
    EXPECT_EQ(4u, particleData.size());
}

TEST_F(GeometryTests, copyBuffers_cellsWithConnections)
{
    auto data = Description().objects({
        ObjectDescription().id(1).pos({100.0f, 100.0f}),
        ObjectDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(2u, numObjects.objects);
    EXPECT_EQ(2u, numObjects.lineIndices);

    // Verify buffer entries
    auto lines = geometryBuffers->getLineIndices();
    EXPECT_EQ(2u, lines.size());
}

TEST_F(GeometryTests, copyBuffers_triangle)
{
    auto data = Description().objects({
        ObjectDescription().id(1).pos({100.0f, 100.0f}),
        ObjectDescription().id(2).pos({101.0f, 100.0f}),
        ObjectDescription().id(3).pos({100.5f, 100.866f}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 1);
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

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
    auto data = Description().objects({
        ObjectDescription().id(1).pos({100.0f, 100.0f}),
        ObjectDescription().id(2).pos({101.0f, 100.0f}),
        ObjectDescription().id(3).pos({101.0f, 101.0f}),
        ObjectDescription().id(4).pos({100.0f, 101.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);
    data.addConnection(4, 1);
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

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

TEST_F(GeometryTests, copyBuffers_mixedCellsAndParticles)
{
    auto data = Description()
                    .objects({
                        ObjectDescription().id(1).pos({100.0f, 100.0f}),
                        ObjectDescription().id(2).pos({101.0f, 100.0f}),
                    })
                    .energies({
                        EnergyDescription().id(3).pos({200.0f, 200.0f}).energy(10.0f),
                        EnergyDescription().id(4).pos({201.0f, 200.0f}).energy(10.0f),
                        EnergyDescription().id(5).pos({202.0f, 200.0f}).energy(10.0f),
                    });
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(2u, numObjects.objects);
    EXPECT_EQ(3u, numObjects.energies);

    // Verify buffer entries
    auto particleData = geometryBuffers->getEnergyParticleData();
    EXPECT_EQ(3u, particleData.size());

    auto cellData = geometryBuffers->getCellData();
    EXPECT_EQ(2u, cellData.size());
}

TEST_F(GeometryTests, copyBuffers_creature)
{
    auto data = Description().addCreature({
        ObjectDescription().id(1).pos({100.0f, 100.0f}),
        ObjectDescription().id(2).pos({101.0f, 100.0f}),
        ObjectDescription().id(3).pos({102.0f, 100.0f}),
    }, CreatureDescription().id(1));
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

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

TEST_F(GeometryTests, copyBuffers_selectedObjectData_noRestriction_inactive)
{
    auto object = ObjectDescription()
                    .id(1)
                    .pos({100.0f, 100.0f})
                    .signalRestriction(SignalRestrictionDescription().mode(SignalRestrictionMode_Inactive).baseAngle(45.0f).openingAngle(90.0f));

    auto data = Description().objects({
        object,
        ObjectDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    // Select cell 1 using position-based selection
    _simulationFacade->setSelection({99.0f, 99.0f}, {100.5f, 101.0f});

    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};
    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto selectedObjects = geometryBuffers->getSelectedObjectData();
    ASSERT_EQ(1u, selectedObjects.size());
    EXPECT_EQ(0, selectedObjects[0].hasSignalRestriction);  // Inactive mode = no restriction
}

TEST_F(GeometryTests, copyBuffers_selectedObjectData_hasRestriction_active)
{
    auto object = ObjectDescription()
                    .id(1)
                    .pos({100.0f, 100.0f})
                    .signalRestriction(SignalRestrictionDescription().mode(SignalRestrictionMode_Active).baseAngle(45.0f).openingAngle(90.0f));

    auto data = Description().objects({
        object,
        ObjectDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    // Select cell 1 using position-based selection
    _simulationFacade->setSelection({99.0f, 99.0f}, {100.5f, 101.0f});

    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};
    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto selectedObjects = geometryBuffers->getSelectedObjectData();
    ASSERT_EQ(1u, selectedObjects.size());
    EXPECT_EQ(1, selectedObjects[0].hasSignalRestriction);  // Active mode = has restriction
}

TEST_F(GeometryTests, copyBuffers_selectedObjectData_hasRestriction_conditional)
{
    auto object = ObjectDescription()
                    .id(1)
                    .pos({100.0f, 100.0f})
                    .signalRestriction(SignalRestrictionDescription().mode(SignalRestrictionMode_Conditional).baseAngle(45.0f).openingAngle(90.0f));

    auto data = Description().objects({
        object,
        ObjectDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    // Select cell 1 using position-based selection
    _simulationFacade->setSelection({99.0f, 99.0f}, {100.5f, 101.0f});

    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};
    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto selectedObjects = geometryBuffers->getSelectedObjectData();
    ASSERT_EQ(1u, selectedObjects.size());
    EXPECT_EQ(1, selectedObjects[0].hasSignalRestriction);  // Conditional mode = has restriction
}

TEST_F(GeometryTests, copyBuffers_connectionData_noRestriction_inactive_bothDirections)
{
    auto object1 = ObjectDescription()
                     .id(1)
                     .pos({100.0f, 100.0f})
                     .signalRestriction(SignalRestrictionDescription().mode(SignalRestrictionMode_Inactive));

    auto object2 = ObjectDescription()
                     .id(2)
                     .pos({101.0f, 100.0f})
                     .signalRestriction(SignalRestrictionDescription().mode(SignalRestrictionMode_Inactive));

    auto data = Description().objects({object1, object2});
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    // Select both cells using position-based selection
    _simulationFacade->setSelection({99.0f, 99.0f}, {102.0f, 101.0f});

    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};
    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto connectionArrows = geometryBuffers->getSelectedConnectionData();
    ASSERT_EQ(2u, connectionArrows.size());  // 2 vertices per connection line
    // arrowFlags: bit 0 = arrow to object1, bit 1 = arrow to object2
    // Both cells have no restriction, so signals can flow both ways (flags = 3)
    EXPECT_EQ(3, connectionArrows[0].arrowFlags);
    EXPECT_EQ(3, connectionArrows[1].arrowFlags);
}

TEST_F(GeometryTests, copyBuffers_connectionData_withRestriction_active_restrictedDirection)
{
    // Use baseAngle = 90 and openingAngle = 90 to point away from connection
    // Connection angle is 0 (first connection), so range [45+180, 135+180] = [225, 315] doesn't include 0
    auto object1 = ObjectDescription()
                     .id(1)
                     .pos({100.0f, 100.0f})
                     .signalRestriction(SignalRestrictionDescription().mode(SignalRestrictionMode_Active).baseAngle(90.0f).openingAngle(90.0f));

    auto object2 = ObjectDescription()
                     .id(2)
                     .pos({101.0f, 100.0f})
                     .signalRestriction(SignalRestrictionDescription().mode(SignalRestrictionMode_Inactive));

    auto data = Description().objects({object1, object2});
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    // Select both cells using position-based selection
    _simulationFacade->setSelection({99.0f, 99.0f}, {102.0f, 101.0f});

    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};
    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto connectionArrows = geometryBuffers->getSelectedConnectionData();
    ASSERT_EQ(2u, connectionArrows.size());
    // Cell1 has restriction that blocks signal to object2 (connection angle 0 is outside range [225,315])
    // Cell2 has no restriction, so signal can flow to object1
    // Expected: arrow to object1 (bit 0 = 1), no arrow to object2 (bit 1 = 0) => flags = 1
    EXPECT_EQ(1, connectionArrows[0].arrowFlags);
    EXPECT_EQ(1, connectionArrows[1].arrowFlags);
}

TEST_F(GeometryTests, copyBuffers_connectionData_withRestriction_conditional_restrictedDirection)
{
    // Use baseAngle = 90 and openingAngle = 90 to point away from connection
    auto object1 = ObjectDescription()
                     .id(1)
                     .pos({100.0f, 100.0f})
                     .signalRestriction(SignalRestrictionDescription().mode(SignalRestrictionMode_Conditional).baseAngle(90.0f).openingAngle(90.0f));

    auto object2 = ObjectDescription()
                     .id(2)
                     .pos({101.0f, 100.0f})
                     .signalRestriction(SignalRestrictionDescription().mode(SignalRestrictionMode_Inactive));

    auto data = Description().objects({object1, object2});
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    // Select both cells using position-based selection
    _simulationFacade->setSelection({99.0f, 99.0f}, {102.0f, 101.0f});

    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};
    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto connectionArrows = geometryBuffers->getSelectedConnectionData();
    ASSERT_EQ(2u, connectionArrows.size());
    // Conditional mode should render the same as Active mode for arrow directions
    // Cell1 has restriction that blocks signal to object2
    // Cell2 has no restriction, so signal can flow to object1
    // Expected: arrow to object1 (bit 0 = 1), no arrow to object2 (bit 1 = 0) => flags = 1
    EXPECT_EQ(1, connectionArrows[0].arrowFlags);
    EXPECT_EQ(1, connectionArrows[1].arrowFlags);
}
