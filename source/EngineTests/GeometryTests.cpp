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
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({100.0f, 100.0f}),
        ObjectDesc().id(2).pos({101.0f, 100.0f}),
        ObjectDesc().id(3).pos({102.0f, 100.0f}),
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
    auto data = Desc().energies({
        EnergyDesc().id(1).pos({100.0f, 100.0f}).energy(10.0f),
        EnergyDesc().id(2).pos({101.0f, 100.0f}).energy(10.0f),
        EnergyDesc().id(3).pos({102.0f, 100.0f}).energy(10.0f),
        EnergyDesc().id(4).pos({103.0f, 100.0f}).energy(10.0f),
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
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({100.0f, 100.0f}),
        ObjectDesc().id(2).pos({101.0f, 100.0f}),
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
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({100.0f, 100.0f}),
        ObjectDesc().id(2).pos({101.0f, 100.0f}),
        ObjectDesc().id(3).pos({100.5f, 100.866f}),
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
    auto data = Desc().addCreature({
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
    auto data = Desc()
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
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({100.0f, 100.0f}),
        ObjectDesc().id(2).pos({101.0f, 100.0f}),
        ObjectDesc().id(3).pos({102.0f, 100.0f}),
    }, CreatureDesc().id(1));
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

