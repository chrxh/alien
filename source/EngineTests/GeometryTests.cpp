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
    {}
};

TEST_F(GeometryTests, getNumRenderObjects_emptySim)
{
    _simulationFacade->clear();

    auto numObjects = _simulationFacade->testOnly_getNumRenderObjects();

    EXPECT_EQ(0u, numObjects.cells);
    EXPECT_EQ(0u, numObjects.energyParticles);
    EXPECT_EQ(0u, numObjects.lineIndices);
    EXPECT_EQ(0u, numObjects.triangleIndices);
}

TEST_F(GeometryTests, getNumRenderObjects_singleCell)
{
    auto data = Description().cells({CellDescription().id(1).pos({100.0f, 100.0f})});
    _simulationFacade->setSimulationData(data);

    auto numObjects = _simulationFacade->testOnly_getNumRenderObjects();

    EXPECT_EQ(1u, numObjects.cells);
    EXPECT_EQ(0u, numObjects.energyParticles);
}

TEST_F(GeometryTests, getNumRenderObjects_multipleCells)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}),
        CellDescription().id(2).pos({101.0f, 100.0f}),
        CellDescription().id(3).pos({102.0f, 100.0f}),
    });
    _simulationFacade->setSimulationData(data);

    auto numObjects = _simulationFacade->testOnly_getNumRenderObjects();

    EXPECT_EQ(3u, numObjects.cells);
}

TEST_F(GeometryTests, getNumRenderObjects_singleParticle)
{
    auto data = Description().particles({ParticleDescription().id(1).pos({100.0f, 100.0f}).energy(10.0f)});
    _simulationFacade->setSimulationData(data);

    auto numObjects = _simulationFacade->testOnly_getNumRenderObjects();

    EXPECT_EQ(0u, numObjects.cells);
    EXPECT_EQ(1u, numObjects.energyParticles);
}

TEST_F(GeometryTests, getNumRenderObjects_multipleParticles)
{
    auto data = Description().particles({
        ParticleDescription().id(1).pos({100.0f, 100.0f}).energy(10.0f),
        ParticleDescription().id(2).pos({101.0f, 100.0f}).energy(10.0f),
        ParticleDescription().id(3).pos({102.0f, 100.0f}).energy(10.0f),
        ParticleDescription().id(4).pos({103.0f, 100.0f}).energy(10.0f),
    });
    _simulationFacade->setSimulationData(data);

    auto numObjects = _simulationFacade->testOnly_getNumRenderObjects();

    EXPECT_EQ(4u, numObjects.energyParticles);
}

TEST_F(GeometryTests, getNumRenderObjects_cellsWithConnections)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    auto numObjects = _simulationFacade->testOnly_getNumRenderObjects();

    EXPECT_EQ(2u, numObjects.cells);
    EXPECT_EQ(2u, numObjects.lineIndices);
}

TEST_F(GeometryTests, getNumRenderObjects_triangularCluster)
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

    auto numObjects = _simulationFacade->testOnly_getNumRenderObjects();

    EXPECT_EQ(3u, numObjects.cells);
    EXPECT_EQ(6u, numObjects.lineIndices);
    EXPECT_EQ(6u, numObjects.triangleIndices);
}

TEST_F(GeometryTests, getNumRenderObjects_mixedCellsAndParticles)
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

    auto numObjects = _simulationFacade->testOnly_getNumRenderObjects();

    EXPECT_EQ(2u, numObjects.cells);
    EXPECT_EQ(3u, numObjects.energyParticles);
}

TEST_F(GeometryTests, getNumRenderObjects_creature)
{
    auto data = Description().addCreature(CreatureDescription().id(1).cells({
        CellDescription().id(1).pos({100.0f, 100.0f}),
        CellDescription().id(2).pos({101.0f, 100.0f}),
        CellDescription().id(3).pos({102.0f, 100.0f}),
    }));
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    _simulationFacade->setSimulationData(data);

    auto numObjects = _simulationFacade->testOnly_getNumRenderObjects();

    EXPECT_EQ(3u, numObjects.cells);
    EXPECT_EQ(4u, numObjects.lineIndices);
}

class GeometryTests_CopyBuffers
    : public IntegrationTestFramework
    , public testing::WithParamInterface<bool>
{
public:
    GeometryTests_CopyBuffers()
        : IntegrationTestFramework()
    {
        GlobalSettings::get().setNoInterop(GetParam());

        glfwInit();
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        _window = glfwCreateWindow(100, 100, "Test", nullptr, nullptr);
        if (_window) {
            glfwMakeContextCurrent(_window);
            gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress));
        }
    }

    ~GeometryTests_CopyBuffers()
    {
        GlobalSettings::get().setNoInterop(false);
        if (_window) {
            glfwDestroyWindow(_window);
        }
        glfwTerminate();
    }

protected:
    GLFWwindow* _window = nullptr;
};

INSTANTIATE_TEST_SUITE_P(InteropModes, GeometryTests_CopyBuffers, ::testing::Values(false, true));

TEST_P(GeometryTests_CopyBuffers, copyBuffers_emptySim)
{
    if (!GetParam()) {
        GTEST_SKIP() << "Interop mode requires display with CUDA-OpenGL support";
    }
    _simulationFacade->clear();
    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(0u, numObjects.cells);
    EXPECT_EQ(0u, numObjects.energyParticles);
}

TEST_P(GeometryTests_CopyBuffers, copyBuffers_singleCell)
{
    if (!GetParam()) {
        GTEST_SKIP() << "Interop mode requires display with CUDA-OpenGL support";
    }
    auto data = Description().cells({CellDescription().id(1).pos({100.0f, 100.0f})});
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(1u, numObjects.cells);
}

TEST_P(GeometryTests_CopyBuffers, copyBuffers_multipleCells)
{
    if (!GetParam()) {
        GTEST_SKIP() << "Interop mode requires display with CUDA-OpenGL support";
    }
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

TEST_P(GeometryTests_CopyBuffers, copyBuffers_singleParticle)
{
    if (!GetParam()) {
        GTEST_SKIP() << "Interop mode requires display with CUDA-OpenGL support";
    }
    auto data = Description().particles({ParticleDescription().id(1).pos({100.0f, 100.0f}).energy(10.0f)});
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(1u, numObjects.energyParticles);
}

TEST_P(GeometryTests_CopyBuffers, copyBuffers_cellsWithConnections)
{
    if (!GetParam()) {
        GTEST_SKIP() << "Interop mode requires display with CUDA-OpenGL support";
    }
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(2u, numObjects.cells);
    EXPECT_EQ(2u, numObjects.lineIndices);
}

TEST_P(GeometryTests_CopyBuffers, copyBuffers_mixedCellsAndParticles)
{
    if (!GetParam()) {
        GTEST_SKIP() << "Interop mode requires display with CUDA-OpenGL support";
    }
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
