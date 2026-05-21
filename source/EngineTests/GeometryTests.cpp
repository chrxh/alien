#include <gtest/gtest.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <Base/GlobalSettings.h>

#include <Shaders/TriangleFS.h>
#include <Shaders/TriangleGS.h>
#include <Shaders/TriangleVS.h>
#include <EngineInterface/Desc.h>
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
    GLuint createShader(GLenum type, std::string_view source)
    {
        auto shader = glCreateShader(type);
        auto* sourcePtr = source.data();
        auto sourceLength = static_cast<GLint>(source.size());
        glShaderSource(shader, 1, &sourcePtr, &sourceLength);
        glCompileShader(shader);
        return shader;
    }

    GLuint createTriangleShaderProgram()
    {
        auto vertexShader = createShader(GL_VERTEX_SHADER, Shaders::TriangleVS);
        auto geometryShader = createShader(GL_GEOMETRY_SHADER, Shaders::TriangleGS);
        auto fragmentShader = createShader(GL_FRAGMENT_SHADER, Shaders::TriangleFS);

        auto program = glCreateProgram();
        glAttachShader(program, vertexShader);
        glAttachShader(program, geometryShader);
        glAttachShader(program, fragmentShader);
        glLinkProgram(program);

        glDeleteShader(vertexShader);
        glDeleteShader(geometryShader);
        glDeleteShader(fragmentShader);

        return program;
    }

    void setupTriangleVao(GeometryBuffers const& geometryBuffers)
    {
        glBindVertexArray(geometryBuffers->getVaoForTriangles());
        glBindBuffer(GL_ARRAY_BUFFER, geometryBuffers->getVboForObjects());
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(ObjectVertexData), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(ObjectVertexData), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glVertexAttribIPointer(2, 1, GL_INT, sizeof(ObjectVertexData), (void*)(6 * sizeof(float)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(ObjectVertexData), (void*)(6 * sizeof(float) + sizeof(int)));
        glEnableVertexAttribArray(3);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, geometryBuffers->getEboForTriangles());
    }

    int renderTrianglePixels(GeometryBuffers const& geometryBuffers)
    {
        setupTriangleVao(geometryBuffers);

        auto program = createTriangleShaderProgram();
        glUseProgram(program);
        glUniform1f(glGetUniformLocation(program, "zoom"), 50.0f);
        glUniform2f(glGetUniformLocation(program, "worldSize"), 1000.0f, 1000.0f);
        glUniform2f(glGetUniformLocation(program, "rectUpperLeft"), 99.5f, 99.5f);
        glUniform2f(glGetUniformLocation(program, "viewportSize"), 100.0f, 100.0f);

        glViewport(0, 0, 100, 100);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glDrawElements(GL_TRIANGLES, toInt(geometryBuffers->getNumObjects().triangleIndices), GL_UNSIGNED_INT, 0);
        glDisable(GL_DEPTH_TEST);
        glFinish();

        std::vector<unsigned char> pixels(100 * 100 * 3);
        glReadPixels(0, 0, 100, 100, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
        glDeleteProgram(program);

        return std::count_if(pixels.begin(), pixels.end(), [](unsigned char value) { return value > 0; });
    }

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
    EXPECT_EQ(0u, numObjects.fluidParticles);
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
    EXPECT_EQ(0u, numObjects.fluidParticles);

    // Verify buffer entries
    auto cellData = geometryBuffers->getCellData();
    EXPECT_EQ(3u, cellData.size());
}

TEST_F(GeometryTests, copyBuffers_cullsObjectsOutsideVisibleRect)
{
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({5.0f, 5.0f}),
        ObjectDesc().id(2).pos({500.0f, 500.0f}),
    });
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {10, 10}};

    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(1u, numObjects.objects);

    auto cellData = geometryBuffers->getCellData();
    ASSERT_EQ(1u, cellData.size());
    EXPECT_FLOAT_EQ(5.0f, cellData.at(0).pos[0]);
    EXPECT_FLOAT_EQ(5.0f, cellData.at(0).pos[1]);
}

TEST_F(GeometryTests, copyBuffers_fluidParticles)
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
    EXPECT_EQ(4u, numObjects.fluidParticles);

    // Verify buffer entries
    auto particleData = geometryBuffers->getFluidParticleData();
    EXPECT_EQ(4u, particleData.size());
}

TEST_F(GeometryTests, copyBuffers_cullsFluidParticlesOutsideVisibleRect)
{
    auto data = Desc().energies({
        EnergyDesc().id(1).pos({5.0f, 5.0f}).energy(10.0f),
        EnergyDesc().id(2).pos({500.0f, 500.0f}).energy(10.0f),
    });
    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {10, 10}};

    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(1u, numObjects.fluidParticles);

    auto particleData = geometryBuffers->getFluidParticleData();
    ASSERT_EQ(1u, particleData.size());
    EXPECT_FLOAT_EQ(5.0f, particleData.at(0).pos[0]);
    EXPECT_FLOAT_EQ(5.0f, particleData.at(0).pos[1]);
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

TEST_F(GeometryTests, copyBuffers_cullsConnectionsOutsideVisibleRect)
{
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({5.0f, 5.0f}),
        ObjectDesc().id(2).pos({6.0f, 5.0f}),
        ObjectDesc().id(3).pos({500.0f, 500.0f}),
        ObjectDesc().id(4).pos({501.0f, 500.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(3, 4);
    _simulationFacade->setSimulationData(data);

    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {10, 10}};

    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(2u, numObjects.objects);
    EXPECT_EQ(2u, numObjects.lineIndices);

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

TEST_F(GeometryTests, copyBuffers_triangleWithZeroReferenceAngle)
{
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({100.0f, 100.0f}),
        ObjectDesc().id(2).pos({101.0f, 100.0f}),
        ObjectDesc().id(3).pos({100.5f, 100.866f}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 1);
    data.getConnectionRef(1, 2)._angleFromPrevious = 0.0f;
    data.getConnectionRef(1, 3)._angleFromPrevious = 360.0f;

    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    auto numObjects = geometryBuffers->getNumObjects();
    EXPECT_EQ(3u, numObjects.objects);
    EXPECT_EQ(6u, numObjects.lineIndices);
    EXPECT_EQ(6u, numObjects.triangleIndices);

    auto triangles = geometryBuffers->getTriangleIndices();
    EXPECT_EQ(6u, triangles.size());
}

TEST_F(GeometryTests, renderTriangleWithZeroReferenceAngle)
{
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({100.0f, 100.0f}),
        ObjectDesc().id(2).pos({101.0f, 100.0f}),
        ObjectDesc().id(3).pos({100.5f, 100.866f}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 1);
    data.getConnectionRef(1, 2)._angleFromPrevious = 0.0f;
    data.getConnectionRef(1, 3)._angleFromPrevious = 360.0f;

    _simulationFacade->setSimulationData(data);
    auto geometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};

    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    EXPECT_GT(renderTrianglePixels(geometryBuffers), 0);
}

TEST_F(GeometryTests, renderTriangleFanWithZeroReferenceAnglesUsesActualConnectionOrder)
{
    auto createTriangleFan = [] {
        auto result = Desc().addCreature({
            ObjectDesc().id(1).pos({100.0f, 100.0f}),
            ObjectDesc().id(2).pos({99.0f, 100.0f}),
            ObjectDesc().id(3).pos({100.0f, 101.0f}),
            ObjectDesc().id(4).pos({101.0f, 100.0f}),
        });
        result.addConnection(1, 2);
        result.addConnection(1, 3);
        result.addConnection(1, 4);
        result.addConnection(2, 3);
        result.addConnection(3, 4);
        return result;
    };

    auto referenceData = createTriangleFan();
    _simulationFacade->setSimulationData(referenceData);
    auto referenceGeometryBuffers = _GeometryBuffers::create();
    RealRect visibleWorldRect{{0, 0}, {1000, 1000}};
    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(referenceGeometryBuffers, visibleWorldRect);
    auto const referencePixels = renderTrianglePixels(referenceGeometryBuffers);

    auto zeroAngleData = createTriangleFan();
    auto& centerConnections = zeroAngleData.getObjectRef(1)._connections;
    std::swap(centerConnections.at(1), centerConnections.at(2));
    centerConnections.at(0)._angleFromPrevious = 180.0f;
    centerConnections.at(1)._angleFromPrevious = 180.0f;
    centerConnections.at(2)._angleFromPrevious = 0.0f;

    _simulationFacade->setSimulationData(zeroAngleData);
    auto geometryBuffers = _GeometryBuffers::create();
    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);

    EXPECT_EQ(referencePixels, renderTrianglePixels(geometryBuffers));
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
    EXPECT_EQ(3u, numObjects.fluidParticles);

    // Verify buffer entries
    auto particleData = geometryBuffers->getFluidParticleData();
    EXPECT_EQ(3u, particleData.size());

    auto cellData = geometryBuffers->getCellData();
    EXPECT_EQ(2u, cellData.size());
}

TEST_F(GeometryTests, copyBuffers_creature)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
            ObjectDesc().id(3).pos({102.0f, 100.0f}),
        },
        CreatureDesc().id(1));
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
