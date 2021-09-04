#pragma once

#include "EngineImpl/Definitions.h"

struct GLFWwindow;

class Shader;

class MainWindow
{
public:
    GLFWwindow* init(SimulationController* simController);
    void mainLoop(GLFWwindow* window);
    void shutdown(GLFWwindow* window);

private:
    void drawMenubar();
    void drawToolbar();

    unsigned int VAO, VBO, EBO;
    Shader* _shader = nullptr;
    void* _cudaResource = nullptr;
    unsigned int _textureId;
    SimulationController* _simController = nullptr;
};