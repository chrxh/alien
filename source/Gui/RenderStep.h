#pragma once

#include "EngineInterface/Definitions.h"

#include "Definitions.h"
#include <filesystem>
#include <variant>

struct TextureTarget
{};
struct ScreenTarget
{
    int fbo = 0;
};
using RenderTarget = std::variant<ScreenTarget, TextureTarget>;

class _RenderStep
{
public:
    _RenderStep(std::filesystem::path const& vertexShader, std::filesystem::path const& fragmentShader, std::vector<RenderStep> const& dependentSteps);
    virtual ~_RenderStep() = default;

    virtual void resize(IntVector2D const& size);
    virtual void execute(RenderTarget const& target, NumRenderObjects const& numObjects, SimulationFacade const& simulationFacade) = 0;

    std::vector<RenderStep> const& getDependentSteps() const;
    Shader getShader() const;
    unsigned int getTexture() const;

protected:
    std::vector<RenderStep> _dependentSteps;

    Shader _shader;
    bool _outputTextureInitialized = false;
    unsigned int _outputTexture = 0;
    unsigned int _fbo = 0;
};

class _PointRenderStep : public _RenderStep 
{
public:
    _PointRenderStep(std::filesystem::path const& vertexShader, std::filesystem::path const& fragmentShader, RenderStep const& dependentStep = nullptr);

    void execute(RenderTarget const& target, NumRenderObjects const& numObjects, SimulationFacade const& simulationFacade) override;

    void setBool(std::string const& name, bool value);

private:
    RenderStep _dependentStep;
    std::map<std::string, bool> _boolValues;
};

class _PostProcessingRenderStep : public _RenderStep
{
public:
    _PostProcessingRenderStep(
        std::filesystem::path const& vertexShader,
        std::filesystem::path const& fragmentShader,
        std::vector<RenderStep> const& dependentSteps);

    void execute(RenderTarget const& target, NumRenderObjects const& numObjects, SimulationFacade const& simulationFacade) override;
};
