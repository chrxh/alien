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
    friend _RenderPipeline;

public:
    virtual ~_RenderStep() = default;

    std::vector<RenderStep> const& getDependentSteps() const;
    Shader getShader() const;
    unsigned int getTexture() const;

    void setBool(std::string const& name, bool value);

protected:
    _RenderStep(Shader const& shader, std::vector<RenderStep> const& dependentSteps);

    virtual void resize(IntVector2D const& size);
    virtual void execute(RenderTarget const& target, NumRenderObjects const& numObjects, SimulationFacade const& simulationFacade) = 0;

    void activateShader(SimulationFacade const& simulationFacade);

    std::vector<RenderStep> _dependentSteps;

    Shader _shader;
    bool _outputTextureInitialized = false;
    unsigned int _outputTexture = 0;
    unsigned int _fbo = 0;

    std::map<std::string, bool> _boolValues;
};

class _AbstractPointRenderStep : public _RenderStep 
{
protected:
    _AbstractPointRenderStep(Shader const& shader);

    void execute(RenderTarget const& target, NumRenderObjects const& numObjects, SimulationFacade const& simulationFacade) override;
};

class _PointRenderStep : public _AbstractPointRenderStep
{
public:
    _PointRenderStep(std::filesystem::path const& vertexShader, std::filesystem::path const& fragmentShader);
};

class _SharedVboPointRenderStep : public _AbstractPointRenderStep
{
public:
    _SharedVboPointRenderStep(std::filesystem::path const& vertexShader, std::filesystem::path const& fragmentShader, RenderStep const& sharedStep);
};


class _PostProcessingRenderStep : public _RenderStep
{
public:
    _PostProcessingRenderStep(
        std::filesystem::path const& vertexShader,
        std::filesystem::path const& fragmentShader,
        std::vector<RenderStep> const& dependentSteps);

protected:
    void execute(RenderTarget const& target, NumRenderObjects const& numObjects, SimulationFacade const& simulationFacade) override;
};
