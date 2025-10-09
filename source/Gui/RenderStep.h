#pragma once

#include "Definitions.h"
#include <filesystem>

class _RenderStep
{
public:
    _RenderStep(std::filesystem::path const& vertexShader, std::filesystem::path const& fragmentShader, std::vector<RenderStep> const& dependentSteps);

    virtual void resize() = 0;
    virtual void execute() = 0;

protected:
    std::vector<RenderStep> _dependentSteps;

    Shader _shader;
    unsigned int _outputTexture = 0;
    unsigned int _fbo = 0;
};

class _PointRenderStep : public _RenderStep
{
public:
    _PointRenderStep(std::filesystem::path const& vertexShader, std::filesystem::path const& fragmentShader, std::vector<RenderStep> const& dependentSteps);

    void resize() override;
    void execute() override;

private:
};

class _PostProcessingRenderStep : public _RenderStep
{
public:
    _PostProcessingRenderStep(std::filesystem::path const& vertexShader, std::filesystem::path const& fragmentShader);

    void resize() override;
    void execute() override;

private:
};
