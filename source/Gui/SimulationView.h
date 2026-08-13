#pragma once

#include <chrono>
#include <filesystem>

#include <Base/Definitions.h>
#include <Base/Singleton.h>

#include <EngineInterface/Definitions.h>

#include "Definitions.h"

class SimulationView
{
    MAKE_SINGLETON(SimulationView);

public:
    void setup();
    void shutdown();

    void resize(IntVector2D const& viewportSize);

    void draw();
    void processSimulationScrollbars();

    bool isScrollbarDragging() const;

    bool isRenderSimulation() const;
    void setRenderSimulation(bool value);

    bool isOverlayActive() const;
    void setOverlayActive(bool active);

    float getBrightness() const;
    void setBrightness(float value);
    float getContrast() const;
    void setContrast(float value);
    float getMotionBlur() const;
    void setMotionBlur(float value);

    void updateMotionBlur();

    // Renders the currently visible world region offscreen and writes it as PNG file
    void savePicture(std::filesystem::path const& filename, IntVector2D const& resolution);

    static auto constexpr DefaultBrightness = 1.0f;
    static auto constexpr DefaultContrast = 1.0f;
    static auto constexpr DefaultMotionBlur = 0.25f;

private:
    void setupRenderPipeline();

    void markReferenceDomain();

    // Widgets
    SimulationScrollbars _scrollbars;

    // Overlay
    bool _cellDetailOverlayActive = false;

    RenderPipeline _renderPipeline;

    // Screen background texture (dark blue background)
    unsigned int _screenBackgroundTexture;

    bool _areTexturesInitialized = false;

    float _brightness = DefaultBrightness;
    float _contrast = DefaultContrast;
    float _motionBlur = DefaultMotionBlur;

    // Settings
    bool _renderSimulation = true;
};
