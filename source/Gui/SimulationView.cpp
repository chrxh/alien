#include "SimulationView.h"

#include <algorithm>
#include <glad/glad.h>
#include <imgui.h>

#include "Base/GlobalSettings.h"
#include "Base/Resources.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/SpaceCalculator.h"
#include "EngineInterface/RenderData.h"

#include "AlienGui.h"
#include "SimulationScrollbars.h"
#include "Shader.h"
#include "Viewport.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr ZoomFactorForOverlay = 12.0f;
}

void SimulationView::setup(SimulationFacade const& simulationFacade)
{
    _simulationFacade = simulationFacade;

    _cellDetailOverlayActive = GlobalSettings::get().getValue("settings.simulation view.overlay", _cellDetailOverlayActive);
    _brightness = GlobalSettings::get().getValue("windows.simulation view.brightness", _brightness);
    _contrast = GlobalSettings::get().getValue("windows.simulation view.contrast", _contrast);
    _motionBlur = GlobalSettings::get().getValue("windows.simulation view.motion blur factor", _motionBlur);

    setupBackgroundObjectShader();
    setupForegroundObjectShader();
    setupBlurHorizontalShader();
    setupBlurVerticalShader();
    setupMetaballsShader();
    setupSubsurfaceScatterShader();
    setupFresnelShader();
    setupMergeShader();

    _scrollbars = std::make_shared<_SimulationScrollbars>(true);

    resize(Viewport::get().getViewSize());

    // Setup shaders
    _blurHorizontalShader->use();
    _blurHorizontalShader->setInt("inputTexture", 0);
    
    _blurVerticalShader->use();
    _blurVerticalShader->setInt("inputTexture", 0);
    
    _metaballsShader->use();
    _metaballsShader->setInt("inputTexture", 0);
    
    _subsurfaceScatterShader->use();
    _subsurfaceScatterShader->setInt("inputTexture", 0);
    
    _fresnelShader->use();
    _fresnelShader->setInt("inputTexture", 0);
    
    _mergeShader->use();
    _mergeShader->setInt("backgroundObjectTexture", 0);
    _mergeShader->setInt("foregroundObjectTexture", 1);
}

void SimulationView::shutdown()
{
    GlobalSettings::get().setValue("settings.simulation view.overlay", _cellDetailOverlayActive);
    GlobalSettings::get().setValue("windows.simulation view.brightness", _brightness);
    GlobalSettings::get().setValue("windows.simulation view.contrast", _contrast);
    GlobalSettings::get().setValue("windows.simulation view.motion blur factor", _motionBlur);
}

void SimulationView::resize(IntVector2D const& size)
{
    if (_areTexturesInitialized) {
        glDeleteFramebuffers(1, &_backgroundObjectFbo);
        glDeleteFramebuffers(1, &_foregroundObjectFbo);
        glDeleteFramebuffers(1, &_blurHorizontalFbo);
        glDeleteFramebuffers(1, &_blurVerticalFbo);
        glDeleteFramebuffers(1, &_metaballsFbo);
        glDeleteFramebuffers(1, &_subsurfaceScatterFbo);
        glDeleteFramebuffers(1, &_fresnelFbo);
        glDeleteTextures(1, &_backgroundObjectTexture);
        glDeleteTextures(1, &_foregroundObjectTexture);
        glDeleteTextures(1, &_blurHorizontalTexture);
        glDeleteTextures(1, &_blurVerticalTexture);
        glDeleteTextures(1, &_metaballsTexture);
        glDeleteTextures(1, &_subsurfaceScatterTexture);
        glDeleteTextures(1, &_fresnelTexture);
        _areTexturesInitialized = true;
    }

    // Init textures - use RGBA16F for proper floating point color handling
    glGenTextures(1, &_backgroundObjectTexture);
    glBindTexture(GL_TEXTURE_2D, _backgroundObjectTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, size.x, size.y, 0, GL_RGBA, GL_FLOAT, NULL);

    glGenTextures(1, &_foregroundObjectTexture);
    glBindTexture(GL_TEXTURE_2D, _foregroundObjectTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, size.x, size.y, 0, GL_RGBA, GL_FLOAT, NULL);

    glGenTextures(1, &_blurHorizontalTexture);
    glBindTexture(GL_TEXTURE_2D, _blurHorizontalTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, size.x, size.y, 0, GL_RGBA, GL_FLOAT, NULL);

    glGenTextures(1, &_blurVerticalTexture);
    glBindTexture(GL_TEXTURE_2D, _blurVerticalTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, size.x, size.y, 0, GL_RGBA, GL_FLOAT, NULL);

    glGenTextures(1, &_metaballsTexture);
    glBindTexture(GL_TEXTURE_2D, _metaballsTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, size.x, size.y, 0, GL_RGBA, GL_FLOAT, NULL);

    glGenTextures(1, &_subsurfaceScatterTexture);
    glBindTexture(GL_TEXTURE_2D, _subsurfaceScatterTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, size.x, size.y, 0, GL_RGBA, GL_FLOAT, NULL);

    glGenTextures(1, &_fresnelTexture);
    glBindTexture(GL_TEXTURE_2D, _fresnelTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, size.x, size.y, 0, GL_RGBA, GL_FLOAT, NULL);

    // Init framebuffers
    glGenFramebuffers(1, &_backgroundObjectFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _backgroundObjectFbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _backgroundObjectTexture, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glGenFramebuffers(1, &_foregroundObjectFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _foregroundObjectFbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _foregroundObjectTexture, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glGenFramebuffers(1, &_blurHorizontalFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _blurHorizontalFbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _blurHorizontalTexture, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glGenFramebuffers(1, &_blurVerticalFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _blurVerticalFbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _blurVerticalTexture, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glGenFramebuffers(1, &_metaballsFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _metaballsFbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _metaballsTexture, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glGenFramebuffers(1, &_subsurfaceScatterFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _subsurfaceScatterFbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _subsurfaceScatterTexture, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glGenFramebuffers(1, &_fresnelFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _fresnelFbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _fresnelTexture, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    Viewport::get().setViewSize(size);
}

void SimulationView::draw()
{
    if (_renderSimulation) {
        auto worldSize = _simulationFacade->getWorldSize();
        auto worldRect = Viewport::get().getVisibleWorldRect();
        auto viewSize = Viewport::get().getViewSize();
        auto zoomFactor = Viewport::get().getZoomFactor();

        // Extract object data from CUDA and transfer to OpenGL buffer
        RenderBuffers renderBuffers{.vboForPoints = _backgroundObjectShader->getVbo()};
        auto numRenderObjects = _simulationFacade->tryCopyBuffersFromCudaToOpenGL(renderBuffers);
        if (numRenderObjects.has_value()) {
            _numObjects = *numRenderObjects;
        }
        
        GLint screenFbo;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &screenFbo);

        if (zoomFactor > 2.0f) {

            //*******************************************************************
            //* 1. Step: Render objects for background to texture (objectTexture)
            //*******************************************************************
            glBindFramebuffer(GL_FRAMEBUFFER, _backgroundObjectFbo);
            //glBindFramebuffer(GL_FRAMEBUFFER, screenFbo);
            glViewport(0, 0, viewSize.x, viewSize.y);

            // Clear with black background
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            // Enable blending for anti-aliasing
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE);

            // Enable point sprites
            glEnable(GL_PROGRAM_POINT_SIZE);
            glEnable(GL_POINT_SPRITE);

            // Use background object shader
            _backgroundObjectShader->use();
            _backgroundObjectShader->setFloat("zoom", zoomFactor);
            _backgroundObjectShader->setFloat("radius", std::max(6.0f, zoomFactor));    // std::max to avoid moiré patterns at low zoom factors
            _backgroundObjectShader->setVec2("worldSize", toFloat(worldSize.x), toFloat(worldSize.y));
            _backgroundObjectShader->setVec2("rectUpperLeft", worldRect.topLeft.x, worldRect.topLeft.y);
            _backgroundObjectShader->setVec2("viewportSize", toFloat(viewSize.x), toFloat(viewSize.y));

            // Draw points
            glBindVertexArray(_backgroundObjectShader->getVao());
            glDrawArrays(GL_POINTS, 0, toInt(_numObjects.vertices));

            //*******************************************************************
            //* 2. Step: Render objects for foreground to texture (objectTexture)
            //*******************************************************************
            glBindFramebuffer(GL_FRAMEBUFFER, _foregroundObjectFbo);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            _foregroundObjectShader->use();
            _foregroundObjectShader->setFloat("zoom", zoomFactor);
            _foregroundObjectShader->setFloat("radius", std::max(6.0f, zoomFactor * 0.5f)); // std::max to avoid moiré patterns at low zoom factors
            _foregroundObjectShader->setVec2("worldSize", toFloat(worldSize.x), toFloat(worldSize.y));
            _foregroundObjectShader->setVec2("rectUpperLeft", worldRect.topLeft.x, worldRect.topLeft.y);
            _foregroundObjectShader->setVec2("viewportSize", toFloat(viewSize.x), toFloat(viewSize.y));

            glBindVertexArray(_foregroundObjectShader->getVao());
            glDrawArrays(GL_POINTS, 0, toInt(_numObjects.vertices));

            // Disable blending and point sprites
            glDisable(GL_PROGRAM_POINT_SIZE);
            glDisable(GL_BLEND);


            //*******************************************************
            //* 3. Step: Apply horizontal Gaussian blur to background
            //*******************************************************
            float blurRadius = zoomFactor / 4;

            // Apply horizontal Gaussian blur preprocessing
            _blurHorizontalShader->use();
            _blurHorizontalShader->setVec2("viewportSize", toFloat(viewSize.x), toFloat(viewSize.y));
            _blurHorizontalShader->setFloat("blurRadius", blurRadius);
            glBindVertexArray(_blurHorizontalShader->getVao());
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, _backgroundObjectTexture);
            glBindFramebuffer(GL_FRAMEBUFFER, _blurHorizontalFbo);

            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

            //*****************************************************
            //* 4. Step: Apply vertical Gaussian blur to background
            //*****************************************************
            _blurVerticalShader->use();
            _blurVerticalShader->setVec2("viewportSize", toFloat(viewSize.x), toFloat(viewSize.y));
            _blurVerticalShader->setFloat("blurRadius", blurRadius);
            glBindVertexArray(_blurVerticalShader->getVao());
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, _blurHorizontalTexture);
            glBindFramebuffer(GL_FRAMEBUFFER, _blurVerticalFbo);

            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

            //***********************************************
            //* 5. Step: Apply metaballs effect to background
            //***********************************************
            _metaballsShader->use();
            _metaballsShader->setVec2("viewportSize", toFloat(viewSize.x), toFloat(viewSize.y));
            glBindVertexArray(_metaballsShader->getVao());
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, _blurVerticalTexture);
            glBindFramebuffer(GL_FRAMEBUFFER, _metaballsFbo);
            //glBindFramebuffer(GL_FRAMEBUFFER, screenFbo);

            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

            //*********************************************
            //* 6. Step: Apply Fresnel effect to background
            //*********************************************
            _fresnelShader->use();
            _fresnelShader->setVec2("viewportSize", toFloat(viewSize.x), toFloat(viewSize.y));
            _fresnelShader->setFloat("zoom", zoomFactor);
            glBindVertexArray(_fresnelShader->getVao());
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, _metaballsTexture);
            glBindFramebuffer(GL_FRAMEBUFFER, _fresnelFbo);

            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
            //***********************************************************
            //* 7. Step: Apply subsurface scattering effect to background
            //***********************************************************
            _subsurfaceScatterShader->use();
            _subsurfaceScatterShader->setVec2("viewportSize", toFloat(viewSize.x), toFloat(viewSize.y));
            _subsurfaceScatterShader->setFloat("zoom", zoomFactor);
            glBindVertexArray(_subsurfaceScatterShader->getVao());
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, _fresnelTexture);
            glBindFramebuffer(GL_FRAMEBUFFER, _subsurfaceScatterFbo);

            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

            //****************************************************************
            //* 7. Step: Merge background with foreground and render to screen
            //****************************************************************
            _mergeShader->use();
            glBindVertexArray(_mergeShader->getVao());
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, _subsurfaceScatterTexture);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, _foregroundObjectTexture);
            glBindFramebuffer(GL_FRAMEBUFFER, screenFbo);

            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        } else {
            //*******************************************************************
            //* 1. Step: Render objects for background to texture (objectTexture)
            //*******************************************************************
            glBindFramebuffer(GL_FRAMEBUFFER, _backgroundObjectFbo);
            glViewport(0, 0, viewSize.x, viewSize.y);

            // Clear with black background
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            // Enable blending for anti-aliasing
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE);

            // Enable point sprites
            glEnable(GL_PROGRAM_POINT_SIZE);
            glEnable(GL_POINT_SPRITE);

            // Use background object shader
            _backgroundObjectShader->use();
            _backgroundObjectShader->setFloat("zoom", zoomFactor);
            _backgroundObjectShader->setFloat("radius", std::max(4.5f, zoomFactor));  // std::max to avoid moiré patterns at low zoom factors
            _backgroundObjectShader->setVec2("worldSize", toFloat(worldSize.x), toFloat(worldSize.y));
            _backgroundObjectShader->setVec2("rectUpperLeft", worldRect.topLeft.x, worldRect.topLeft.y);
            _backgroundObjectShader->setVec2("viewportSize", toFloat(viewSize.x), toFloat(viewSize.y));

            // Draw points
            glBindVertexArray(_backgroundObjectShader->getVao());
            glDrawArrays(GL_POINTS, 0, toInt(_numObjects.vertices));

            // Disable blending and point sprites
            glDisable(GL_PROGRAM_POINT_SIZE);
            glDisable(GL_BLEND);

            //*******************************************************
            //* 2. Step: Apply horizontal Gaussian blur to background
            //*******************************************************
            float blurRadius = 1.0f;
            //zoomFactor / 4;

            // Apply horizontal Gaussian blur preprocessing
            _blurHorizontalShader->use();
            _blurHorizontalShader->setVec2("viewportSize", toFloat(viewSize.x), toFloat(viewSize.y));
            _blurHorizontalShader->setFloat("blurRadius", blurRadius);
            glBindVertexArray(_blurHorizontalShader->getVao());
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, _backgroundObjectTexture);
            glBindFramebuffer(GL_FRAMEBUFFER, _blurHorizontalFbo);

            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

            //*****************************************************
            //* 3. Step: Apply vertical Gaussian blur to background
            //*****************************************************
            _blurVerticalShader->use();
            _blurVerticalShader->setVec2("viewportSize", toFloat(viewSize.x), toFloat(viewSize.y));
            _blurVerticalShader->setFloat("blurRadius", blurRadius);
            glBindVertexArray(_blurVerticalShader->getVao());
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, _blurHorizontalTexture);
            glBindFramebuffer(GL_FRAMEBUFFER, screenFbo);

            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        }

        if (_simulationFacade->getSimulationParameters().markReferenceDomain.value) {
            markReferenceDomain();
        }

    } else {
        glClearColor(0, 0, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        auto textWidth = scale(300.0f);
        auto textHeight = scale(80.0f);
        ImDrawList* drawList = ImGui::GetBackgroundDrawList();
        auto& styleRep = StyleRepository::get();
        auto right = ImGui::GetMainViewport()->Pos.x + ImGui::GetMainViewport()->Size.x;
        auto bottom = ImGui::GetMainViewport()->Pos.y + ImGui::GetMainViewport()->Size.y;
        auto maxLength = std::max(right, bottom);

        AlienGui::RotateStart(drawList);
        auto font = styleRep.getReefLargeFont();
        auto text = "Rendering disabled";
        ImVec4 clipRect(-100000.0f, -100000.0f, 100000.0f, 100000.0f);
        for (int i = 0; toFloat(i) * textWidth < maxLength * 2; ++i) {
            for (int j = 0; toFloat(j) * textHeight < maxLength * 2; ++j) {
                font->RenderText(
                    drawList,
                    scale(34.0f),
                    {toFloat(i) * textWidth - maxLength / 2, toFloat(j) * textHeight - maxLength / 2},
                    Const::RenderingDisabledTextColor,
                    clipRect,
                    text,
                    text + strlen(text),
                    0.0f,
                    false);
            }
        }
        AlienGui::RotateEnd(45.0f, drawList);
    }
}

void SimulationView::processSimulationScrollbars()
{
    if (_renderSimulation) {
        ImGuiViewport* viewport = ImGui::GetMainViewport();
        auto mainMenubarHeight = scale(22);

        auto worldCenter = Viewport::get().getCenterInWorldPos();
        auto worldRect = RealRect{{0,0}, toRealVector2D(_simulationFacade->getWorldSize())};
        auto visibleWorldRect = Viewport::get().getVisibleWorldRect();
        auto viewRect =
            RealRect{{viewport->Pos.x, viewport->Pos.y + mainMenubarHeight}, {viewport->Pos.x + viewport->Size.x, viewport->Pos.y + viewport->Size.y}};
        _scrollbars->process(worldCenter, worldRect, visibleWorldRect, viewRect);
        Viewport::get().setCenterInWorldPos({worldCenter.x, worldCenter.y});
    }
}

bool SimulationView::isScrollbarDragging() const
{
    return _scrollbars->isHoveredOrDragged();
}

bool SimulationView::isRenderSimulation() const
{
    return _renderSimulation;
}

void SimulationView::setRenderSimulation(bool value)
{
    _renderSimulation = value;
}

bool SimulationView::isOverlayActive() const
{
    return _cellDetailOverlayActive;
}

void SimulationView::setOverlayActive(bool active)
{
    _cellDetailOverlayActive = active;
}

float SimulationView::getBrightness() const
{
    return _brightness;
}

void SimulationView::setBrightness(float value)
{
    _brightness = value;
    // Metaballs shader doesn't use brightness parameter
}

float SimulationView::getContrast() const
{
    return _contrast;
}

void SimulationView::setContrast(float value)
{
    _contrast = value;
    // Metaballs shader doesn't use contrast parameter
}

float SimulationView::getMotionBlur() const
{
    return _motionBlur;
}

void SimulationView::setMotionBlur(float value)
{
    _motionBlur = value;
    // Metaballs shader doesn't use motion blur parameter
}

void SimulationView::updateMotionBlur()
{
    // Metaballs shader doesn't use motion blur parameter
}

void SimulationView::setupBackgroundObjectShader()
{
    _backgroundObjectShader = std::make_shared<_Shader>(Const::ObjectVertexShader, Const::BackgroundObjectFragmentShader);

    auto vao = _backgroundObjectShader->getVao();
    auto vbo = _backgroundObjectShader->getVbo();

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 1000000 * sizeof(VertexData), nullptr, GL_DYNAMIC_DRAW);

    // Setup vertex attributes for RenderingObjectData
    // Position (2 floats)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)0);
    glEnableVertexAttribArray(0);

    // Color (3 floats)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
}

void SimulationView::setupForegroundObjectShader()
{
    _foregroundObjectShader = std::make_shared<_Shader>(Const::ObjectVertexShader, Const::ForegroundObjectFragmentShader);

    auto vao = _foregroundObjectShader->getVao();
    auto vbo = _foregroundObjectShader->getVbo();

    glBindVertexArray(vao);
    // Use the same VBO as background shader (data is shared)
    glBindBuffer(GL_ARRAY_BUFFER, _backgroundObjectShader->getVbo());

    // Setup vertex attributes for RenderingObjectData
    // Position (2 floats)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)0);
    glEnableVertexAttribArray(0);

    // Color (3 floats)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
}

void SimulationView::setupBlurHorizontalShader()
{
    _blurHorizontalShader = std::make_shared<_Shader>(Const::BlurHorizontalVertexShader, Const::BlurHorizontalFragmentShader);

    // Setup full-screen quad
    float vertices[] = {
        1.0f,  1.0f,  0.0f, 1.0f, 1.0f,  // top right
        1.0f,  -1.0f, 0.0f, 1.0f, 0.0f,  // bottom right
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,  // bottom left
        -1.0f, 1.0f,  0.0f, 0.0f, 1.0f   // top left
    };
    unsigned int indices[] = {
        0, 1, 3,  // first triangle
        1, 2, 3   // second triangle
    };

    auto vao = _blurHorizontalShader->getVao();
    auto vbo = _blurHorizontalShader->getVbo();
    auto ebo = _blurHorizontalShader->getEbo();
    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Texture coordinate attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
}

void SimulationView::setupBlurVerticalShader()
{
    _blurVerticalShader = std::make_shared<_Shader>(Const::BlurVerticalVertexShader, Const::BlurVerticalFragmentShader);

    // Setup full-screen quad
    float vertices[] = {
        1.0f,  1.0f,  0.0f, 1.0f, 1.0f,  // top right
        1.0f,  -1.0f, 0.0f, 1.0f, 0.0f,  // bottom right
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,  // bottom left
        -1.0f, 1.0f,  0.0f, 0.0f, 1.0f   // top left
    };
    unsigned int indices[] = {
        0, 1, 3,  // first triangle
        1, 2, 3   // second triangle
    };

    auto vao = _blurVerticalShader->getVao();
    auto vbo = _blurVerticalShader->getVbo();
    auto ebo = _blurVerticalShader->getEbo();
    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Texture coordinate attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
}

void SimulationView::setupMetaballsShader()
{
    _metaballsShader = std::make_shared<_Shader>(Const::MetaballsVertexShader, Const::MetaballsFragmentShader);

    // Setup full-screen quad
    float vertices[] = {
        1.0f,  1.0f,  0.0f, 1.0f, 1.0f,  // top right
        1.0f,  -1.0f, 0.0f, 1.0f, 0.0f,  // bottom right
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,  // bottom left
        -1.0f, 1.0f,  0.0f, 0.0f, 1.0f   // top left
    };
    unsigned int indices[] = {
        0, 1, 3,  // first triangle
        1, 2, 3   // second triangle
    };

    auto vao = _metaballsShader->getVao();
    auto vbo = _metaballsShader->getVbo();
    auto ebo = _metaballsShader->getEbo();
    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Texture coordinate attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
}

void SimulationView::setupSubsurfaceScatterShader()
{
    _subsurfaceScatterShader = std::make_shared<_Shader>(Const::SubsurfaceScatterVertexShader, Const::SubsurfaceScatterFragmentShader);

    // Setup full-screen quad
    float vertices[] = {
        1.0f,  1.0f,  0.0f, 1.0f, 1.0f,  // top right
        1.0f,  -1.0f, 0.0f, 1.0f, 0.0f,  // bottom right
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,  // bottom left
        -1.0f, 1.0f,  0.0f, 0.0f, 1.0f   // top left
    };
    unsigned int indices[] = {
        0, 1, 3,  // first triangle
        1, 2, 3   // second triangle
    };

    auto vao = _subsurfaceScatterShader->getVao();
    auto vbo = _subsurfaceScatterShader->getVbo();
    auto ebo = _subsurfaceScatterShader->getEbo();
    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Texture coordinate attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
}

void SimulationView::setupFresnelShader()
{
    _fresnelShader = std::make_shared<_Shader>(Const::FresnelVertexShader, Const::FresnelFragmentShader);

    // Setup full-screen quad
    float vertices[] = {
        1.0f,  1.0f,  0.0f, 1.0f, 1.0f,  // top right
        1.0f,  -1.0f, 0.0f, 1.0f, 0.0f,  // bottom right
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,  // bottom left
        -1.0f, 1.0f,  0.0f, 0.0f, 1.0f   // top left
    };
    unsigned int indices[] = {
        0, 1, 3,  // first triangle
        1, 2, 3   // second triangle
    };

    auto vao = _fresnelShader->getVao();
    auto vbo = _fresnelShader->getVbo();
    auto ebo = _fresnelShader->getEbo();
    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Texture coordinate attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
}

void SimulationView::setupMergeShader()
{
    _mergeShader = std::make_shared<_Shader>(Const::MergeVertexShader, Const::MergeFragmentShader);

    // Setup full-screen quad
    float vertices[] = {
        1.0f,  1.0f,  0.0f, 1.0f, 1.0f,  // top right
        1.0f,  -1.0f, 0.0f, 1.0f, 0.0f,  // bottom right
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,  // bottom left
        -1.0f, 1.0f,  0.0f, 0.0f, 1.0f   // top left
    };
    unsigned int indices[] = {
        0, 1, 3,  // first triangle
        1, 2, 3   // second triangle
    };

    auto vao = _mergeShader->getVao();
    auto vbo = _mergeShader->getVbo();
    auto ebo = _mergeShader->getEbo();
    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Texture coordinate attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
}

void SimulationView::markReferenceDomain()
{
    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    auto p1 = Viewport::get().mapWorldToViewPosition({0, 0}, false);
    auto worldSize = _simulationFacade->getWorldSize();
    auto p2 = Viewport::get().mapWorldToViewPosition(toRealVector2D(worldSize), false);
    auto color = ImColor::HSV(0.66f, 1.0f, 1.0f, 0.8f);
    drawList->AddLine({p1.x, p1.y}, {p2.x, p1.y}, color);
    drawList->AddLine({p2.x, p1.y}, {p2.x, p2.y}, color);
    drawList->AddLine({p2.x, p2.y}, {p1.x, p2.y}, color);
    drawList->AddLine({p1.x, p2.y}, {p1.x, p1.y}, color);
}

