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
#include "RenderPipeline.h"
#include "RenderStep.h"
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

    setupRenderPipeline();

    _scrollbars = std::make_shared<_SimulationScrollbars>(true);

    resize(Viewport::get().getViewSize());
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
    _renderPipeline->resize(size);
    
    //if (_areTexturesInitialized) {
    //    glDeleteTextures(1, &_screenBackgroundTexture);
    //    _areTexturesInitialized = true;
    //}

    //// Init textures - use RGBA16F for proper floating point color handling
    //// Create screen background texture (dark blue)
    //glGenTextures(1, &_screenBackgroundTexture);
    //glBindTexture(GL_TEXTURE_2D, _screenBackgroundTexture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    //
    //// Fill with dark blue color (RGB: 0.0, 0.05, 0.15)
    //std::vector<float> darkBlueData(size.x * size.y * 4);
    //for (int i = 0; i < size.x * size.y; ++i) {
    //    darkBlueData[i * 4 + 0] = 0.0f;   // R
    //    darkBlueData[i * 4 + 1] = 0.0f;  // G
    //    darkBlueData[i * 4 + 2] = 0.15f;  // B
    //    darkBlueData[i * 4 + 3] = 1.0f;   // A
    //}
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, size.x, size.y, 0, GL_RGBA, GL_FLOAT, darkBlueData.data());

    Viewport::get().setViewSize(size);
}

void SimulationView::draw()
{
    if (_renderSimulation) {
        _renderPipeline->execute();

        //auto worldSize = _simulationFacade->getWorldSize();
        //auto worldRect = Viewport::get().getVisibleWorldRect();
        //auto viewSize = Viewport::get().getViewSize();
        //auto zoomFactor = Viewport::get().getZoomFactor();

        //// Extract object data from CUDA and transfer to OpenGL buffer
        //RenderBuffers renderBuffers{.vboForPoints = _objectBackgroundShader->getVbo()};
        //auto numRenderObjects = _simulationFacade->tryCopyBuffersFromCudaToOpenGL(renderBuffers);
        //if (numRenderObjects.has_value()) {
        //    _numObjects = *numRenderObjects;
        //}
        
        //GLint screenFbo;
        //glGetIntegerv(GL_FRAMEBUFFER_BINDING, &screenFbo);

        //if (zoomFactor > 2.0f) {

        //    //*******************************************************************
        //    //* 1. Step: Render objects for background to texture (objectTexture)
        //    //*******************************************************************
        //    glBindFramebuffer(GL_FRAMEBUFFER, _objectBackgroundFbo);
        //    glViewport(0, 0, viewSize.x, viewSize.y);

        //    // Clear with black background
        //    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        //    glClear(GL_COLOR_BUFFER_BIT);

        //    // Enable blending for anti-aliasing
        //    glEnable(GL_BLEND);
        //    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

        //    // Enable point sprites
        //    glEnable(GL_PROGRAM_POINT_SIZE);
        //    glEnable(GL_POINT_SPRITE);

        //    // Use background object shader
        //    _objectBackgroundShader->use();
        //    _objectBackgroundShader->setFloat("zoom", zoomFactor);
        //    _objectBackgroundShader->setFloat("radius", std::max(6.0f, zoomFactor));    // std::max to avoid moire patterns at low zoom factors
        //    _objectBackgroundShader->setVec2("worldSize", toFloat(worldSize.x), toFloat(worldSize.y));
        //    _objectBackgroundShader->setVec2("rectUpperLeft", worldRect.topLeft.x, worldRect.topLeft.y);
        //    _objectBackgroundShader->setVec2("viewportSize", toFloat(viewSize.x), toFloat(viewSize.y));

        //    // Draw points
        //    glBindVertexArray(_objectBackgroundShader->getVao());
        //    glDrawArrays(GL_POINTS, 0, toInt(_numObjects.vertices));

        //    //*******************************************************************
        //    //* 2. Step: Render objects for foreground to texture (objectTexture)
        //    //*******************************************************************
        //    glBindFramebuffer(GL_FRAMEBUFFER, _objectForegroundFbo);
        //    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        //    glClear(GL_COLOR_BUFFER_BIT);

        //    _objectForegroundShader->use();
        //    _objectForegroundShader->setFloat("zoom", zoomFactor);
        //    _objectForegroundShader->setFloat("radius", std::max(6.0f, zoomFactor * 0.5f)); // std::max to avoid moire patterns at low zoom factors
        //    _objectForegroundShader->setVec2("worldSize", toFloat(worldSize.x), toFloat(worldSize.y));
        //    _objectForegroundShader->setVec2("rectUpperLeft", worldRect.topLeft.x, worldRect.topLeft.y);
        //    _objectForegroundShader->setVec2("viewportSize", toFloat(viewSize.x), toFloat(viewSize.y));

        //    glBindVertexArray(_objectForegroundShader->getVao());
        //    glDrawArrays(GL_POINTS, 0, toInt(_numObjects.vertices));

        //    // Disable blending and point sprites
        //    glDisable(GL_PROGRAM_POINT_SIZE);
        //    glDisable(GL_BLEND);


        //    //*******************************************************
        //    //* 3. Step: Apply horizontal Gaussian blur to background
        //    //*******************************************************
        //    float blurRadius = zoomFactor / 4;

        //    _blurHorizontalShader->use();
        //    _blurHorizontalShader->setVec2("viewportSize", toFloat(viewSize.x), toFloat(viewSize.y));
        //    _blurHorizontalShader->setFloat("blurRadius", blurRadius);
        //    glBindVertexArray(_blurHorizontalShader->getVao());
        //    glActiveTexture(GL_TEXTURE0);
        //    glBindTexture(GL_TEXTURE_2D, _objectBackgroundTexture);
        //    glBindFramebuffer(GL_FRAMEBUFFER, _blurHorizontalFbo);

        //    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        //    //*****************************************************
        //    //* 4. Step: Apply vertical Gaussian blur to background
        //    //*****************************************************
        //    _blurVerticalShader->use();
        //    _blurVerticalShader->setVec2("viewportSize", toFloat(viewSize.x), toFloat(viewSize.y));
        //    _blurVerticalShader->setFloat("blurRadius", blurRadius);
        //    glBindVertexArray(_blurVerticalShader->getVao());
        //    glActiveTexture(GL_TEXTURE0);
        //    glBindTexture(GL_TEXTURE_2D, _blurHorizontalTexture);
        //    glBindFramebuffer(GL_FRAMEBUFFER, _blurVerticalFbo);

        //    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        //    //***********************************************
        //    //* 5. Step: Apply metaballs effect to background
        //    //***********************************************
        //    _metaballsShader->use();
        //    _metaballsShader->setVec2("viewportSize", toFloat(viewSize.x), toFloat(viewSize.y));
        //    glBindVertexArray(_metaballsShader->getVao());
        //    glActiveTexture(GL_TEXTURE0);
        //    glBindTexture(GL_TEXTURE_2D, _blurVerticalTexture);
        //    glBindFramebuffer(GL_FRAMEBUFFER, _metaballsFbo);
        //    //glBindFramebuffer(GL_FRAMEBUFFER, screenFbo);

        //    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        //    //*********************************************
        //    //* 6. Step: Apply Fresnel effect to background
        //    //*********************************************
        //    _fresnelShader->use();
        //    _fresnelShader->setVec2("viewportSize", toFloat(viewSize.x), toFloat(viewSize.y));
        //    _fresnelShader->setFloat("zoom", zoomFactor);
        //    glBindVertexArray(_fresnelShader->getVao());
        //    glActiveTexture(GL_TEXTURE0);
        //    glBindTexture(GL_TEXTURE_2D, _metaballsTexture);
        //    glBindFramebuffer(GL_FRAMEBUFFER, _fresnelFbo);

        //    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        //    //***********************************************************
        //    //* 7. Step: Apply subsurface scattering effect to background
        //    //***********************************************************
        //    _subsurfaceScatterShader->use();
        //    _subsurfaceScatterShader->setVec2("viewportSize", toFloat(viewSize.x), toFloat(viewSize.y));
        //    _subsurfaceScatterShader->setFloat("zoom", zoomFactor);
        //    glBindVertexArray(_subsurfaceScatterShader->getVao());
        //    glActiveTexture(GL_TEXTURE0);
        //    glBindTexture(GL_TEXTURE_2D, _fresnelTexture);
        //    glBindFramebuffer(GL_FRAMEBUFFER, _subsurfaceScatterFbo);

        //    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        //    //********************************************
        //    //* 8. Step: Merge layers and render to screen
        //    //********************************************
        //    _mergeShader->use();
        //    glBindVertexArray(_mergeShader->getVao());
        //    glActiveTexture(GL_TEXTURE0);
        //    glBindTexture(GL_TEXTURE_2D, _subsurfaceScatterTexture);
        //    glActiveTexture(GL_TEXTURE1);
        //    glBindTexture(GL_TEXTURE_2D, _objectForegroundTexture);
        //    glActiveTexture(GL_TEXTURE2);
        //    glBindTexture(GL_TEXTURE_2D, _screenBackgroundTexture);
        //    glBindFramebuffer(GL_FRAMEBUFFER, screenFbo);

        //    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        //} else {
        //    //*******************************************************************
        //    //* 1. Step: Render objects for background to texture (objectTexture)
        //    //*******************************************************************
        //    glBindFramebuffer(GL_FRAMEBUFFER, _objectBackgroundFbo);
        //    glViewport(0, 0, viewSize.x, viewSize.y);

        //    // Clear with black background
        //    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        //    glClear(GL_COLOR_BUFFER_BIT);

        //    // Enable blending for anti-aliasing
        //    glEnable(GL_BLEND);
        //    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

        //    // Enable point sprites
        //    glEnable(GL_PROGRAM_POINT_SIZE);
        //    glEnable(GL_POINT_SPRITE);

        //    // Use background object shader
        //    _objectBackgroundShader->use();
        //    _objectBackgroundShader->setFloat("zoom", zoomFactor);
        //    _objectBackgroundShader->setFloat("radius", std::max(4.5f, zoomFactor));  // std::max to avoid moire patterns at low zoom factors
        //    _objectBackgroundShader->setVec2("worldSize", toFloat(worldSize.x), toFloat(worldSize.y));
        //    _objectBackgroundShader->setVec2("rectUpperLeft", worldRect.topLeft.x, worldRect.topLeft.y);
        //    _objectBackgroundShader->setVec2("viewportSize", toFloat(viewSize.x), toFloat(viewSize.y));

        //    // Draw points
        //    glBindVertexArray(_objectBackgroundShader->getVao());
        //    glDrawArrays(GL_POINTS, 0, toInt(_numObjects.vertices));

        //    // Disable blending and point sprites
        //    glDisable(GL_PROGRAM_POINT_SIZE);
        //    glDisable(GL_BLEND);

        //    //*******************************************************
        //    //* 2. Step: Apply horizontal Gaussian blur to background
        //    //*******************************************************
        //    float blurRadius = 1.0f;
        //    //zoomFactor / 4;

        //    _blurHorizontalShader->use();
        //    _blurHorizontalShader->setVec2("viewportSize", toFloat(viewSize.x), toFloat(viewSize.y));
        //    _blurHorizontalShader->setFloat("blurRadius", blurRadius);
        //    glBindVertexArray(_blurHorizontalShader->getVao());
        //    glActiveTexture(GL_TEXTURE0);
        //    glBindTexture(GL_TEXTURE_2D, _objectBackgroundTexture);
        //    glBindFramebuffer(GL_FRAMEBUFFER, _blurHorizontalFbo);

        //    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        //    //*****************************************************
        //    //* 3. Step: Apply vertical Gaussian blur to background
        //    //*****************************************************
        //    _blurVerticalShader->use();
        //    _blurVerticalShader->setVec2("viewportSize", toFloat(viewSize.x), toFloat(viewSize.y));
        //    _blurVerticalShader->setFloat("blurRadius", blurRadius);
        //    glBindVertexArray(_blurVerticalShader->getVao());
        //    glActiveTexture(GL_TEXTURE0);
        //    glBindTexture(GL_TEXTURE_2D, _blurHorizontalTexture);
        //    glBindFramebuffer(GL_FRAMEBUFFER, _blurVerticalFbo);

        //    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        //    //********************************************
        //    //* 4. Step: Merge layers and render to screen
        //    //********************************************
        //    _mergeShader->use();
        //    glBindVertexArray(_mergeShader->getVao());
        //    glActiveTexture(GL_TEXTURE0);
        //    glBindTexture(GL_TEXTURE_2D, _blurVerticalTexture);
        //    glActiveTexture(GL_TEXTURE1);
        //    glBindTexture(GL_TEXTURE_2D, _blurVerticalTexture);
        //    glActiveTexture(GL_TEXTURE2);
        //    glBindTexture(GL_TEXTURE_2D, _screenBackgroundTexture);
        //    glBindFramebuffer(GL_FRAMEBUFFER, screenFbo);

        //    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        //}

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

void SimulationView::setupRenderPipeline()
{
    _renderPipeline = std::make_shared<_RenderPipeline>(_simulationFacade);

    auto renderStep1 = _PointRenderStep::create(Const::ObjectBackgroundVertexShader, Const::ObjectBackgroundFragmentShader);
    _renderPipeline->addStep(renderStep1);

    auto renderStepLine = _LineRenderStep::create(Const::LineVertexShader, Const::LineFragmentShader, renderStep1);
    _renderPipeline->addStep(renderStepLine);

    auto renderStep2 = _PostProcessingRenderStep::create(Const::BlurHorizontalVertexShader, Const::BlurHorizontalFragmentShader, std::vector<RenderStep>{renderStep1});
    _renderPipeline->addStep(renderStep2);

    auto renderStep3 = _PostProcessingRenderStep::create(Const::BlurVerticalVertexShader, Const::BlurVerticalFragmentShader, std::vector<RenderStep>{renderStep2});
    _renderPipeline->addStep(renderStep3);

    auto renderStep4 = _PostProcessingRenderStep::create(Const::MetaballsVertexShader, Const::MetaballsFragmentShader, std::vector<RenderStep>{renderStep3});
    _renderPipeline->addStep(renderStep4);

    auto renderStep5 = _PostProcessingRenderStep::create(Const::FresnelVertexShader, Const::FresnelFragmentShader, std::vector<RenderStep>{renderStep4});
    _renderPipeline->addStep(renderStep5);

    auto renderStep6 = _PostProcessingRenderStep::create(Const::SubsurfaceScatterVertexShader, Const::SubsurfaceScatterFragmentShader, std::vector<RenderStep>{renderStep5});
    _renderPipeline->addStep(renderStep6);

    auto renderStep7 = _PointRenderStep::createWithSharedVbo(Const::ObjectForegroundVertexShader, Const::ObjectForegroundFragmentShader, renderStep1);
    _renderPipeline->addStep(renderStep7);

    auto renderStep8 = _PostProcessingRenderStep::create(Const::MergeVertexShader, Const::MergeFragmentShader, std::vector<RenderStep>{renderStep6, renderStep7});
    _renderPipeline->addStep(renderStep8);
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

