#include "SimulationView.h"

#include <algorithm>
#include <glad/glad.h>
#include <imgui.h>

#include "Base/GlobalSettings.h"
#include "Base/Resources.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/SpaceCalculator.h"

#include "AlienImGui.h"
#include "Shader.h"
#include "SimulationScrollbar.h"
#include "Viewport.h"
#include "SimulationInteractionController.h"
#include "StyleRepository.h"
#include "CellTypeStrings.h"

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

    _shader = std::make_shared<_Shader>(Const::SimulationVertexShader, Const::SimulationFragmentShader);

    _scrollbarX = std::make_shared<_SimulationScrollbar>(
        "SimScrollbarX", _SimulationScrollbar ::Orientation::Horizontal, _simulationFacade);
    _scrollbarY = std::make_shared<_SimulationScrollbar>(
        "SimScrollbarY", _SimulationScrollbar::Orientation::Vertical, _simulationFacade);

    float vertices[] = {
        // positions        // texture coordinates
        1.0f,  1.0f,  0.0f, 1.0f, 1.0f,  // top right
        1.0f,  -1.0f, 0.0f, 1.0f, 0.0f,  // bottom right
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,  // bottom left
        -1.0f, 1.0f,  0.0f, 0.0f, 1.0f   // top left
    };
    unsigned int indices[] = {
        0,
        1,
        3,  // first triangle
        1,
        2,
        3  // second triangle
    };
    glGenVertexArrays(1, &_vao);
    glGenBuffers(1, &_vbo);
    glGenBuffers(1, &_ebo);

    glBindVertexArray(_vao);

    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
  
    // texture coordinate attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    resize(Viewport::get().getViewSize());

    _shader->use();
    _shader->setInt("texture1", 0);
    _shader->setInt("texture2", 1);
    _shader->setInt("texture3", 2);
    _shader->setBool("glowEffect", true);
    _shader->setBool("motionEffect", true);
    updateMotionBlur();
    setBrightness(1.0f);
    setContrast(1.0f);
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
        glDeleteFramebuffers(1, &_fbo1);
        glDeleteFramebuffers(1, &_fbo2);
        glDeleteTextures(1, &_textureSimulationId);
        glDeleteTextures(1, &_textureFramebufferId1);
        glDeleteTextures(1, &_textureFramebufferId2);
        _areTexturesInitialized = true;
    }
    glGenTextures(1, &_textureSimulationId);
    glBindTexture(GL_TEXTURE_2D, _textureSimulationId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16, size.x, size.y, 0, GL_RGB, GL_UNSIGNED_SHORT, NULL);
    _simulationFacade->setImageResource(reinterpret_cast<void*>(uintptr_t(_textureSimulationId)));

    glGenTextures(1, &_textureFramebufferId1);
    glBindTexture(GL_TEXTURE_2D, _textureFramebufferId1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16, size.x, size.y, 0, GL_RGB, GL_UNSIGNED_SHORT, NULL);

    glGenTextures(1, &_textureFramebufferId2);
    glBindTexture(GL_TEXTURE_2D, _textureFramebufferId2);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16, size.x, size.y, 0, GL_RGB, GL_UNSIGNED_SHORT, NULL);

    glGenFramebuffers(1, &_fbo1);
    glBindFramebuffer(GL_FRAMEBUFFER, _fbo1);  
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _textureFramebufferId1, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);  

    glGenFramebuffers(1, &_fbo2);
    glBindFramebuffer(GL_FRAMEBUFFER, _fbo2);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _textureFramebufferId2, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);  

    Viewport::get().setViewSize(size);
}

void SimulationView::draw()
{
    if (_renderSimulation) {
        updateImageFromSimulation();

        _shader->use();

        GLint currentFbo;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &currentFbo);

        glBindFramebuffer(GL_FRAMEBUFFER, _fbo1);
        _shader->setInt("phase", 10);
        glBindVertexArray(_vao);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, _textureSimulationId);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glBindFramebuffer(GL_FRAMEBUFFER, _fbo2);
        _shader->setInt("phase", 11);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, _textureSimulationId);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, _textureFramebufferId1);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, _textureFramebufferId2);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glBindFramebuffer(GL_FRAMEBUFFER, currentFbo);
        _shader->setInt("phase", 12);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, _textureFramebufferId2);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        if (_simulationFacade->getSimulationParameters().markReferenceDomain) {
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

        AlienImGui::RotateStart(drawList);
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
        AlienImGui::RotateEnd(45.0f, drawList);
    }
}

void SimulationView::processSimulationScrollbars()
{
    if (_renderSimulation) {
        ImGuiViewport* viewport = ImGui::GetMainViewport();
        auto mainMenubarHeight = scale(22);
        auto scrollbarThickness = 17;  //fixed
        _scrollbarX->process({{viewport->Pos.x, viewport->Size.y - scrollbarThickness}, {viewport->Size.x - 1 - scrollbarThickness, 1}});
        _scrollbarY->process({{viewport->Size.x - scrollbarThickness, viewport->Pos.y + mainMenubarHeight}, {1, viewport->Size.y - 1 - scrollbarThickness}});
    }
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
    _shader->setFloat("brightness", value);
}

float SimulationView::getContrast() const
{
    return _contrast;
}

void SimulationView::setContrast(float value)
{
    _contrast = value;
    _shader->setFloat("contrast", value);
}

float SimulationView::getMotionBlur() const
{
    return _motionBlur;
}

void SimulationView::setMotionBlur(float value)
{
    _motionBlur = value;
    updateMotionBlur();
}

void SimulationView::updateImageFromSimulation()
{
    auto worldRect = Viewport::get().getVisibleWorldRect();
    auto viewSize = Viewport::get().getViewSize();
    auto zoomFactor = Viewport::get().getZoomFactor();

    if (zoomFactor >= ZoomFactorForOverlay) {
        auto overlay = _simulationFacade->tryDrawVectorGraphicsAndReturnOverlay(
            worldRect.topLeft, worldRect.bottomRight, {viewSize.x, viewSize.y}, zoomFactor);
        if (overlay) {
            std::sort(overlay->elements.begin(), overlay->elements.end(), [](OverlayElementDescription const& left, OverlayElementDescription const& right) {
                return left.id < right.id;
            });
            _overlay = overlay;
        }
    } else {
        _simulationFacade->tryDrawVectorGraphics(
            worldRect.topLeft, worldRect.bottomRight, {viewSize.x, viewSize.y}, zoomFactor);
        _overlay = std::nullopt;
    }

    //draw overlay
    if (_overlay) {
        ImDrawList* drawList = ImGui::GetBackgroundDrawList();
        auto parameters = _simulationFacade->getSimulationParameters();
        auto timestep = _simulationFacade->getCurrentTimestep();
        for (auto const& overlayElement : _overlay->elements) {
            if (_cellDetailOverlayActive && overlayElement.cell) {
                {
                    auto fontSizeUnit = std::min(40.0f, Viewport::get().getZoomFactor()) / 2;
                    auto viewPos = Viewport::get().mapWorldToViewPosition({overlayElement.pos.x, overlayElement.pos.y + 0.3f}, parameters.borderlessRendering);
                    auto text = Const::CellTypeToStringMap.at(overlayElement.cellType);
                    drawList->AddText(
                        StyleRepository::get().getMediumFont(),
                        fontSizeUnit,
                        {viewPos.x - 1.7f * fontSizeUnit, viewPos.y},
                        Const::CellTypeOverlayShadowColor,
                        text.c_str());
                    drawList->AddText(
                        StyleRepository::get().getMediumFont(),
                        fontSizeUnit,
                        {viewPos.x - 1.7f * fontSizeUnit + 1, viewPos.y + 1},
                        Const::CellTypeOverlayColor,
                        text.c_str());
                }
            }

            if (overlayElement.selected == 1) {
                auto viewPos = Viewport::get().mapWorldToViewPosition({overlayElement.pos.x, overlayElement.pos.y}, parameters.borderlessRendering);
                if (Viewport::get().isVisible(viewPos)) {
                    drawList->AddCircle({viewPos.x, viewPos.y}, Viewport::get().getZoomFactor() * 0.45f, Const::SelectedCellOverlayColor, 0, 2.0f);
                }
            }
        }
    }
}

void SimulationView::updateMotionBlur()
{
    //motionBlurFactor = 0: max motion blur
    //motionBlurFactor = 1: no motion blur
    _shader->setFloat("motionBlurFactor", 1.0f / (1.0f + _motionBlur));
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

