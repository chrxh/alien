#include "SimulationView.h"

#include <algorithm>
#include <glad/glad.h>
#include <imgui.h>
#include <cmath>

#include "Base/GlobalSettings.h"
#include "Base/Resources.h"
#include "EngineInterface/SimulationController.h"

#include "AlienImGui.h"
#include "Shader.h"
#include "SimulationScrollbar.h"
#include "Viewport.h"
#include "ModeController.h"
#include "StyleRepository.h"
#include "CellFunctionStrings.h"
#include "EditorModel.h"
#include "EngineInterface/Colors.h"

namespace
{
    auto constexpr MotionBlurStatic = 0.8f;
    auto constexpr MotionBlurMoving = 0.5f;
    auto constexpr ZoomFactorForOverlay = 12.0f;
    auto constexpr EditCursorRadius = 10.0f;
}

_SimulationView::_SimulationView(
    SimulationController const& simController,
    ModeController const& modeWindow,
    Viewport const& viewport,
    EditorModel const& editorModel)
    : _viewport(viewport)
    , _editorModel(editorModel)
{
    _isCellDetailOverlayActive = GlobalSettings::getInstance().getBoolState("settings.simulation view.overlay", _isCellDetailOverlayActive);
    _modeWindow = modeWindow;

    _simController = simController;
    _shader = std::make_shared<_Shader>(Const::SimulationVertexShader, Const::SimulationFragmentShader);

    _scrollbarX = std::make_shared<_SimulationScrollbar>(
        "SimScrollbarX", _SimulationScrollbar ::Orientation::Horizontal, _simController, _viewport);
    _scrollbarY = std::make_shared<_SimulationScrollbar>(
        "SimScrollbarY", _SimulationScrollbar::Orientation::Vertical, _simController, _viewport);

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
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

    resize(_viewport->getViewSize());

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

_SimulationView::~_SimulationView()
{
    GlobalSettings::getInstance().setBoolState("settings.simulation view.overlay", _isCellDetailOverlayActive);
}

void _SimulationView::resize(IntVector2D const& size)
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
    _simController->setImageResource(reinterpret_cast<void*>(uintptr_t(_textureSimulationId)));

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

    _viewport->setViewSize(size);
}

void _SimulationView::leftMouseButtonPressed(IntVector2D const& viewPos)
{
    _navigationState = NavigationState::Moving;
    _lastZoomTimepoint.reset();
    updateMotionBlur();
}

void _SimulationView::leftMouseButtonHold(IntVector2D const& viewPos, IntVector2D const& prevViewPos)
{
    if (_modeWindow->getMode() == _ModeController::Mode::Navigation) {
        _viewport->zoom(viewPos, calcZoomFactor(_lastZoomTimepoint ? *_lastZoomTimepoint : std::chrono::steady_clock::now()));
    }
}

void _SimulationView::mouseWheelUp(IntVector2D const& viewPos, float strongness)
{
    _mouseWheelAction =
        MouseWheelAction{.up = true, .strongness = strongness, .start = std::chrono::steady_clock::now(), .lastTime = std::chrono::steady_clock::now()};
}

void _SimulationView::leftMouseButtonReleased()
{
    _navigationState = NavigationState::Static;
    updateMotionBlur();
}

void _SimulationView::rightMouseButtonPressed()
{
    _navigationState = NavigationState::Moving;
    _lastZoomTimepoint.reset();
    updateMotionBlur();
}

void _SimulationView::rightMouseButtonHold(IntVector2D const& viewPos)
{
    if (_modeWindow->getMode() == _ModeController::Mode::Navigation) {
        _viewport->zoom(viewPos, 1.0f / calcZoomFactor(_lastZoomTimepoint ? *_lastZoomTimepoint : std::chrono::steady_clock::now()));
    }
}

void _SimulationView::mouseWheelDown(IntVector2D const& viewPos, float strongness)
{
    _mouseWheelAction =
        MouseWheelAction{.up = false, .strongness = strongness, .start = std::chrono::steady_clock::now(), .lastTime = std::chrono::steady_clock::now()};
}

void _SimulationView::rightMouseButtonReleased()
{
    _navigationState = NavigationState::Static;
    updateMotionBlur();
}

void _SimulationView::processMouseWheel(IntVector2D const& viewPos)
{
    if (_mouseWheelAction) {
        auto zoomFactor = powf(calcZoomFactor(_mouseWheelAction->lastTime), 2.2f * _mouseWheelAction->strongness);
        auto now = std::chrono::steady_clock::now();
        _mouseWheelAction->lastTime = now;
        _viewport->zoom(viewPos, _mouseWheelAction->up ? zoomFactor : 1.0f / zoomFactor);
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - _mouseWheelAction->start).count() > 100) {
            _mouseWheelAction.reset();
        }
    }
}

void _SimulationView::middleMouseButtonPressed(IntVector2D const& viewPos)
{
    _worldPosForMovement = _viewport->mapViewToWorldPosition({toFloat(viewPos.x), toFloat(viewPos.y)});
}

void _SimulationView::middleMouseButtonHold(IntVector2D const& viewPos)
{
    _viewport->centerTo(*_worldPosForMovement, viewPos);
}

void _SimulationView::middleMouseButtonReleased()
{
    _worldPosForMovement = std::nullopt;
}

void _SimulationView::processEvents()
{
    auto mousePos = ImGui::GetMousePos();
    IntVector2D mousePosInt{toInt(mousePos.x), toInt(mousePos.y)};
    IntVector2D prevMousePosInt = _prevMousePosInt ? *_prevMousePosInt : mousePosInt;

    if (!ImGui::GetIO().WantCaptureMouse) {
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            leftMouseButtonPressed(mousePosInt);
        }
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            leftMouseButtonHold(mousePosInt, prevMousePosInt);
        }
        if (ImGui::GetIO().MouseWheel > 0) {
            mouseWheelUp(mousePosInt, std::abs(ImGui::GetIO().MouseWheel));
        }
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            leftMouseButtonReleased();
        }

        if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
            rightMouseButtonPressed();
        }
        if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
            rightMouseButtonHold(mousePosInt);
        }
        if (ImGui::GetIO().MouseWheel < 0) {
            mouseWheelDown(mousePosInt, std::abs(ImGui::GetIO().MouseWheel));
        }
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
            rightMouseButtonReleased();
        }

        if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
            middleMouseButtonPressed(mousePosInt);
        }
        if (ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
            middleMouseButtonHold(mousePosInt);
        }
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Middle)) {
            middleMouseButtonReleased();
        }

        drawEditCursor();
    }
    processMouseWheel(mousePosInt);

    _prevMousePosInt = mousePosInt;
}

void _SimulationView::draw(bool renderSimulation)
{
    if (renderSimulation) {
        processEvents();

        updateImageFromSimulation();

        _shader->use();

        GLint currentFbo;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &currentFbo);

        glBindFramebuffer(GL_FRAMEBUFFER, _fbo1);
        _shader->setInt("phase", 0);
        glBindVertexArray(_vao);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, _textureSimulationId);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glBindFramebuffer(GL_FRAMEBUFFER, _fbo2);
        _shader->setInt("phase", 1);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, _textureSimulationId);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, _textureFramebufferId1);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, _textureFramebufferId2);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glBindFramebuffer(GL_FRAMEBUFFER, currentFbo);
        _shader->setInt("phase", 2);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, _textureFramebufferId2);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        ImDrawList* drawList = ImGui::GetBackgroundDrawList();
        auto p1 = _viewport->mapWorldToViewPosition({0, 0});
        auto worldSize = _simController->getWorldSize();
        auto p2 = _viewport->mapWorldToViewPosition(toRealVector2D(worldSize));
        auto color = ImColor::HSV(0.66f, 1.0f, 1.0f, 0.8f);
        drawList->AddLine({p1.x, p1.y}, {p2.x, p1.y}, color);
        drawList->AddLine({p2.x, p1.y}, {p2.x, p2.y}, color);
        drawList->AddLine({p2.x, p2.y}, {p1.x, p2.y}, color);
        drawList->AddLine({p1.x, p2.y}, {p1.x, p1.y}, color);

    } else {
        glClearColor(0, 0, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        auto textWidth = scale(300.0f);
        auto textHeight = scale(80.0f);
        ImDrawList* drawList = ImGui::GetBackgroundDrawList();
        auto& styleRep = StyleRepository::getInstance();
        auto right = ImGui::GetMainViewport()->Pos.x + ImGui::GetMainViewport()->Size.x;
        auto bottom = ImGui::GetMainViewport()->Pos.y + ImGui::GetMainViewport()->Size.y;
        auto maxLength = std::max(right, bottom);

        AlienImGui::RotateStart(ImGui::GetBackgroundDrawList());
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
        AlienImGui::RotateEnd(45.0f, ImGui::GetBackgroundDrawList());
    }
}

void _SimulationView::processControls(bool renderSimulation)
{
    if (renderSimulation) {
        ImGuiViewport* viewport = ImGui::GetMainViewport();
        auto mainMenubarHeight = scale(22);
        auto scrollbarThickness = 17;  //fixed
        _scrollbarX->process({{viewport->Pos.x, viewport->Size.y - scrollbarThickness}, {viewport->Size.x - 1 - scrollbarThickness, 1}});
        _scrollbarY->process({{viewport->Size.x - scrollbarThickness, viewport->Pos.y + mainMenubarHeight}, {1, viewport->Size.y - 1 - scrollbarThickness}});
    }
}

bool _SimulationView::isOverlayActive() const
{
    return _isCellDetailOverlayActive;
}

void _SimulationView::setOverlayActive(bool active)
{
    _isCellDetailOverlayActive = active;
}

void _SimulationView::setBrightness(float value)
{
    _shader->setFloat("brightness", value);
}

void _SimulationView::setContrast(float value)
{
    _shader->setFloat("contrast", value);
}

void _SimulationView::setMotionBlur(float value)
{
    _motionBlurFactor = value;
    updateMotionBlur();
}

void _SimulationView::updateImageFromSimulation()
{
    auto worldRect = _viewport->getVisibleWorldRect();
    auto viewSize = _viewport->getViewSize();
    auto zoomFactor = _viewport->getZoomFactor();

    if (zoomFactor >= ZoomFactorForOverlay) {
        auto overlay = _simController->tryDrawVectorGraphicsAndReturnOverlay(
            worldRect.topLeft, worldRect.bottomRight, {viewSize.x, viewSize.y}, zoomFactor);
        if (overlay) {
            std::sort(overlay->elements.begin(), overlay->elements.end(), [](OverlayElementDescription const& left, OverlayElementDescription const& right) {
                return left.id < right.id;
            });
            _overlay = overlay;
        }
    } else {
        _simController->tryDrawVectorGraphics(
            worldRect.topLeft, worldRect.bottomRight, {viewSize.x, viewSize.y}, zoomFactor);
        _overlay = std::nullopt;
    }

    //draw overlay
    if (_overlay) {
        ImDrawList* drawList = ImGui::GetBackgroundDrawList();
        for (auto const& overlayElement : _overlay->elements) {
            if (_isCellDetailOverlayActive && overlayElement.cell) {
                {
                    auto fontSize = std::min(40.0f, _viewport->getZoomFactor()) / 2;
                    auto viewPos = _viewport->mapWorldToViewPosition({overlayElement.pos.x, overlayElement.pos.y + 0.4f});
                    if (overlayElement.cellType != CellFunction_None) {
                        auto text = Const::CellFunctionToStringMap.at(overlayElement.cellType);
                        drawList->AddText(
                            StyleRepository::getInstance().getMediumFont(),
                            fontSize,
                            {viewPos.x - 2 * fontSize, viewPos.y},
                            Const::CellFunctionOverlayShadowColor,
                            text.c_str());
                        drawList->AddText(
                            StyleRepository::getInstance().getMediumFont(),
                            fontSize,
                            {viewPos.x - 2 * fontSize + 1, viewPos.y + 1},
                            Const::CellFunctionOverlayColor,
                            text.c_str());
                    }
                }
                {
                    auto viewPos = _viewport->mapWorldToViewPosition({overlayElement.pos.x - 0.12f, overlayElement.pos.y - 0.25f});
                    auto fontSize = _viewport->getZoomFactor() / 2;
                    drawList->AddText(
                        StyleRepository::getInstance().getLargeFont(),
                        fontSize,
                        {viewPos.x, viewPos.y},
                        Const::ExecutionNumberOverlayShadowColor,
                        std::to_string(overlayElement.executionOrderNumber).c_str());
                    drawList->AddText(
                        StyleRepository::getInstance().getLargeFont(),
                        fontSize,
                        {viewPos.x + 1, viewPos.y + 1},
                        Const::ExecutionNumberOverlayColor,
                        std::to_string(overlayElement.executionOrderNumber).c_str());
                }
            }

            if (overlayElement.selected == 1) {
                auto viewPos = _viewport->mapWorldToViewPosition({overlayElement.pos.x, overlayElement.pos.y});
                if (_viewport->isVisible(viewPos)) {
                    drawList->AddCircle({viewPos.x, viewPos.y}, _viewport->getZoomFactor() * 0.45f, Const::SelectedCellOverlayColor, 0, 2.0f);
                }
            }
        }
    }
}

void _SimulationView::updateMotionBlur()
{
    auto motionBlur = _navigationState == NavigationState::Static ? MotionBlurStatic : MotionBlurMoving;
    if (_motionBlurFactor == 0) {
        motionBlur = 1.0f;
    } else {
        motionBlur = std::min(1.0f, motionBlur / _motionBlurFactor);
    }
    _shader->setFloat("motionBlurFactor", motionBlur);
}

void _SimulationView::drawEditCursor()
{
    if (_modeWindow->getMode() == _ModeController::Mode::Editor) {
        auto mousePos = ImGui::GetMousePos();
        ImDrawList* drawList = ImGui::GetBackgroundDrawList();
        if (!_editorModel->isDrawMode() || _simController->isSimulationRunning()) {
            drawList->AddCircleFilled(mousePos, EditCursorRadius, Const::NavigationCursorColor);
        } else {
            auto radius = _editorModel->getPencilWidth() * _viewport->getZoomFactor();
            auto color = Const::IndividualCellColors[_editorModel->getDefaultColorCode()];
            float h, s, v;
            AlienImGui::ConvertRGBtoHSV(color, h, s, v);
            drawList->AddCircleFilled(mousePos, radius, ImColor::HSV(h, s, v, 0.6f));
        }
        if (!ImGui::GetIO().WantCaptureMouse) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_None);
        }
    }
}

float _SimulationView::calcZoomFactor(std::chrono::steady_clock::time_point const& lastTimepoint)
{
    auto now = std::chrono::steady_clock::now();
    auto duration = toFloat(std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTimepoint).count());
    _lastZoomTimepoint = now;
    return pow(_viewport->getZoomSensitivity(), duration / 15);
}

