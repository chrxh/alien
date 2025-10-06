#include "SimulationView.h"

#include <algorithm>
#include <glad/glad.h>
#include <imgui.h>

#include "Base/GlobalSettings.h"
#include "Base/Resources.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/SpaceCalculator.h"

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

    setupObjectShader();
    setupProcessingShader();

    _scrollbars = std::make_shared<_SimulationScrollbars>(true);

    resize(Viewport::get().getViewSize());

    // Setup shaders
    _postProcessingShader->use();
    _postProcessingShader->setInt("texture1", 0);
    _postProcessingShader->setInt("texture2", 1);
    _postProcessingShader->setInt("texture3", 2);
    _postProcessingShader->setBool("glowEffect", true);
    _postProcessingShader->setBool("motionEffect", true);
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
        glDeleteFramebuffers(1, &_objectFbo);
        glDeleteTextures(1, &_textureFramebufferId1);
        glDeleteTextures(1, &_textureFramebufferId2);
        glDeleteTextures(1, &_objectTexture);
        _areTexturesInitialized = true;
    }
    
    // Create texture for object rendering
    glGenTextures(1, &_objectTexture);
    glBindTexture(GL_TEXTURE_2D, _objectTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16, size.x, size.y, 0, GL_RGBA, GL_UNSIGNED_SHORT, NULL);

    // Create FBO for object rendering
    glGenFramebuffers(1, &_objectFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _objectFbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _objectTexture, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

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
        auto worldRect = Viewport::get().getVisibleWorldRect();
        auto viewSize = Viewport::get().getViewSize();
        auto zoomFactor = Viewport::get().getZoomFactor();

        // Extract object data from CUDA and transfer to OpenGL buffer
        _simulationFacade->tryDrawVectorGraphicsWithShaders(
            reinterpret_cast<void*>(uintptr_t(_objectShader->getVbo())), worldRect.topLeft, worldRect.bottomRight, zoomFactor);

        // Get number of objects to render
        int numObjects = _simulationFacade->getNumExtractedObjects();

        //GLint currentFbo;
        //glGetIntegerv(GL_FRAMEBUFFER_BINDING, &currentFbo);

        // Render objects to texture using shaders
        //glBindFramebuffer(GL_FRAMEBUFFER, _objectFbo);
        glViewport(0, 0, viewSize.x, viewSize.y);

        // Clear with black background
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Enable blending for anti-aliasing
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // Enable point sprites
        glEnable(GL_PROGRAM_POINT_SIZE);
        glEnable(GL_POINT_SPRITE);

        // Use object shader
        _objectShader->use();
        _objectShader->setFloat("zoom", static_cast<float>(zoomFactor));
        _objectShader->setVec2("worldSize", static_cast<float>(_simulationFacade->getWorldSize().x), static_cast<float>(_simulationFacade->getWorldSize().y));
        _objectShader->setVec2("rectUpperLeft", static_cast<float>(worldRect.topLeft.x), static_cast<float>(worldRect.topLeft.y));
        _objectShader->setVec2("viewportSize", static_cast<float>(viewSize.x), static_cast<float>(viewSize.y));

        // Draw points
        glBindVertexArray(_objectShader->getVao());
        glDrawArrays(GL_POINTS, 0, numObjects);

        // Disable blending and point sprites
        //    glDisable(GL_PROGRAM_POINT_SIZE);
        //glDisable(GL_BLEND);

        //glBindFramebuffer(GL_FRAMEBUFFER, currentFbo);

        //_shader->use();

        //GLint currentFbo;
        //glGetIntegerv(GL_FRAMEBUFFER_BINDING, &currentFbo);

        //// Post-processing pipeline (horizontal blur)
        //glBindFramebuffer(GL_FRAMEBUFFER, _fbo1);
        //_shader->setInt("phase", 0);
        //glBindVertexArray(_vao);
        //glActiveTexture(GL_TEXTURE0);
        //glBindTexture(GL_TEXTURE_2D, _objectTexture);
        //glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        //// Post-processing pipeline (vertical blur + mix)
        //glBindFramebuffer(GL_FRAMEBUFFER, _fbo2);
        //_shader->setInt("phase", 1);
        //glActiveTexture(GL_TEXTURE0);
        //glBindTexture(GL_TEXTURE_2D, _objectTexture);
        //glActiveTexture(GL_TEXTURE1);
        //glBindTexture(GL_TEXTURE_2D, _textureFramebufferId1);
        //glActiveTexture(GL_TEXTURE2);
        //glBindTexture(GL_TEXTURE_2D, _textureFramebufferId2);
        //glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        //// Final render to screen
        //glBindFramebuffer(GL_FRAMEBUFFER, currentFbo);
        //_shader->setInt("phase", 2);
        //glActiveTexture(GL_TEXTURE0);
        //glBindTexture(GL_TEXTURE_2D, _textureFramebufferId2);
        //glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

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
    _postProcessingShader->setFloat("brightness", value);
}

float SimulationView::getContrast() const
{
    return _contrast;
}

void SimulationView::setContrast(float value)
{
    _contrast = value;
    _postProcessingShader->setFloat("contrast", value);
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

void SimulationView::updateMotionBlur()
{
    //motionBlurFactor = 0: max motion blur
    //motionBlurFactor = 1: no motion blur
    _postProcessingShader->setFloat("motionBlurFactor", 1.0f / (1.0f + _motionBlur));
}

void SimulationView::setupObjectShader()
{
    _objectShader = std::make_shared<_Shader>(Const::ObjectVertexShader, Const::ObjectFragmentShader);

    auto vao = _objectShader->getVao();
    auto vbo = _objectShader->getVbo();

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 100000 * sizeof(RenderingObjectData), nullptr, GL_DYNAMIC_DRAW);

    // Setup vertex attributes for RenderingObjectData
    // Position (2 floats)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(RenderingObjectData), (void*)0);
    glEnableVertexAttribArray(0);

    // Color (3 floats)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(RenderingObjectData), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
}

void SimulationView::setupProcessingShader()
{
    _postProcessingShader = std::make_shared<_Shader>(Const::SimulationVertexShader, Const::SimulationFragmentShader);

    // Setup post-processing quad
    // Format: Positions (3 floats) annd texture coordinates (2 floats)
    float vertices[] = {
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

    auto vao = _postProcessingShader->getVao();
    auto vbo = _postProcessingShader->getVbo();
    auto ebo = _postProcessingShader->getEbo();
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

