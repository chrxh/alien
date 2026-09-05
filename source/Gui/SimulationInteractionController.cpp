#include "SimulationInteractionController.h"

#include <cmath>
#include <ranges>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/GlobalSettings.h>

#include <EngineInterface/SimulationFacade.h>

#include "CreatorWindow.h"
#include "EditorController.h"
#include "EditorModel.h"
#include "SimulationView.h"
#include "StyleRepository.h"
#include "Viewport.h"

namespace
{
    auto constexpr CursorRadius = 13.0f;
    auto constexpr EditToggleLeftMargin = 20.0f;
    auto constexpr EditToggleBottomMargin = 30.0f;
    auto constexpr EditToggleSize = 44.0f;
    auto constexpr EditToggleIconSize = 18.0f;
    // Corrects the side bearings of the glyph and its weight in the lower left corner
    auto constexpr EditToggleIconOffsetX = 2.0f;
    auto constexpr EditToggleIconOffsetY = 0.0f;
    auto constexpr EditToggleBorderThickness = 1.5f;
    auto constexpr EditToggleGlowSteps = 4;
    auto constexpr EditToggleGlowWidth = 2.0f;
    auto constexpr EditToggleLabelSpacing = 10.0f;
    auto constexpr EditToggleLabelPaddingX = 10.0f;
    auto constexpr EditToggleLabelPaddingY = 5.0f;
    auto constexpr EditToggleLabelRounding = 5.0f;
    auto constexpr EditToggleShortcutSpacing = 10.0f;
}

void SimulationInteractionController::init()
{
    setEditMode(GlobalSettings::get().getValue("controllers.simulation interaction.edit mode", _modes.editMode));
}

void SimulationInteractionController::shutdown()
{
    GlobalSettings::get().setValue("controllers.simulation interaction.edit mode", _modes.editMode);
}

void SimulationInteractionController::process()
{
    processEditWidget();

    if (_modes.editMode) {
        processSelectionRect();
    }
    if (!CreatorWindow::get().isOn() && _modes.interactionMode != InteractionMode_PositionSelection) {
        _modes.interactionMode = InteractionMode_Selection;
    }
    processEvents();
}

bool SimulationInteractionController::isEditMode() const
{
    return _modes.editMode;
}

void SimulationInteractionController::setEditMode(bool value)
{
    _modes.editMode = value;
    EditorController::get().setOn(_modes.editMode);
}

InteractionMode SimulationInteractionController::getInteractionMode() const
{
    return _modes.interactionMode;
}

void SimulationInteractionController::setInteractionMode(InteractionMode value)
{
    _modes.interactionMode = value;
}

std::optional<RealVector2D> SimulationInteractionController::getPositionSelectionData() const
{
    if (ImGui::GetIO().WantCaptureMouse) {
        return std::nullopt;
    }

    auto mousePos = ImGui::GetMousePos();
    return Viewport::get().mapViewToWorldPosition({mousePos.x, mousePos.y});
}

namespace
{
    void drawEditToggleLabel(ImDrawList* drawList, ImVec2 const& leftCenterPos)
    {
        auto text = "Edit mode";
        auto shortcut = "ALT+E";
        auto textSize = ImGui::CalcTextSize(text);
        auto shortcutSize = ImGui::CalcTextSize(shortcut);
        auto width = textSize.x + scale(EditToggleShortcutSpacing) + shortcutSize.x + 2 * scale(EditToggleLabelPaddingX);
        auto height = textSize.y + 2 * scale(EditToggleLabelPaddingY);

        auto topLeft = ImVec2{leftCenterPos.x, leftCenterPos.y - height / 2};
        auto bottomRight = ImVec2{topLeft.x + width, topLeft.y + height};
        drawList->AddRectFilled(topLeft, bottomRight, Const::FloatingCardBackgroundColor, scale(EditToggleLabelRounding));
        drawList->AddRect(topLeft, bottomRight, Const::FloatingCardBorderColor, scale(EditToggleLabelRounding));

        auto textPos = ImVec2{topLeft.x + scale(EditToggleLabelPaddingX), topLeft.y + scale(EditToggleLabelPaddingY)};
        drawList->AddText(textPos, Const::EditToggleLabelColor, text);
        drawList->AddText({textPos.x + textSize.x + scale(EditToggleShortcutSpacing), textPos.y}, Const::EditToggleShortcutColor, shortcut);
    }
}

void SimulationInteractionController::processEditWidget()
{
    if (!SimulationView::get().isRenderSimulation()) {
        return;
    }

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    auto anchorPos = ImVec2(viewport->Pos.x + scale(EditToggleLeftMargin), viewport->Pos.y + viewport->Size.y - scale(EditToggleBottomMargin));
    ImGui::SetNextWindowPos(anchorPos, ImGuiCond_Always, ImVec2(0.0f, 1.0f));

    ImGuiWindowFlags windowFlags = 0 | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar
        | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_AlwaysAutoResize;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("edit toggle", NULL, windowFlags);

    auto size = scale(EditToggleSize);
    auto pos = ImGui::GetCursorScreenPos();
    if (ImGui::InvisibleButton("editToggle", {size, size})) {
        setEditMode(!_modes.editMode);
    }
    auto hovered = ImGui::IsItemHovered();

    // The glow and the label reach beyond the auto-sized window
    auto drawList = ImGui::GetWindowDrawList();
    drawList->PushClipRectFullScreen();

    auto radius = size / 2;
    auto center = ImVec2{pos.x + radius, pos.y + radius};
    if (_modes.editMode) {
        for (auto step : std::views::iota(1, EditToggleGlowSteps + 1)) {
            auto glowColor = Const::EditToggleGlowColor;
            glowColor.Value.w *= 1.0f - toFloat(step - 1) / toFloat(EditToggleGlowSteps);
            drawList->AddCircle(center, radius + toFloat(step) * scale(EditToggleGlowWidth), glowColor, 0, scale(EditToggleGlowWidth));
        }
    }
    drawList->AddCircleFilled(center, radius, _modes.editMode ? Const::EditToggleSelectedColor : Const::EditToggleColor);
    drawList->AddCircle(
        center, radius, _modes.editMode ? Const::EditToggleSelectedBorderColor : Const::EditToggleBorderColor, 0, scale(EditToggleBorderThickness));

    auto iconColor = _modes.editMode ? Const::EditToggleSelectedIconColor : hovered ? Const::EditToggleHoveredIconColor : Const::EditToggleIconColor;
    auto iconFont = StyleRepository::get().getIconFont();
    auto iconFontSize = scale(EditToggleIconSize);
    auto iconSize = iconFont->CalcTextSizeA(iconFontSize, FLT_MAX, 0.0f, ICON_FA_EDIT);

    // ImGui truncates the text position, therefore it is rounded here to avoid a bias towards the upper left
    auto iconPos =
        ImVec2{std::round(center.x - iconSize.x / 2 + scale(EditToggleIconOffsetX)), std::round(center.y - iconSize.y / 2 + scale(EditToggleIconOffsetY))};
    drawList->AddText(iconFont, iconFontSize, iconPos, iconColor, ICON_FA_EDIT);

    if (hovered) {
        drawEditToggleLabel(drawList, {pos.x + size + scale(EditToggleLabelSpacing), center.y});
    }

    drawList->PopClipRect();
    ImGui::End();
    ImGui::PopStyleVar();
}

void SimulationInteractionController::processEvents()
{
    auto mousePos = ImGui::GetMousePos();
    IntVector2D mousePosInt{toInt(mousePos.x), toInt(mousePos.y)};
    IntVector2D prevMousePosInt = _prevMousePosInt ? *_prevMousePosInt : mousePosInt;

    if (!ImGui::GetIO().WantCaptureMouse && !SimulationView::get().isScrollbarDragging()) {
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            leftMouseButtonPressed(mousePosInt);
        }
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            leftMouseButtonHold(mousePosInt, prevMousePosInt);
        }
        if (ImGui::GetIO().MouseWheel > 0) {
            mouseWheelUp(mousePosInt, std::abs(ImGui::GetIO().MouseWheel));
        }

        if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
            rightMouseButtonPressed(mousePosInt);
        }
        if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
            rightMouseButtonHold(mousePosInt, prevMousePosInt);
        }
        if (ImGui::GetIO().MouseWheel < 0) {
            mouseWheelDown(mousePosInt, std::abs(ImGui::GetIO().MouseWheel));
        }

        if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
            middleMouseButtonPressed(mousePosInt);
        }
        if (ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
            middleMouseButtonHold(mousePosInt);
        }
        drawCursor();
    }
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        leftMouseButtonReleased(mousePosInt, prevMousePosInt);
    }
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
        rightMouseButtonReleased();
    }
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Middle)) {
        middleMouseButtonReleased();
    }

    processMouseWheel(mousePosInt);

    _prevMousePosInt = mousePosInt;
}

void SimulationInteractionController::leftMouseButtonPressed(IntVector2D const& mousePos)
{
    _modesAtClick = _modes;

    if (_modes.interactionMode == InteractionMode_PositionSelection) {
        _modes.interactionMode = InteractionMode_Selection;
        return;
    }

    if (!_modes.editMode) {
        _lastZoomTimepoint.reset();
        SimulationView::get().setMotionBlur(SimulationView::get().getMotionBlur() * 2);
    } else {
        if (!ImGui::GetIO().KeyAlt) {
            if (_modes.interactionMode == InteractionMode_PointPlacement) {
                CreatorWindow::get().onAddPoint(Viewport::get().mapViewToWorldPosition(toRealVector2D(mousePos)));
            } else if (_modes.interactionMode == InteractionMode_Drawing) {
                CreatorWindow::get().onDrawing();
            } else {
                EditorController::get().onSelectObjects(toRealVector2D(mousePos), ImGui::GetIO().KeyCtrl);
                _worldPosOnClick = Viewport::get().mapViewToWorldPosition(toRealVector2D(mousePos));
                if (_SimulationFacade::get()->isSimulationRunning()) {
                    _SimulationFacade::get()->setDetached(true);
                }

                auto shallowData = _SimulationFacade::get()->getSelectionShallowData();
                _selectionPositionOnClick = {shallowData.centerPosX, shallowData.centerPosY};
            }
        }
    }
}

void SimulationInteractionController::leftMouseButtonHold(IntVector2D const& mousePos, IntVector2D const& prevMousePos)
{
    if (_modesAtClick.interactionMode == InteractionMode_PositionSelection) {
        return;
    }

    if (!_modesAtClick.editMode) {
        Viewport::get().zoom(mousePos, calcZoomFactor(_lastZoomTimepoint ? *_lastZoomTimepoint : std::chrono::steady_clock::now()));
    } else if (_modesAtClick.interactionMode == InteractionMode_Drawing) {
        CreatorWindow::get().onDrawing();
    } else if (_modesAtClick.interactionMode == InteractionMode_Selection) {
        RealVector2D prevWorldPos = Viewport::get().mapViewToWorldPosition(toRealVector2D(prevMousePos));

        if (!_SimulationFacade::get()->isSimulationRunning()) {
            EditorController::get().onMoveSelectedObjects(toRealVector2D(mousePos), prevWorldPos);
        } else {
            EditorController::get().onFixateSelectedObjects(toRealVector2D(mousePos), *_worldPosOnClick, *_selectionPositionOnClick);
        }
    }
}

void SimulationInteractionController::mouseWheelUp(IntVector2D const& mousePos, float strongness)
{
    _mouseWheelAction =
        MouseWheelAction{.up = true, .strongness = strongness, .start = std::chrono::steady_clock::now(), .lastTime = std::chrono::steady_clock::now()};
}

void SimulationInteractionController::leftMouseButtonReleased(IntVector2D const& mousePos, IntVector2D const& prevMousePos)
{
    if (_modesAtClick.interactionMode == InteractionMode_PositionSelection) {
        return;
    }

    if (!_modesAtClick.editMode) {
        SimulationView::get().setMotionBlur(SimulationView::get().getMotionBlur() / 2);
    } else if (_modesAtClick.interactionMode == InteractionMode_Drawing) {
        CreatorWindow::get().finishDrawing();
    } else if (_modesAtClick.interactionMode == InteractionMode_Selection) {
        if (_SimulationFacade::get()->isSimulationRunning()) {
            _SimulationFacade::get()->setDetached(false);
            RealVector2D prevWorldPos = Viewport::get().mapViewToWorldPosition(toRealVector2D(prevMousePos));
            EditorController::get().onAccelerateSelectedObjects(toRealVector2D(mousePos), prevWorldPos);
        }
    }
}

void SimulationInteractionController::rightMouseButtonPressed(IntVector2D const& mousePos)
{
    _modesAtClick = _modes;

    if (_modes.interactionMode == InteractionMode_PositionSelection) {
        _modes.interactionMode = InteractionMode_Selection;
        return;
    }

    if (!_modes.editMode) {
        _lastZoomTimepoint.reset();
        SimulationView::get().setMotionBlur(SimulationView::get().getMotionBlur() * 2);
    } else {
        if (!ImGui::GetIO().KeyAlt) {
            if (_modes.interactionMode == InteractionMode_PointPlacement) {
                CreatorWindow::get().onRemoveLastPoint();
            } else if (_modes.interactionMode == InteractionMode_Selection && !_SimulationFacade::get()->isSimulationRunning()) {
                auto viewPos = toRealVector2D(mousePos);
                RealRect rect{viewPos, viewPos};
                _selectionRect = rect;
            }
        }
    }
}

void SimulationInteractionController::rightMouseButtonHold(IntVector2D const& mousePos, IntVector2D const& prevMousePos)
{
    if (_modesAtClick.interactionMode == InteractionMode_PositionSelection) {
        return;
    }

    if (!_modesAtClick.editMode) {
        Viewport::get().zoom(mousePos, 1.0f / calcZoomFactor(_lastZoomTimepoint ? *_lastZoomTimepoint : std::chrono::steady_clock::now()));
    } else if (_modesAtClick.interactionMode != InteractionMode_PointPlacement) {
        if (!ImGui::GetIO().KeyAlt) {
            auto isSimulationRunning = _SimulationFacade::get()->isSimulationRunning();
            if (!isSimulationRunning && _modesAtClick.interactionMode == InteractionMode_Selection && _selectionRect.has_value()) {
                _selectionRect->bottomRight = toRealVector2D(mousePos);
                EditorController::get().onUpdateSelectionRect(*_selectionRect);
            }
            if (isSimulationRunning) {
                RealVector2D prevWorldPos = Viewport::get().mapViewToWorldPosition(toRealVector2D(prevMousePos));
                EditorController::get().onApplyForces(toRealVector2D(mousePos), prevWorldPos);
            }
        }
    }
}

void SimulationInteractionController::mouseWheelDown(IntVector2D const& mousePos, float strongness)
{
    _mouseWheelAction =
        MouseWheelAction{.up = false, .strongness = strongness, .start = std::chrono::steady_clock::now(), .lastTime = std::chrono::steady_clock::now()};
}

void SimulationInteractionController::rightMouseButtonReleased()
{
    if (_modesAtClick.interactionMode == InteractionMode_PositionSelection) {
        return;
    }

    if (!_modesAtClick.editMode) {
        SimulationView::get().setMotionBlur(SimulationView::get().getMotionBlur() / 2);
    } else if (_modesAtClick.interactionMode != InteractionMode_PointPlacement) {
        if (!_SimulationFacade::get()->isSimulationRunning()) {
            _selectionRect.reset();
        }
    }
}

void SimulationInteractionController::processMouseWheel(IntVector2D const& mousePos)
{
    if (_mouseWheelAction) {
        auto zoomFactor = powf(calcZoomFactor(_mouseWheelAction->lastTime), 2.2f * _mouseWheelAction->strongness);
        auto now = std::chrono::steady_clock::now();
        _mouseWheelAction->lastTime = now;
        Viewport::get().zoom(mousePos, _mouseWheelAction->up ? zoomFactor : 1.0f / zoomFactor);
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - _mouseWheelAction->start).count() > 100) {
            _mouseWheelAction.reset();
        }
    }
}

void SimulationInteractionController::middleMouseButtonPressed(IntVector2D const& mousePos)
{
    _worldPosForPanning = Viewport::get().mapViewToWorldPosition({toFloat(mousePos.x), toFloat(mousePos.y)});
}

void SimulationInteractionController::middleMouseButtonHold(IntVector2D const& mousePos)
{
    Viewport::get().moveCenter(*_worldPosForPanning, mousePos);
}

void SimulationInteractionController::middleMouseButtonReleased()
{
    _worldPosForPanning.reset();
}

void SimulationInteractionController::drawCursor()
{
    auto mousePos = ImGui::GetMousePos();
    ImDrawList* drawList = ImGui::GetBackgroundDrawList();

    if (!ImGui::GetIO().WantCaptureMouse) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_None);
    }

    // Position selection cursor
    if (_modes.interactionMode == InteractionMode_PositionSelection) {
        auto cursorSize = scale(CursorRadius);

        // Shadow
        drawList->AddRectFilled(
            {mousePos.x - scale(2.0f), mousePos.y - cursorSize}, {mousePos.x + scale(2.0f), mousePos.y - cursorSize / 2}, Const::CursorShadowColor);
        drawList->AddRectFilled(
            {mousePos.x - scale(2.0f), mousePos.y + cursorSize / 2}, {mousePos.x + scale(2.0f), mousePos.y + cursorSize}, Const::CursorShadowColor);
        drawList->AddRectFilled(
            {mousePos.x - cursorSize, mousePos.y - scale(2.0f)}, {mousePos.x - cursorSize / 2, mousePos.y + scale(2.0f)}, Const::CursorShadowColor);
        drawList->AddRectFilled(
            {mousePos.x + cursorSize / 2, mousePos.y - scale(2.0f)}, {mousePos.x + cursorSize, mousePos.y + scale(2.0f)}, Const::CursorShadowColor);

        // Foreground
        drawList->AddRectFilled(
            {mousePos.x - scale(1.0f), mousePos.y - cursorSize}, {mousePos.x + scale(1.0f), mousePos.y - cursorSize / 2}, Const::CursorColor);
        drawList->AddRectFilled(
            {mousePos.x - scale(1.0f), mousePos.y + cursorSize / 2}, {mousePos.x + scale(1.0f), mousePos.y + cursorSize}, Const::CursorColor);
        drawList->AddRectFilled(
            {mousePos.x - cursorSize, mousePos.y - scale(1.0f)}, {mousePos.x - cursorSize / 2, mousePos.y + scale(1.0f)}, Const::CursorColor);
        drawList->AddRectFilled(
            {mousePos.x + cursorSize / 2, mousePos.y - scale(1.0f)}, {mousePos.x + cursorSize, mousePos.y + scale(1.0f)}, Const::CursorColor);
        return;
    }

    // Editing cursors
    if (_modes.editMode) {
        if (_modes.interactionMode != InteractionMode_Drawing) {
            auto cursorSize = scale(CursorRadius);

            // Shadow
            drawList->AddRectFilled(
                {mousePos.x - scale(2.0f), mousePos.y - cursorSize}, {mousePos.x + scale(2.0f), mousePos.y - cursorSize / 2}, Const::CursorShadowColor);
            drawList->AddRectFilled(
                {mousePos.x - scale(2.0f), mousePos.y + cursorSize / 2}, {mousePos.x + scale(2.0f), mousePos.y + cursorSize}, Const::CursorShadowColor);
            drawList->AddRectFilled(
                {mousePos.x - cursorSize, mousePos.y - scale(2.0f)}, {mousePos.x - cursorSize / 2, mousePos.y + scale(2.0f)}, Const::CursorShadowColor);
            drawList->AddRectFilled(
                {mousePos.x + cursorSize / 2, mousePos.y - scale(2.0f)}, {mousePos.x + cursorSize, mousePos.y + scale(2.0f)}, Const::CursorShadowColor);

            // Foreground
            drawList->AddRectFilled(
                {mousePos.x - scale(1.0f), mousePos.y - cursorSize}, {mousePos.x + scale(1.0f), mousePos.y - cursorSize / 2}, Const::CursorColor);
            drawList->AddRectFilled(
                {mousePos.x - scale(1.0f), mousePos.y + cursorSize / 2}, {mousePos.x + scale(1.0f), mousePos.y + cursorSize}, Const::CursorColor);
            drawList->AddRectFilled(
                {mousePos.x - cursorSize, mousePos.y - scale(1.0f)}, {mousePos.x - cursorSize / 2, mousePos.y + scale(1.0f)}, Const::CursorColor);
            drawList->AddRectFilled(
                {mousePos.x + cursorSize / 2, mousePos.y - scale(1.0f)}, {mousePos.x + cursorSize, mousePos.y + scale(1.0f)}, Const::CursorColor);
        } else {
            auto zoom = Viewport::get().getZoomFactor();
            auto radius = EditorModel::get().getPencilWidth() * zoom;
            drawList->AddCircleFilled(mousePos, radius, Const::ConstructionPreviewBrushColor);
        }
        return;
    }

    // Navigation cursor
    if (!_modes.editMode) {
        auto cursorSize = scale(CursorRadius);

        // Shadow
        drawList->AddCircle(mousePos, cursorSize / 2, Const::CursorShadowColor, 0, scale(4.0f));
        drawList->AddLine(
            {mousePos.x + sqrtf(2.0f) / 2.0f * cursorSize / 2, mousePos.y + sqrtf(2.0f) / 2.0f * cursorSize / 2},
            {mousePos.x + cursorSize, mousePos.y + cursorSize},
            Const::CursorShadowColor,
            scale(4.0f));

        // Foreground
        drawList->AddCircle(mousePos, cursorSize / 2, Const::CursorColor, 0, scale(2.0f));
        drawList->AddLine(
            {mousePos.x + sqrtf(2.0f) / 2.0f * cursorSize / 2, mousePos.y + sqrtf(2.0f) / 2.0f * cursorSize / 2},
            {mousePos.x + cursorSize, mousePos.y + cursorSize},
            Const::CursorColor,
            scale(2.0f));
    }
}

void SimulationInteractionController::processSelectionRect()
{
    if (_selectionRect) {
        ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
        auto startPos = _selectionRect->topLeft;
        auto endPos = _selectionRect->bottomRight;
        draw_list->AddRectFilled({startPos.x, startPos.y}, {endPos.x, endPos.y}, Const::SelectionAreaFillColor);
        draw_list->AddRect({startPos.x, startPos.y}, {endPos.x, endPos.y}, Const::SelectionAreaBorderColor, 0, 0, 1.0f);
    }
}

float SimulationInteractionController::calcZoomFactor(std::chrono::steady_clock::time_point const& lastTimepoint)
{
    auto now = std::chrono::steady_clock::now();
    auto duration = toFloat(std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTimepoint).count());
    _lastZoomTimepoint = now;
    return powf(Viewport::get().getZoomSensitivity(), duration / 15);
}
