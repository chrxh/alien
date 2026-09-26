#include "CreatorWidget.h"

#include <imgui.h>

#include <EngineInterface/SimulationFacade.h>

#include "AlienGui.h"
#include "EditorModel.h"
#include "EntityAttributeHelp.h"
#include "HelpStrings.h"

namespace
{
    auto constexpr RightColumnWidth = 140.0f;
    auto constexpr PointButtonWidth = 70.0f;
}

void CreatorWidget::process()
{
    auto& controller = CreatorController::get();
    auto parameters = controller.getParameters();
    auto tool = EditorModel::get().getTool();

    processColorWidget();
    if (tool == EditTool_Freehand) {
        processPencilWidget(parameters);
    }
    processMaterialWidgets(parameters);
    processShapeWidgets(parameters);
    if (tool != EditTool_Object && tool != EditTool_Freehand) {
        processObjectDistanceWidget(parameters);
    }
    if (tool != EditTool_Object) {
        processStickyWidget(parameters);
    }
    processStaticWidget(parameters);
    controller.setParameters(parameters);

    if (tool == EditTool_Line || tool == EditTool_Curve || tool == EditTool_Polygon) {
        processPointButtons();
    }
}

void CreatorWidget::processColorWidget()
{
    auto color = EditorModel::get().getDefaultColorCode();
    AlienGui::ComboColor(
        AlienGui::ComboColorParameters()
            .customizationColors(_SimulationFacade::get()->getSimulationParameters().customizationColors.value)
            .name("Color")
            .textWidth(RightColumnWidth)
            .tooltip(EntityAttributeHelp::get(EntityAttribute::Color)),
        color);
    EditorModel::get().setDefaultColorCode(color);
}

void CreatorWidget::processPencilWidget(CreatorParameters& parameters)
{
    AlienGui::SliderFloat(
        AlienGui::SliderFloatParameters()
            .name("Pencil radius")
            .min(1.0f)
            .max(8.0f)
            .textWidth(RightColumnWidth)
            .format("%.1f")
            .tooltip(Const::CreatorPencilRadiusTooltip),
        &parameters.pencilWidth);
}

void CreatorWidget::processMaterialWidgets(CreatorParameters& parameters)
{
    AlienGui::Switcher(
        AlienGui::SwitcherParameters()
            .name("Material")
            .textWidth(RightColumnWidth)
            .values({"Solid", "Fluid", "Free cells", "Energy particles"})
            .tooltip(Const::CreatorDrawingTypeTooltip),
        &parameters.material);
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name("Energy").format("%.2f").textWidth(RightColumnWidth).tooltip(Const::CellEnergyTooltip), parameters.energy);
    if (parameters.material == CreationMaterial_Fluid) {
        AlienGui::SliderFloat(
            AlienGui::SliderFloatParameters()
                .name("Glow")
                .min(0)
                .max(1.0f)
                .format("%.2f")
                .textWidth(RightColumnWidth)
                .tooltip(EntityAttributeHelp::get(EntityAttribute::FluidGlow)),
            &parameters.glow);
    }
    if (parameters.material != CreationMaterial_EnergyParticle && parameters.material != CreationMaterial_Fluid) {
        AlienGui::SliderFloat(
            AlienGui::SliderFloatParameters()
                .name("Stiffness")
                .max(1.0f)
                .min(0.0f)
                .textWidth(RightColumnWidth)
                .tooltip(EntityAttributeHelp::get(EntityAttribute::Stiffness)),
            &parameters.stiffness);
    }
}

void CreatorWidget::processShapeWidgets(CreatorParameters& parameters)
{
    auto tool = EditorModel::get().getTool();
    if (tool == EditTool_Rectangle) {
        AlienGui::InputInt(
            AlienGui::InputIntParameters().name("Horizontal objects").textWidth(RightColumnWidth).tooltip(Const::CreatorRectangleWidthTooltip),
            parameters.rectHorizontalObjects);
        AlienGui::InputInt(
            AlienGui::InputIntParameters().name("Vertical objects").textWidth(RightColumnWidth).tooltip(Const::CreatorRectangleHeightTooltip),
            parameters.rectVerticalObjects);
    }
    if (tool == EditTool_Hexagon) {
        AlienGui::InputInt(
            AlienGui::InputIntParameters().name("Layers").textWidth(RightColumnWidth).tooltip(Const::CreatorHexagonLayersTooltip), parameters.layers);
    }
    if (tool == EditTool_Disc) {
        AlienGui::InputFloat(
            AlienGui::InputFloatParameters().name("Outer radius").textWidth(RightColumnWidth).format("%.0f").tooltip(Const::CreatorDiscOuterRadiusTooltip),
            parameters.outerRadius);
        AlienGui::InputFloat(
            AlienGui::InputFloatParameters().name("Inner radius").textWidth(RightColumnWidth).format("%.0f").tooltip(Const::CreatorDiscInnerRadiusTooltip),
            parameters.innerRadius);
    }
}

void CreatorWidget::processObjectDistanceWidget(CreatorParameters& parameters)
{
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name("Object distance").format("%.2f").step(0.1).textWidth(RightColumnWidth).tooltip(Const::CreatorDistanceTooltip),
        parameters.objectDistance);
}

void CreatorWidget::processStickyWidget(CreatorParameters& parameters)
{
    AlienGui::Checkbox(
        AlienGui::CheckboxParameters().name("Sticky").textWidth(RightColumnWidth).tooltip(EntityAttributeHelp::get(EntityAttribute::Sticky)),
        &parameters.sticky);
}

void CreatorWidget::processStaticWidget(CreatorParameters& parameters)
{
    if (parameters.material != CreationMaterial_EnergyParticle) {
        AlienGui::Checkbox(
            AlienGui::CheckboxParameters().name("Static").textWidth(RightColumnWidth).tooltip(EntityAttributeHelp::get(EntityAttribute::Static)),
            &parameters.isStatic);
    }
}

void CreatorWidget::processPointButtons()
{
    auto& controller = CreatorController::get();
    AlienGui::Separator();

    ImGui::BeginDisabled(!controller.isFinishingPointsPossible());
    if (AlienGui::Button("Finish", PointButtonWidth)) {
        controller.onFinishPoints();
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(!controller.hasPoints());
    if (AlienGui::Button("Abort", PointButtonWidth)) {
        controller.onAbortPoints();
    }
    ImGui::EndDisabled();
}
