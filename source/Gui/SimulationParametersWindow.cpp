#include "SimulationParametersWindow.h"

#include <ImFileDialog.h>
#include <imgui.h>
#include <Fonts/IconsFontAwesome5.h>

#include "Base/GlobalSettings.h"
#include "EngineInterface/SerializerService.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/LegacyAuxiliaryDataParserService.h"

#include "AlienImGui.h"
#include "CellFunctionStrings.h"
#include "GenericFileDialogs.h"
#include "HelpStrings.h"
#include "MessageDialog.h"
#include "SimulationInteractionController.h"
#include "RadiationSourcesWindow.h"
#include "OverlayMessageController.h"
#include "SimulationView.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr RightColumnWidth = 285.0f;

    template <int numRows, int numCols, typename T>
    std::vector<std::vector<T>> toVector(T const v[numRows][numCols])
    {
        std::vector<std::vector<T>> result;
        for (int row = 0; row < numRows; ++row) {
            std::vector<T> rowVector;
            for (int col = 0; col < numCols; ++col) {
                rowVector.emplace_back(v[row][col]);
            }
            result.emplace_back(rowVector);
        }
        return result;
    }

    template<int numElements>
    std::vector<float> toVector(float const v[numElements])
    {
        std::vector<float> result;
        for (int i = 0; i < numElements; ++i) {
            result.emplace_back(v[i]);
        }
        return result;
    }
}

void SimulationParametersWindow::init(SimulationFacade const& simulationFacade)
{
    _simulationFacade = simulationFacade;

    for (int n = 0; n < IM_ARRAYSIZE(_savedPalette); n++) {
        ImVec4 color;
        ImGui::ColorConvertHSVtoRGB(n / 31.0f, 0.8f, 0.2f, color.x, color.y, color.z);
        color.w = 1.0f; //alpha
        _savedPalette[n] = static_cast<ImU32>(ImColor(color));
    }

    auto path = std::filesystem::current_path();
    if (path.has_parent_path()) {
        path = path.parent_path();
    }
    _startingPath = GlobalSettings::get().getString("windows.simulation parameters.starting path", path.string());
    _featureListOpen = GlobalSettings::get().getBool("windows.simulation parameters.feature list.open", _featureListOpen);
    _featureListHeight = GlobalSettings::get().getFloat("windows.simulation parameters.feature list.height", _featureListHeight);

    for (int i = 0; i < CellFunction_Count; ++i) {
        _cellFunctionStrings.emplace_back(Const::CellFunctionToStringMap.at(i));
    }

}

void SimulationParametersWindow::shutdown()
{
    GlobalSettings::get().setString("windows.simulation parameters.starting path", _startingPath);
    GlobalSettings::get().setBool("windows.simulation parameters.feature list.open", _featureListOpen);
    GlobalSettings::get().setFloat("windows.simulation parameters.feature list.height", _featureListHeight);
}

SimulationParametersWindow::SimulationParametersWindow()
    : AlienWindow("Simulation parameters", "windows.simulation parameters", false)
{}

void SimulationParametersWindow::processIntern()
{
    processToolbar();
    if (ImGui::BeginChild("##Parameter", {0, _featureListOpen ? -scale(_featureListHeight) : -scale(50.0f)})) {
        processTabWidget();
    }
    ImGui::EndChild();
    processAddonList();
}

SimulationParametersSpot SimulationParametersWindow::createSpot(SimulationParameters const& simParameters, int index)
{
    auto worldSize = _simulationFacade->getWorldSize();
    SimulationParametersSpot spot;
    spot.posX = toFloat(worldSize.x / 2);
    spot.posY = toFloat(worldSize.y / 2);

    auto maxRadius = toFloat(std::min(worldSize.x, worldSize.y)) / 2;
    spot.shapeType = SpotShapeType_Circular;
    createDefaultSpotData(spot);
    spot.fadeoutRadius = maxRadius / 3;
    spot.color = _savedPalette[((2 + index) * 8) % IM_ARRAYSIZE(_savedPalette)];

    spot.values = simParameters.baseValues;
    return spot;
}

void SimulationParametersWindow::createDefaultSpotData(SimulationParametersSpot& spot)
{
    auto worldSize = _simulationFacade->getWorldSize();

    auto maxRadius = toFloat(std::min(worldSize.x, worldSize.y)) / 2;
    if (spot.shapeType == SpotShapeType_Circular) {
        spot.shapeData.circularSpot.coreRadius = maxRadius / 3;
    } else {
        spot.shapeData.rectangularSpot.height = maxRadius / 3;
        spot.shapeData.rectangularSpot.width = maxRadius / 3;
    }
}

void SimulationParametersWindow::processToolbar()
{
    if (AlienImGui::ToolbarButton(ICON_FA_FOLDER_OPEN)) {
        onOpenParameters();
    }
    AlienImGui::Tooltip("Open simulation parameters from file");

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_SAVE)) {
        onSaveParameters();
    }
    AlienImGui::Tooltip("Save simulation parameters to file");

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_COPY)) {
        _copiedParameters = _simulationFacade->getSimulationParameters();
        printOverlayMessage("Simulation parameters copied");
    }
    AlienImGui::Tooltip("Copy simulation parameters");

    ImGui::SameLine();
    ImGui::BeginDisabled(!_copiedParameters);
    if (AlienImGui::ToolbarButton(ICON_FA_PASTE)) {
        _simulationFacade->setSimulationParameters(*_copiedParameters);
        _simulationFacade->setOriginalSimulationParameters(*_copiedParameters);
        printOverlayMessage("Simulation parameters pasted");
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Paste simulation parameters");

    AlienImGui::Separator();
}

void SimulationParametersWindow::processTabWidget()
{
    auto currentSessionId = _simulationFacade->getSessionId();

    std::optional<bool> scheduleAppendTab;
    std::optional<int> scheduleDeleteTabAtIndex;
    
    if (ImGui::BeginChild("##", ImVec2(0, 0), false)) {

        if (ImGui::BeginTabBar("##Parameters", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown)) {
            auto parameters = _simulationFacade->getSimulationParameters();

            //add spot
            if (parameters.numSpots < MAX_SPOTS) {
                if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) {
                    scheduleAppendTab = true;
                }
                AlienImGui::Tooltip("Add parameter zone");
            }

            processBase();

            for (int tab = 0; tab < parameters.numSpots; ++tab) {
                if (!processSpot(tab)) {
                    scheduleDeleteTabAtIndex = tab;
                }
            }

            ImGui::EndTabBar();
        }
    }
    ImGui::EndChild();

    _focusBaseTab = !_sessionId.has_value() || currentSessionId != *_sessionId;
    _sessionId= currentSessionId;

    if (scheduleAppendTab.has_value()) {
        onAppendTab();
    }
    if (scheduleDeleteTabAtIndex.has_value()) {
        onDeleteTab(scheduleDeleteTabAtIndex.value());
    }
}

void SimulationParametersWindow::processBase()
{
    if (ImGui::BeginTabItem("Base", nullptr, _focusBaseTab ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None)) {
        auto parameters = _simulationFacade->getSimulationParameters();
        auto origParameters = _simulationFacade->getOriginalSimulationParameters();
        auto lastParameters = parameters;

        if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {

            /**
             * Rendering
             */
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Visualization"))) {
                AlienImGui::ColorButtonWithPicker(
                    AlienImGui::ColorButtonWithPickerParameters().name("Background color").textWidth(RightColumnWidth).defaultValue(origParameters.backgroundColor),
                    parameters.backgroundColor,
                    _backupColor,
                    _savedPalette);
                AlienImGui::Switcher(
                    AlienImGui::SwitcherParameters()
                        .name("Primary cell coloring")
                        .textWidth(RightColumnWidth)
                        .defaultValue(origParameters.cellColoring)
                        .values(
                            {"Energy",
                             "Standard cell colors",
                             "Mutants",
                             "Mutants and cell functions",
                             "Cell states",
                             "Genome complexities",
                             "Single cell function",
                             "All cell functions"})
                        .tooltip(Const::ColoringParameterTooltip),
                    parameters.cellColoring);
                if (parameters.cellColoring == CellColoring_CellFunction) {
                    AlienImGui::Switcher(
                        AlienImGui::SwitcherParameters()
                            .name("Highlighted cell function")
                            .textWidth(RightColumnWidth)
                            .defaultValue(origParameters.highlightedCellFunction)
                            .values(_cellFunctionStrings)
                            .tooltip("The specific cell function type to be highlighted can be selected here."),
                        parameters.highlightedCellFunction);
                }
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Cell radius")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(0.5f)
                        .defaultValue(&origParameters.cellRadius)
                        .tooltip("Specifies the radius of the drawn cells in unit length."),
                    &parameters.cellRadius);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Zoom level for cell activity")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(32.0f)
                        .infinity(true)
                        .defaultValue(&origParameters.zoomLevelNeuronalActivity)
                        .tooltip("The zoom level from which the neuronal activities become visible."),
                    &parameters.zoomLevelNeuronalActivity);
                AlienImGui::Checkbox(
                    AlienImGui::CheckboxParameters()
                        .name("Attack visualization")
                        .textWidth(RightColumnWidth)
                        .defaultValue(origParameters.attackVisualization)
                        .tooltip("If activated, successful attacks of attacker cells are visualized."),
                    parameters.attackVisualization);
                AlienImGui::Checkbox(
                    AlienImGui::CheckboxParameters()
                        .name("Muscle movement visualization")
                        .textWidth(RightColumnWidth)
                        .defaultValue(origParameters.muscleMovementVisualization)
                        .tooltip("If activated, the direction in which muscle cells are moving are visualized."),
                    parameters.muscleMovementVisualization);
                AlienImGui::Checkbox(
                    AlienImGui::CheckboxParameters()
                        .name("Borderless rendering")
                        .textWidth(RightColumnWidth)
                        .defaultValue(origParameters.borderlessRendering)
                        .tooltip("If activated, the simulation is rendered periodically in the view port."),
                    parameters.borderlessRendering);
                AlienImGui::Checkbox(
                    AlienImGui::CheckboxParameters()
                        .name("Adaptive space grid")
                        .textWidth(RightColumnWidth)
                        .defaultValue(origParameters.gridLines)
                        .tooltip("Draws a suitable grid in the background depending on the zoom level."),
                    parameters.gridLines);
                AlienImGui::Checkbox(
                    AlienImGui::CheckboxParameters()
                        .name("Mark reference domain")
                        .textWidth(RightColumnWidth)
                        .defaultValue(origParameters.markReferenceDomain)
                        .tooltip("Draws borders along the world before it repeats itself."),
                    parameters.markReferenceDomain);
                AlienImGui::EndTreeNode();
            }

            /**
             * Numerics
             */
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Numerics"))) {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Time step size")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(1.0f)
                        .defaultValue(&origParameters.timestepSize)
                        .tooltip(std::string("The time duration calculated in a single simulation step. Smaller values increase the accuracy of the simulation "
                                             "while larger values can lead to numerical instabilities.")),
                    &parameters.timestepSize);
                AlienImGui::EndTreeNode();
            }

            /**
             * Physics: Motion
             */
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Physics: Motion"))) {
                if (AlienImGui::Switcher(
                        AlienImGui::SwitcherParameters()
                            .name("Motion type")
                            .textWidth(RightColumnWidth)
                            .defaultValue(origParameters.motionType)
                            .values({"Fluid dynamics", "Collision-based"})
                            .tooltip(std::string(
                                "The algorithm for the particle motions is defined here. If 'Fluid dynamics' is selected, an SPH fluid solver is used for the "
                                "calculation of the forces. The particles then behave like (compressible) liquids or gases. The other option 'Collision-based' "
                                "calculates the forces based on particle collisions and should be preferred for mechanical simulation with solids.")),
                        parameters.motionType)) {
                    if (parameters.motionType == MotionType_Fluid) {
                        parameters.motionData.fluidMotion = FluidMotion();
                    } else {
                        parameters.motionData.collisionMotion = CollisionMotion();
                    }
                }
                if (parameters.motionType == MotionType_Fluid) {
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Smoothing length")
                            .textWidth(RightColumnWidth)
                            .min(0)
                            .max(3.0f)
                            .defaultValue(&origParameters.motionData.fluidMotion.smoothingLength)
                            .tooltip(std::string("The smoothing length determines the region of influence of the neighboring particles for the calculation of "
                                                 "density, pressure and viscosity. Values that are too small lead to numerical instabilities, while values that "
                                                 "are too large cause the particles to drift apart.")),
                        &parameters.motionData.fluidMotion.smoothingLength);
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Pressure")
                            .textWidth(RightColumnWidth)
                            .min(0)
                            .max(0.3f)
                            .defaultValue(&origParameters.motionData.fluidMotion.pressureStrength)
                            .tooltip(std::string("This parameter allows to control the strength of the pressure.")),
                        &parameters.motionData.fluidMotion.pressureStrength);
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Viscosity")
                            .textWidth(RightColumnWidth)
                            .min(0)
                            .max(0.3f)
                            .defaultValue(&origParameters.motionData.fluidMotion.viscosityStrength)
                            .tooltip(std::string("This parameter be used to control the strength of the viscosity. Larger values lead to a smoother movement.")),
                        &parameters.motionData.fluidMotion.viscosityStrength);
                } else {
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Repulsion strength")
                            .textWidth(RightColumnWidth)
                            .min(0)
                            .max(0.3f)
                            .defaultValue(&origParameters.motionData.collisionMotion.cellRepulsionStrength)
                            .tooltip(std::string("The strength of the repulsive forces, between two cells that are not connected.")),
                        &parameters.motionData.collisionMotion.cellRepulsionStrength);
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Maximum collision distance")
                            .textWidth(RightColumnWidth)
                            .min(0)
                            .max(3.0f)
                            .defaultValue(&origParameters.motionData.collisionMotion.cellMaxCollisionDistance)
                            .tooltip(std::string("Maximum distance up to which a collision of two cells is possible.")),
                        &parameters.motionData.collisionMotion.cellMaxCollisionDistance);
                }
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Friction")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(1.0f)
                        .logarithmic(true)
                        .format("%.4f")
                        .defaultValue(&origParameters.baseValues.friction)
                        .tooltip(std::string("This specifies the fraction of the velocity that is slowed down per time step.")),
                    &parameters.baseValues.friction);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Rigidity")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(1.0f)
                        .format("%.2f")
                        .defaultValue(&origParameters.baseValues.rigidity)
                        .tooltip(std::string(
                            "Controls the rigidity of connected cells.\nA higher value will cause connected cells to move more uniformly as a rigid body.")),
                    &parameters.baseValues.rigidity);
                AlienImGui::EndTreeNode();
            }

            /**
             * Physics: Thresholds
             */
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Physics: Thresholds"))) {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Maximum velocity")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(6.0f)
                        .defaultValue(&origParameters.cellMaxVelocity)
                        .tooltip(std::string("Maximum velocity that a cell can reach.")),
                    &parameters.cellMaxVelocity);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Maximum force")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(3.0f)
                        .colorDependence(true)
                        .defaultValue(origParameters.baseValues.cellMaxForce)
                        .tooltip(std::string("Maximum force that can be applied to a cell without causing it to disintegrate.")),
                    parameters.baseValues.cellMaxForce);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Minimum distance")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(1.0f)
                        .defaultValue(&origParameters.cellMinDistance)
                        .tooltip(std::string("Minimum distance between two cells.")),
                    &parameters.cellMinDistance);
                AlienImGui::EndTreeNode();
            }

            /**
             * Physics: Binding
             */
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Physics: Binding"))) {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Maximum distance")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(5.0f)
                        .colorDependence(true)
                        .defaultValue(origParameters.cellMaxBindingDistance)
                        .tooltip(std::string("Maximum distance up to which a connection of two cells is possible.")),
                    parameters.cellMaxBindingDistance);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Fusion velocity")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(2.0f)
                        .defaultValue(&origParameters.baseValues.cellFusionVelocity)
                        .tooltip(std::string("Minimum relative velocity of two colliding cells so that a connection can be established.")),
                    &parameters.baseValues.cellFusionVelocity);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Maximum energy")
                        .textWidth(RightColumnWidth)
                        .min(50.0f)
                        .max(10000000.0f)
                        .logarithmic(true)
                        .infinity(true)
                        .format("%.0f")
                        .defaultValue(&origParameters.baseValues.cellMaxBindingEnergy)
                        .tooltip(std::string("Maximum energy of a cell at which it can contain bonds to adjacent cells. If the energy of a cell exceeds this "
                                             "value, all bonds will be destroyed.")),
                    &parameters.baseValues.cellMaxBindingEnergy);
                AlienImGui::EndTreeNode();
            }

            /**
             * Physics: Radiation
             */
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Physics: Radiation"))) {
                if (AlienImGui::Button(AlienImGui::ButtonParameters()
                                           .buttonText("Open editor")
                                           .name("Radiation sources")
                                           .textWidth(RightColumnWidth)
                                           .showDisabledRevertButton(true)
                            .tooltip("If no radiation source is specified, the cells emit energy particles at their respective positions. If, on the other hand, "
                                     "one or more radiation sources are defined, the energy particles emitted by cells are created at these sources."))) {
                    RadiationSourcesWindow::get().setOn(true);
                }

                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Absorption factor")
                        .textWidth(RightColumnWidth)
                        .logarithmic(true)
                        .colorDependence(true)
                        .min(0)
                        .max(1.0)
                        .format("%.4f")
                        .defaultValue(origParameters.baseValues.radiationAbsorption)
                        .tooltip("The fraction of energy that a cell can absorb from an incoming energy particle can be specified here."),
                    parameters.baseValues.radiationAbsorption);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Radiation type I: Strength")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(0)
                        .max(0.01f)
                        .logarithmic(true)
                        .format("%.6f")
                        .defaultValue(origParameters.baseValues.radiationCellAgeStrength)
                        .tooltip("Indicates how energetic the emitted particles of aged cells are."),
                    parameters.baseValues.radiationCellAgeStrength);
                AlienImGui::SliderInt(
                    AlienImGui::SliderIntParameters()
                        .name("Radiation type I: Minimum age")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .infinity(true)
                        .min(0)
                        .max(10000000)
                        .logarithmic(true)
                        .defaultValue(origParameters.radiationMinCellAge)
                        .tooltip("The minimum age of a cell can be defined here, from which it emits energy particles."),
                    parameters.radiationMinCellAge);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Radiation type II: Strength")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(0)
                        .max(0.01f)
                        .logarithmic(true)
                        .format("%.6f")
                        .defaultValue(origParameters.highRadiationFactor)
                        .tooltip("Indicates how energetic the emitted particles of high energy cells are."),
                    parameters.highRadiationFactor);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Radiation type II: Energy threshold")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .infinity(true)
                        .min(0)
                        .max(100000.0f)
                        .logarithmic(true)
                        .format("%.1f")
                        .defaultValue(origParameters.highRadiationMinCellEnergy)
                        .tooltip("The minimum energy of a cell can be defined here, from which it emits energy particles."),
                    parameters.highRadiationMinCellEnergy);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Minimum split energy")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .infinity(true)
                        .min(1.0f)
                        .max(10000.0f)
                        .logarithmic(true)
                        .format("%.0f")
                        .defaultValue(origParameters.particleSplitEnergy)
                        .tooltip("The minimum energy of an energy particle after which it can split into two particles, whereby it receives a small momentum. The "
                                 "splitting does not occur immediately, but only after a certain time."),
                    parameters.particleSplitEnergy);
                AlienImGui::Checkbox(
                    AlienImGui::CheckboxParameters()
                        .name("Energy to cell transformation")
                        .textWidth(RightColumnWidth)
                        .defaultValue(origParameters.particleTransformationAllowed)
                        .tooltip("If activated, an energy particle will transform into a cell if the energy of the particle exceeds the normal energy value."),
                    parameters.particleTransformationAllowed);

                AlienImGui::EndTreeNode();
            }

            /**
             * Cell life cycle
             */
            ImGui::PushID("Transformation");
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Cell life cycle"))) {
                AlienImGui::SliderInt(
                    AlienImGui::SliderIntParameters()
                        .name("Maximum age")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .logarithmic(true)
                        .infinity(true)
                        .min(1)
                        .max(10000000)
                        .defaultValue(origParameters.cellMaxAge)
                        .tooltip("Defines the maximum age of a cell. If a cell exceeds this age it will be transformed to an energy particle."),
                    parameters.cellMaxAge);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Minimum energy")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(10.0f)
                        .max(200.0f)
                        .defaultValue(origParameters.baseValues.cellMinEnergy)
                        .tooltip("Minimum energy a cell needs to exist."),
                    parameters.baseValues.cellMinEnergy);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Normal energy")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(10.0f)
                        .max(200.0f)
                        .defaultValue(origParameters.cellNormalEnergy)
                        .tooltip("The normal energy value of a cell is defined here. This is used as a reference value in various contexts: \n\n" ICON_FA_CHEVRON_RIGHT
                            " Attacker and Transmitter cells: When the energy of these cells is above the normal value, some of their energy is distributed to "
                                 "surrounding cells.\n\n" ICON_FA_CHEVRON_RIGHT
                            " Constructor cells: Creating new cells costs energy. The creation of new cells is executed only when the "
                                 "residual energy of the constructor cell does not fall below the normal value.\n\n" ICON_FA_CHEVRON_RIGHT
                            " If the transformation of energy particles to "
                                 "cells is activated, an energy particle will transform into a cell if the energy of the particle exceeds the normal value."),
                    parameters.cellNormalEnergy);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Decay rate of dying cells")
                        .colorDependence(true)
                        .textWidth(RightColumnWidth)
                        .min(1e-6f)
                        .max(0.1f)
                        .format("%.6f")
                        .logarithmic(true)
                        .defaultValue(origParameters.cellDeathProbability)
                        .tooltip("The probability per time step with which a cell will disintegrate (i.e. transform into an energy particle) when it is in the "
                                    "state 'Dying'. This can occur when one of the following conditions is satisfied:\n\n"
                                    ICON_FA_CHEVRON_RIGHT " The cell has too low energy.\n\n"
                                    ICON_FA_CHEVRON_RIGHT " The cell has exceeded its maximum age."),
                    parameters.cellDeathProbability);
                AlienImGui::Switcher(
                    AlienImGui::SwitcherParameters()
                        .name("Cell death consequences")
                        .textWidth(RightColumnWidth)
                        .defaultValue(origParameters.cellDeathConsequences)
                        .values({"None", "Entire creature dies", "Detached creature parts die"})
                        .tooltip("Here one can define what happens to the organism when one of its cells is in the 'Dying' state.\n\n" ICON_FA_CHEVRON_RIGHT
                                 " None: Only the cell dies.\n\n" ICON_FA_CHEVRON_RIGHT " Entire creature dies: All the cells of the organism will also die.\n\n" ICON_FA_CHEVRON_RIGHT
                                 " Detached creature parts die: Only the parts of the organism that are no longer connected to a "
                                 "constructor cell for self-replication die."),
                    parameters.cellDeathConsequences);
                AlienImGui::EndTreeNode();
            }
            ImGui::PopID();

            /**
             * Mutation 
             */
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Genome copy mutations"))) {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Neural net")
                        .textWidth(RightColumnWidth)
                        .min(0.0f)
                        .max(1.0f)
                        .format("%.7f")
                        .logarithmic(true)
                        .colorDependence(true)
                        .defaultValue(origParameters.baseValues.cellCopyMutationNeuronData)
                        .tooltip("This type of mutation changes a weight or a bias of the neural networks of a single neuron cell encoded in the genome. The "
                                 "probability of a change is given by the specified value times the number of coded cells in the genome."),
                    parameters.baseValues.cellCopyMutationNeuronData);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Cell properties")
                        .textWidth(RightColumnWidth)
                        .min(0.0f)
                        .max(1.0f)
                        .format("%.7f")
                        .logarithmic(true)
                        .colorDependence(true)
                        .defaultValue(origParameters.baseValues.cellCopyMutationCellProperties)
                        .tooltip("This type of mutation changes a random property (e.g. (input) execution order number, required energy, block output and "
                                 "function-specific properties such as minimum density for sensors, neural net weights etc.). The spatial structure, color, cell "
                                 "function type and self-replication capabilities are not changed. The probability of a change is given by the specified value "
                                 "times the number of coded cells in the genome."),
                    parameters.baseValues.cellCopyMutationCellProperties);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Geometry")
                        .textWidth(RightColumnWidth)
                        .min(0.0f)
                        .max(1.0f)
                        .format("%.7f")
                        .logarithmic(true)
                        .colorDependence(true)
                        .defaultValue(origParameters.baseValues.cellCopyMutationGeometry)
                        .tooltip("This type of mutation changes the geometry type, connection distance, stiffness and single construction flag. The probability of "
                                 "a change is given by the specified value times the number of coded cells in the genome."),
                    parameters.baseValues.cellCopyMutationGeometry);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Custom geometry")
                        .textWidth(RightColumnWidth)
                        .min(0.0f)
                        .max(1.0f)
                        .format("%.7f")
                        .logarithmic(true)
                        .colorDependence(true)
                        .defaultValue(origParameters.baseValues.cellCopyMutationCustomGeometry)
                        .tooltip("This type of mutation only changes angles and required connections of custom geometries. The probability of a change is given by "
                                 "the specified value times the number of coded cells in the genome."),
                    parameters.baseValues.cellCopyMutationCustomGeometry);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Cell function type")
                        .textWidth(RightColumnWidth)
                        .min(0.0f)
                        .max(1.0f)
                        .format("%.7f")
                        .logarithmic(true)
                        .colorDependence(true)
                        .defaultValue(origParameters.baseValues.cellCopyMutationCellFunction)
                        .tooltip("This type of mutation changes the type of cell function. The changed cell function will have random properties. The probability "
                                 "of a change is given by the specified value times the number of coded cells in the genome. If the flag 'Preserve "
                                 "self-replication' is disabled it can also alter self-replication capabilities by changing a constructor to "
                                 "something else or vice versa."),
                    parameters.baseValues.cellCopyMutationCellFunction);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Insertion")
                        .textWidth(RightColumnWidth)
                        .min(0.0f)
                        .max(1.0f)
                        .format("%.7f")
                        .logarithmic(true)
                        .colorDependence(true)
                        .defaultValue(origParameters.baseValues.cellCopyMutationInsertion)
                        .tooltip("This type of mutation inserts a new cell description to the genome at a random position. The probability of a change is given by "
                                 "the specified value times the number of coded cells in the genome."),
                    parameters.baseValues.cellCopyMutationInsertion);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Deletion")
                        .textWidth(RightColumnWidth)
                        .min(0.0f)
                        .max(1.0f)
                        .format("%.7f")
                        .logarithmic(true)
                        .colorDependence(true)
                        .defaultValue(origParameters.baseValues.cellCopyMutationDeletion)
                        .tooltip("This type of mutation deletes a cell description from the genome at a random position. The probability of a change is given by "
                                 "the specified value times the number of coded cells in the genome."),
                    parameters.baseValues.cellCopyMutationDeletion);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Translation")
                        .textWidth(RightColumnWidth)
                        .min(0.0f)
                        .max(1.0f)
                        .format("%.7f")
                        .logarithmic(true)
                        .colorDependence(true)
                        .defaultValue(origParameters.baseValues.cellCopyMutationTranslation)
                        .tooltip("This type of mutation moves a block of cell descriptions from the genome at a random position to a new random position."),
                    parameters.baseValues.cellCopyMutationTranslation);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Duplication")
                        .textWidth(RightColumnWidth)
                        .min(0.0f)
                        .max(1.0f)
                        .format("%.7f")
                        .logarithmic(true)
                        .colorDependence(true)
                        .defaultValue(origParameters.baseValues.cellCopyMutationDuplication)
                        .tooltip("This type of mutation copies a block of cell descriptions from the genome at a random position to a new random position."),
                    parameters.baseValues.cellCopyMutationDuplication);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Individual cell color")
                        .textWidth(RightColumnWidth)
                        .min(0.0f)
                        .max(1.0f)
                        .format("%.7f")
                        .logarithmic(true)
                        .colorDependence(true)
                        .defaultValue(origParameters.baseValues.cellCopyMutationCellColor)
                        .tooltip("This type of mutation alters the color of a single cell descriptions in a genome by using the specified color transitions. The "
                                 "probability of a change is given by the specified value times the number of coded cells in the genome."),
                    parameters.baseValues.cellCopyMutationCellColor);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Sub-genome color")
                        .textWidth(RightColumnWidth)
                        .min(0.0f)
                        .max(1.0f)
                        .format("%.7f")
                        .logarithmic(true)
                        .colorDependence(true)
                        .defaultValue(origParameters.baseValues.cellCopyMutationSubgenomeColor)
                        .tooltip("This type of mutation alters the color of all cell descriptions in a sub-genome by using the specified color transitions."),
                    parameters.baseValues.cellCopyMutationSubgenomeColor);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Genome color")
                        .textWidth(RightColumnWidth)
                        .min(0.0f)
                        .max(1.0f)
                        .format("%.7f")
                        .logarithmic(true)
                        .colorDependence(true)
                        .defaultValue(origParameters.baseValues.cellCopyMutationGenomeColor)
                        .tooltip(
                            "This type of mutation alters the color of all cell descriptions in a genome by using the specified color transitions."),
                    parameters.baseValues.cellCopyMutationGenomeColor);
                AlienImGui::CheckboxColorMatrix(
                    AlienImGui::CheckboxColorMatrixParameters()
                        .name("Color transitions")
                        .textWidth(RightColumnWidth)
                        .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origParameters.cellFunctionConstructorMutationColorTransitions))
                        .tooltip(
                            "The color transitions are used for color mutations. The row index indicates the source color and the column index the target color."),
                    parameters.cellFunctionConstructorMutationColorTransitions);
                AlienImGui::Checkbox(
                    AlienImGui::CheckboxParameters()
                        .name("Prevent genome depth increase")
                        .textWidth(RightColumnWidth)
                        .defaultValue(origParameters.cellFunctionConstructorMutationPreventDepthIncrease)
                        .tooltip(std::string("A genome has a tree-like structure because it can contain sub-genomes. If this flag is activated, the mutations will "
                                             "not increase the depth of the genome structure.")),
                    parameters.cellFunctionConstructorMutationPreventDepthIncrease);
                auto preserveSelfReplication = !parameters.cellFunctionConstructorMutationSelfReplication;
                AlienImGui::Checkbox(
                    AlienImGui::CheckboxParameters()
                        .name("Preserve self-replication")
                        .textWidth(RightColumnWidth)
                        .defaultValue(!origParameters.cellFunctionConstructorMutationSelfReplication)
                        .tooltip("If deactivated, a mutation can also alter self-replication capabilities in the genome by changing a constructor cell to "
                                 "something else or vice versa."),
                    preserveSelfReplication);
                parameters.cellFunctionConstructorMutationSelfReplication = !preserveSelfReplication;
                AlienImGui::EndTreeNode();
            }

            /**
             * Attacker
             */
            ImGui::PushID("Attacker");
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Cell function: Attacker"))) {
                AlienImGui::InputFloatColorMatrix(
                    AlienImGui::InputFloatColorMatrixParameters()
                        .name("Food chain color matrix")
                        .max(1)
                        .textWidth(RightColumnWidth)
                        .tooltip(
                            "This matrix can be used to determine how well one cell can attack another cell. The color of the attacking cell correspond to the row "
                            "number and the color of the attacked cell to the column number. A value of 0 means that the attacked cell cannot be digested, "
                            "i.e. no energy can be obtained. A value of 1 means that the maximum energy can be obtained in the digestion process.\n\nExample: If a "
                            "zero is entered in row 2 (red) and column 3 (green), it means that red cells cannot eat green cells.")
                        .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origParameters.baseValues.cellFunctionAttackerFoodChainColorMatrix)),
                    parameters.baseValues.cellFunctionAttackerFoodChainColorMatrix);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Energy cost")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(0)
                        .max(1.0f)
                        .format("%.5f")
                        .logarithmic(true)
                        .defaultValue(origParameters.baseValues.cellFunctionAttackerEnergyCost)
                        .tooltip("Amount of energy lost by an attempted attack of a cell in form of emitted energy particles."),
                    parameters.baseValues.cellFunctionAttackerEnergyCost);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Attack strength")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .logarithmic(true)
                        .min(0)
                        .max(0.5f)
                        .defaultValue(origParameters.cellFunctionAttackerStrength)
                        .tooltip("Indicates the portion of energy through which a successfully attacked cell is weakened. However, this energy portion can be "
                                 "influenced by other factors adjustable within the attacker's simulation parameters."),
                    parameters.cellFunctionAttackerStrength);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Attack radius")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(0)
                        .max(3.0f)
                        .defaultValue(origParameters.cellFunctionAttackerRadius)
                        .tooltip("The maximum distance over which an attacker cell can attack another cell."),
                    parameters.cellFunctionAttackerRadius);
                AlienImGui::Checkbox(
                    AlienImGui::CheckboxParameters()
                        .name("Destroy cells")
                        .textWidth(RightColumnWidth)
                        .defaultValue(origParameters.cellFunctionAttackerDestroyCells)
                        .tooltip(
                            "If activated, the attacker cell is able to destroy other cells. If deactivated, it only damages them."),
                    parameters.cellFunctionAttackerDestroyCells);
                AlienImGui::EndTreeNode();
            }
            ImGui::PopID();

            /**
             * Constructor
             */
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Cell function: Constructor"))) {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Offspring distance")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(0.1f)
                        .max(3.0f)
                        .defaultValue(origParameters.cellFunctionConstructorOffspringDistance)
                        .tooltip("The distance of the constructed cell from the constructor cell."),
                    parameters.cellFunctionConstructorOffspringDistance);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Connection distance")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(0.1f)
                        .max(3.0f)
                        .defaultValue(origParameters.cellFunctionConstructorConnectingCellMaxDistance)
                        .tooltip("The constructor can automatically connect constructed cells to other cells in the vicinity within this distance."),
                    parameters.cellFunctionConstructorConnectingCellMaxDistance);
                AlienImGui::Checkbox(
                    AlienImGui::CheckboxParameters()
                        .name("Completeness check")
                        .textWidth(RightColumnWidth)
                        .defaultValue(origParameters.cellFunctionConstructorCheckCompletenessForSelfReplication)
                        .tooltip("If activated, a self-replication process can only start when all other non-self-replicating constructors in the cell network are "
                                 "finished."),
                    parameters.cellFunctionConstructorCheckCompletenessForSelfReplication);
                AlienImGui::EndTreeNode();
            }

            /**
             * Defender
             */
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Cell function: Defender"))) {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Anti-attacker strength")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(1.0f)
                        .max(5.0f)
                        .defaultValue(origParameters.cellFunctionDefenderAgainstAttackerStrength)
                        .tooltip("If an attacked cell is connected to defender cells or itself a defender cell the attack strength is reduced by this factor."),
                    parameters.cellFunctionDefenderAgainstAttackerStrength);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Anti-injector strength")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(1.0f)
                        .max(5.0f)
                        .defaultValue(origParameters.cellFunctionDefenderAgainstInjectorStrength)
                        .tooltip("If a constructor cell is attacked by an injector and connected to defender cells, the injection duration is increased by this "
                                 "factor."),
                    parameters.cellFunctionDefenderAgainstInjectorStrength);
                AlienImGui::EndTreeNode();
            }

            /**
             * Injector
             */
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Cell function: Injector"))) {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Injection radius")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(0.1f)
                        .max(4.0f)
                        .defaultValue(origParameters.cellFunctionInjectorRadius)
                        .tooltip("The maximum distance over which an injector cell can infect another cell."),
                    parameters.cellFunctionInjectorRadius);
                AlienImGui::InputIntColorMatrix(
                    AlienImGui::InputIntColorMatrixParameters()
                        .name("Injection time")
                        .logarithmic(true)
                        .max(100000)
                        .textWidth(RightColumnWidth)
                        .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origParameters.cellFunctionInjectorDurationColorMatrix))
                        .tooltip("The number of activations an injector cell requires to infect another cell. One activation usually takes 6 time steps. The row "
                                 "number determines the color of the injector cell, while the column number corresponds to the color of the infected cell."),
                    parameters.cellFunctionInjectorDurationColorMatrix);
                AlienImGui::EndTreeNode();
            }

            /**
             * Muscle
             */
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Cell function: Muscle"))) {
                AlienImGui::Checkbox(
                    AlienImGui::CheckboxParameters()
                        .name("Movement toward target")
                        .textWidth(RightColumnWidth)
                        .defaultValue(origParameters.cellFunctionMuscleMovementTowardTargetedObject)
                        .tooltip("If activated, a muscle cell in movement mode will only move if the triggering signal originates from a sensor cell that has "
                                 "targeted an object. The specified angle in the input is interpreted relative to the target."),
                    parameters.cellFunctionMuscleMovementTowardTargetedObject);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Movement acceleration")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(0)
                        .max(0.4f)
                        .logarithmic(true)
                        .defaultValue(origParameters.cellFunctionMuscleMovementAcceleration)
                        .tooltip("The maximum value by which a muscle cell can modify its velocity during activation. This parameter applies only to muscle cells "
                                 "which are in movement mode."),
                    parameters.cellFunctionMuscleMovementAcceleration);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Contraction and expansion delta")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(0)
                        .max(0.1f)
                        .defaultValue(origParameters.cellFunctionMuscleContractionExpansionDelta)
                        .tooltip("The maximum length that a muscle cell can shorten or lengthen a cell connection. This parameter applies only to muscle cells "
                                 "which are in contraction/expansion mode."),
                    parameters.cellFunctionMuscleContractionExpansionDelta);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Bending angle")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(0)
                        .max(10.0f)
                        .defaultValue(origParameters.cellFunctionMuscleBendingAngle)
                        .tooltip("The maximum value by which a muscle cell can increase/decrease the angle between two cell connections. This parameter applies "
                                 "only to muscle cells which are in bending mode."),
                    parameters.cellFunctionMuscleBendingAngle);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Bending acceleration")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(0)
                        .max(0.5f)
                        .defaultValue(origParameters.cellFunctionMuscleBendingAcceleration)
                        .tooltip("The maximum value by which a muscle cell can modify its velocity during a bending action. This parameter applies "
                                 "only to muscle cells which are in bending mode."),
                    parameters.cellFunctionMuscleBendingAcceleration);
                AlienImGui::EndTreeNode();
            }

            /**
             * Sensor
             */
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Cell function: Sensor"))) {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Radius")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(10.0f)
                        .max(800.0f)
                        .defaultValue(origParameters.cellFunctionSensorRange)
                        .tooltip("The maximum radius in which a sensor cell can detect mass concentrations."),
                    parameters.cellFunctionSensorRange);
                AlienImGui::EndTreeNode();
            }

            /**
             * Transmitter
             */
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Cell function: Transmitter"))) {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Energy distribution radius")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(0)
                        .max(5.0f)
                        .defaultValue(origParameters.cellFunctionTransmitterEnergyDistributionRadius)
                        .tooltip("The maximum distance over which a transmitter cell transfers its additional energy to nearby transmitter or constructor cells."),
                    parameters.cellFunctionTransmitterEnergyDistributionRadius);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Energy distribution Value")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(0)
                        .max(20.0f)
                        .defaultValue(origParameters.cellFunctionTransmitterEnergyDistributionValue)
                        .tooltip("The amount of energy which a transmitter cell can transfer to nearby transmitter or constructor cells or to connected cells."),
                    parameters.cellFunctionTransmitterEnergyDistributionValue);
                AlienImGui::Checkbox(
                    AlienImGui::CheckboxParameters()
                        .name("Same creature energy distribution")
                        .textWidth(RightColumnWidth)
                        .defaultValue(origParameters.cellFunctionTransmitterEnergyDistributionSameCreature)
                        .tooltip("If activated, the transmitter cells can only transfer energy to nearby cells belonging to the same creature."),
                    parameters.cellFunctionTransmitterEnergyDistributionSameCreature);
                AlienImGui::EndTreeNode();
            }

            /**
             * Reconnector
             */
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Cell function: Reconnector"))) {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Radius")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(0.0f)
                        .max(3.0f)
                        .defaultValue(origParameters.cellFunctionReconnectorRadius)
                        .tooltip("The maximum radius in which a reconnector cell can establish or destroy connections to other cells."),
                    parameters.cellFunctionReconnectorRadius);
                AlienImGui::EndTreeNode();
            }

            /**
             * Detonator
             */
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Cell function: Detonator"))) {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Blast radius")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(0.0f)
                        .max(10.0f)
                        .defaultValue(origParameters.cellFunctionDetonatorRadius)
                        .tooltip("The radius of the detonation."),
                    parameters.cellFunctionDetonatorRadius);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Chain explosion probability")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(0.0f)
                        .max(1.0f)
                        .defaultValue(origParameters.cellFunctionDetonatorChainExplosionProbability)
                        .tooltip("The probability that the explosion of one detonator will trigger the explosion of other detonators within the blast radius."),
                    parameters.cellFunctionDetonatorChainExplosionProbability);
                AlienImGui::EndTreeNode();
            }

            /**
             * Addon: Advanced absorption control
             */
            if (parameters.features.advancedAbsorptionControl) {
                if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Addon: Advanced energy absorption control"))) {
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Low genome complexity penalty")
                            .textWidth(RightColumnWidth)
                            .colorDependence(true)
                            .min(0)
                            .max(1.0f)
                            .format("%.2f")
                            .defaultValue(origParameters.baseValues.radiationAbsorptionLowGenomeComplexityPenalty)
                            .tooltip(Const::ParameterRadiationAbsorptionLowGenomeComplexityPenaltyTooltip),
                        parameters.baseValues.radiationAbsorptionLowGenomeComplexityPenalty);
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Low connection penalty")
                            .textWidth(RightColumnWidth)
                            .colorDependence(true)
                            .min(0)
                            .max(5.0f)
                            .format("%.1f")
                            .defaultValue(origParameters.radiationAbsorptionLowConnectionPenalty)
                            .tooltip(
                                "When this parameter is increased, cells with fewer cell connections will absorb less energy from an incoming energy particle."),
                        parameters.radiationAbsorptionLowConnectionPenalty);
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("High velocity penalty")
                            .textWidth(RightColumnWidth)
                            .colorDependence(true)
                            .min(0)
                            .max(30.0f)
                            .logarithmic(true)
                            .format("%.2f")
                            .defaultValue(origParameters.radiationAbsorptionHighVelocityPenalty)
                            .tooltip("When this parameter is increased, fast moving cells will absorb less energy from an incoming energy particle."),
                        parameters.radiationAbsorptionHighVelocityPenalty);
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Low velocity penalty")
                            .textWidth(RightColumnWidth)
                            .colorDependence(true)
                            .min(0)
                            .max(1.0f)
                            .format("%.2f")
                            .defaultValue(origParameters.baseValues.radiationAbsorptionLowVelocityPenalty)
                            .tooltip("When this parameter is increased, slowly moving cells will absorb less energy from an incoming energy particle."),
                        parameters.baseValues.radiationAbsorptionLowVelocityPenalty);
                    AlienImGui::EndTreeNode();
                }
            }

            /**
             * Addon: Advanced attacker control
             */
            if (parameters.features.advancedAttackerControl) {
                if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Addon: Advanced attacker control"))) {
                    AlienImGui::InputFloatColorMatrix(
                        AlienImGui::InputFloatColorMatrixParameters()
                            .name("Complex genome protection")
                            .textWidth(RightColumnWidth)
                            .min(0)
                            .max(20.0f)
                            .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origParameters.baseValues.cellFunctionAttackerGenomeComplexityBonus))
                            .tooltip("The larger this parameter is, the less energy can be gained by attacking creatures with more complex genomes."),
                        parameters.baseValues.cellFunctionAttackerGenomeComplexityBonus);
                    AlienImGui::InputFloatColorMatrix(
                        AlienImGui::InputFloatColorMatrixParameters()
                            .name("Same mutant protection")
                            .textWidth(RightColumnWidth)
                            .min(0)
                            .max(1.0f)
                            .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origParameters.cellFunctionAttackerSameMutantPenalty))
                            .tooltip("The larger this parameter is, the less energy can be gained by attacking creatures with the same mutation id."),
                        parameters.cellFunctionAttackerSameMutantPenalty);
                    AlienImGui::InputFloatColorMatrix(
                        AlienImGui::InputFloatColorMatrixParameters()
                            .name("New complex mutant protection")
                            .textWidth(RightColumnWidth)
                            .min(0)
                            .max(1.0f)
                            .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origParameters.baseValues.cellFunctionAttackerNewComplexMutantPenalty))
                            .tooltip("A high value protects new mutants with equal or greater genome complexity from being attacked."),
                        parameters.baseValues.cellFunctionAttackerNewComplexMutantPenalty);
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Sensor detection factor")
                            .textWidth(RightColumnWidth)
                            .colorDependence(true)
                            .min(0)
                            .max(1.0f)
                            .defaultValue(origParameters.cellFunctionAttackerSensorDetectionFactor)
                            .tooltip(
                                "This parameter controls whether the target must be previously detected with sensors in order to be attacked. The larger this "
                                "value is, the less energy can be gained during the attack if the target has not already been detected. For this purpose, the "
                                "attacker "
                                "cell searches for connected (or connected-connected) sensor cells to see which cell networks they have detected last time and "
                                "compares them with the attacked target."),
                        parameters.cellFunctionAttackerSensorDetectionFactor);
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Geometry penalty")
                            .textWidth(RightColumnWidth)
                            .colorDependence(true)
                            .min(0)
                            .max(5.0f)
                            .defaultValue(origParameters.baseValues.cellFunctionAttackerGeometryDeviationExponent)
                            .tooltip("The larger this value is, the less energy a cell can gain from an attack if the local "
                                     "geometry of the attacked cell does not match the attacking cell."),
                        parameters.baseValues.cellFunctionAttackerGeometryDeviationExponent);
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Connections mismatch penalty")
                            .textWidth(RightColumnWidth)
                            .colorDependence(true)
                            .min(0)
                            .max(1.0f)
                            .defaultValue(origParameters.baseValues.cellFunctionAttackerConnectionsMismatchPenalty)
                            .tooltip("The larger this parameter is, the more difficult it is to attack cells that contain more connections."),
                        parameters.baseValues.cellFunctionAttackerConnectionsMismatchPenalty);
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Color inhomogeneity factor")
                            .textWidth(RightColumnWidth)
                            .colorDependence(true)
                            .min(0)
                            .max(2.0f)
                            .defaultValue(origParameters.cellFunctionAttackerColorInhomogeneityFactor)
                            .tooltip("If the attacked cell is connected to cells with different colors, this factor affects the energy of the captured energy."),
                        parameters.cellFunctionAttackerColorInhomogeneityFactor);
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Energy distribution radius")
                            .textWidth(RightColumnWidth)
                            .colorDependence(true)
                            .min(0)
                            .max(5.0f)
                            .defaultValue(origParameters.cellFunctionAttackerEnergyDistributionRadius)
                            .tooltip("The maximum distance over which an attacker cell transfers its energy captured during an attack to nearby transmitter or "
                                     "constructor cells."),
                        parameters.cellFunctionAttackerEnergyDistributionRadius);
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Energy distribution Value")
                            .textWidth(RightColumnWidth)
                            .colorDependence(true)
                            .min(0)
                            .max(20.0f)
                            .defaultValue(origParameters.cellFunctionAttackerEnergyDistributionValue)
                            .tooltip("The amount of energy which a attacker cell can transfer to nearby transmitter or constructor cells or to connected cells."),
                        parameters.cellFunctionAttackerEnergyDistributionValue);
                    AlienImGui::EndTreeNode();
                }
            }

            /**
             * Addon: Cell color transition rules
             */
            if (parameters.features.cellColorTransitionRules) {
                if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Addon: Cell color transition rules").highlighted(false))) {
                    for (int color = 0; color < MAX_COLORS; ++color) {
                        ImGui::PushID(color);
                        auto widgetParameters = AlienImGui::InputColorTransitionParameters()
                                                    .textWidth(RightColumnWidth)
                                                    .color(color)
                                                    .defaultTargetColor(origParameters.baseValues.cellColorTransitionTargetColor[color])
                                                    .defaultTransitionAge(origParameters.baseValues.cellColorTransitionDuration[color])
                                                    .logarithmic(true)
                                                    .infinity(true);
                        if (0 == color) {
                            widgetParameters.name("Target color and duration")
                                .tooltip(
                                    "Rules can be defined that describe how the colors of cells will change over time. For this purpose, a subsequent color can "
                                    "be defined for each cell color. In addition, durations must be specified that define how many time steps the corresponding "
                                    "color are kept.");
                        }
                        AlienImGui::InputColorTransition(
                            widgetParameters,
                            color,
                            parameters.baseValues.cellColorTransitionTargetColor[color],
                            parameters.baseValues.cellColorTransitionDuration[color]);
                        ImGui::PopID();
                    }
                    AlienImGui::EndTreeNode();
                }
            }

            /**
             * Addon: Cell age limiter
             */
            if (parameters.features.cellAgeLimiter) {
                if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Addon: Cell age limiter").highlighted(false))) {
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Maximum inactive cell age")
                            .textWidth(RightColumnWidth)
                            .colorDependence(true)
                            .min(1.0f)
                            .max(10000000.0f)
                            .format("%.0f")
                            .logarithmic(true)
                            .infinity(true)
                            .disabledValue(parameters.baseValues.cellInactiveMaxAge)
                            .defaultEnabledValue(&origParameters.cellInactiveMaxAgeActivated)
                            .defaultValue(origParameters.baseValues.cellInactiveMaxAge)
                            .tooltip("Here, you can set the maximum age for a cell whose function or those of its neighbors have not been triggered. Cells which "
                                     "are in state 'Under construction' are not affected by this option."),
                        parameters.baseValues.cellInactiveMaxAge,
                        &parameters.cellInactiveMaxAgeActivated);
                    AlienImGui::SliderInt(
                        AlienImGui::SliderIntParameters()
                            .name("Maximum emergent cell age")
                            .textWidth(RightColumnWidth)
                            .colorDependence(true)
                            .min(1)
                            .max(10000000)
                            .logarithmic(true)
                            .infinity(true)
                            .disabledValue(parameters.cellEmergentMaxAge)
                            .defaultEnabledValue(&origParameters.cellEmergentMaxAgeActivated)
                            .defaultValue(origParameters.cellEmergentMaxAge)
                            .tooltip("The maximal age of cells that arise from energy particles can be set here."),
                        parameters.cellEmergentMaxAge,
                        &parameters.cellEmergentMaxAgeActivated);
                    AlienImGui::Checkbox(
                        AlienImGui::CheckboxParameters()
                            .name("Reset age after construction")
                            .textWidth(RightColumnWidth)
                            .defaultValue(origParameters.cellResetAgeAfterActivation)
                            .tooltip("If this option is activated, the age of the cells is reset to 0 after the construction of their cell network is completed, "
                                     "i.e. when the state of the cells changes from 'Under construction' to 'Ready'. This option is particularly useful if a low 'Maximum "
                                     "inactive cell age' is set, as cell networks that are under construction are inactive and could die immediately after "
                                     "completion if their construction takes a long time."),
                        parameters.cellResetAgeAfterActivation);
                    AlienImGui::SliderInt(
                        AlienImGui::SliderIntParameters()
                            .name("Maximum age balancing")
                            .textWidth(RightColumnWidth)
                            .logarithmic(true)
                            .min(1000)
                            .max(1000000)
                            .disabledValue(&parameters.cellMaxAgeBalancerInterval)
                            .defaultEnabledValue(&origParameters.cellMaxAgeBalancer)
                            .defaultValue(&origParameters.cellMaxAgeBalancerInterval)
                            .tooltip(
                                "Adjusts the maximum age at regular intervals. It increases the maximum age for the cell color where the fewest replicators exist. "
                                "Conversely, the maximum age is decreased for the cell color with the most replicators."),
                        &parameters.cellMaxAgeBalancerInterval,
                        &parameters.cellMaxAgeBalancer);
                    AlienImGui::EndTreeNode();
                }
            }

            /**
             * Addon: Cell glow
             */
            if (parameters.features.cellGlow) {
                if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Addon: Cell glow").highlighted(false))) {
                    AlienImGui::Switcher(
                        AlienImGui::SwitcherParameters()
                            .name("Coloring")
                            .textWidth(RightColumnWidth)
                            .defaultValue(origParameters.cellGlowColoring)
                            .values(
                                {"Energy",
                                 "Standard cell colors",
                                 "Mutants",
                                 "Mutants and cell functions",
                                 "Cell states",
                                 "Genome complexities",
                                 "Single cell function",
                                 "All cell functions"})
                            .tooltip(Const::ColoringParameterTooltip),
                        parameters.cellGlowColoring);
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Radius")
                            .textWidth(RightColumnWidth)
                            .min(1.0f)
                            .max(8.0f)
                            .defaultValue(&origParameters.cellGlowRadius)
                            .tooltip("The radius of the glow. Please note that a large radius affects the performance."),
                        &parameters.cellGlowRadius);
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Strength")
                            .textWidth(RightColumnWidth)
                            .min(0)
                            .max(1.0f)
                            .defaultValue(&origParameters.cellGlowStrength)
                            .tooltip("The strength of the glow."),
                        &parameters.cellGlowStrength);
                    AlienImGui::EndTreeNode();
                }
            }

            /**
             * Addon: External energy control
             */
            if (parameters.features.externalEnergyControl) {
                if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Addon: External energy control").highlighted(false))) {
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("External energy amount")
                            .textWidth(RightColumnWidth)
                            .min(0.0f)
                            .max(100000000.0f)
                            .format("%.0f")
                            .logarithmic(true)
                            .infinity(true)
                            .defaultValue(&origParameters.externalEnergy)
                            .tooltip("This parameter can be used to set the amount of energy of an external energy source. This type of energy can be "
                                     "transferred to all constructor cells at a certain rate (see inflow settings).\n\nTip: You can explicitly enter a "
                                     "numerical value by clicking on the "
                                     "slider while holding CTRL.\n\nWarning: Too much external energy can result in a massive production of cells and slow "
                                     "down or "
                                     "even crash the simulation."),
                        &parameters.externalEnergy);
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Inflow")
                            .textWidth(RightColumnWidth)
                            .colorDependence(true)
                            .min(0.0f)
                            .max(1.0f)
                            .format("%.5f")
                            .logarithmic(true)
                            .defaultValue(origParameters.externalEnergyInflowFactor)
                            .tooltip("Here one can specify the fraction of energy transferred to constructor cells.\n\nFor example, a value of 0.05 means that "
                                     "each time "
                                     "a constructor cell tries to build a new cell, 5% of the required energy is transferred for free from the external energy "
                                     "source."),
                        parameters.externalEnergyInflowFactor);
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Conditional inflow")
                            .textWidth(RightColumnWidth)
                            .colorDependence(true)
                            .min(0.00f)
                            .max(1.0f)
                            .format("%.5f")
                            .defaultValue(origParameters.externalEnergyConditionalInflowFactor)
                            .tooltip(
                                "Here one can specify the fraction of energy transferred to constructor cells if they can provide the remaining energy for the "
                                "construction process.\n\nFor example, a value of 0.6 means that a constructor cell receives 60% of the energy required to "
                                "build the new cell for free from the external energy source. However, it must provide 40% of the energy required by itself. "
                                "Otherwise, no energy will be transferred."),
                        parameters.externalEnergyConditionalInflowFactor);
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Backflow")
                            .textWidth(RightColumnWidth)
                            .colorDependence(true)
                            .min(0.0f)
                            .max(1.0f)
                            .defaultValue(origParameters.externalEnergyBackflowFactor)
                            .tooltip("The proportion of energy that flows back to the external energy source when a cell loses energy or dies. The remaining "
                                     "fraction of the energy is used to create a new energy particle."),
                        parameters.externalEnergyBackflowFactor);
                    AlienImGui::EndTreeNode();
                }
            }

            /**
             * Addon: Genome complexity measurement
             */
            if (parameters.features.genomeComplexityMeasurement) {
                if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Addon: Genome complexity measurement").highlighted(false))) {
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Size factor")
                            .textWidth(RightColumnWidth)
                            .colorDependence(true)
                            .min(0.0f)
                            .max(1.0f)
                            .format("%.2f")
                            .defaultValue(origParameters.genomeComplexitySizeFactor)
                            .tooltip("This parameter controls how the number of encoded cells in the genome influences the calculation of its complexity."),
                        parameters.genomeComplexitySizeFactor);
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Ramification factor")
                            .textWidth(RightColumnWidth)
                            .colorDependence(true)
                            .min(0.0f)
                            .max(20.0f)
                            .format("%.2f")
                            .defaultValue(origParameters.genomeComplexityRamificationFactor)
                            .tooltip("With this parameter, the number of ramifications of the cell structure to the genome is taken into account for the "
                                     "calculation of the genome complexity. For instance, genomes that contain many sub-genomes or many construction branches will "
                                     "then have a high complexity value."),
                        parameters.genomeComplexityRamificationFactor);
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Neuron factor")
                            .textWidth(RightColumnWidth)
                            .colorDependence(true)
                            .min(0.0f)
                            .max(4.0f)
                            .format("%.2f")
                            .defaultValue(origParameters.genomeComplexityNeuronFactor)
                            .tooltip("This parameter takes into account the number of encoded neurons in the genome for the complexity value."),
                        parameters.genomeComplexityNeuronFactor);
                    AlienImGui::EndTreeNode();
                }
            }

            /**
             * Addon: Legacy behavior
             */
            if (parameters.features.legacyModes) {
                if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Addon: Legacy behavior"))) {
                    AlienImGui::Checkbox(
                        AlienImGui::CheckboxParameters()
                            .name("Fetch angle from adjacent sensor")
                            .textWidth(RightColumnWidth)
                            .defaultValue(origParameters.legacyCellFunctionMuscleMovementAngleFromSensor)
                            .tooltip("This parameter changes the behavior of the parameter 'Movement toward target'. If activated, the muscle cell fetches the "
                                     "movement angle directly from a connected (or connected-connected) sensor cell that has previously detected a target "
                                     "(legacy behavior). If deactivated, the input signal must only originate from a sensor cell and must not be adjacent (new behavior)."),
                        parameters.legacyCellFunctionMuscleMovementAngleFromSensor);
                    AlienImGui::Checkbox(
                        AlienImGui::CheckboxParameters()
                            .name("No activity reset in muscles")
                            .textWidth(RightColumnWidth)
                            .defaultValue(origParameters.legacyCellFunctionMuscleNoActivityReset)
                            .tooltip("If activated, the activity in channel #0 is not set to 0 in muscle cells which are in movement mode. Thus the output of this type "
                                     "of muscles can be reused for other muscle cells."),
                        parameters.legacyCellFunctionMuscleNoActivityReset);
                    AlienImGui::Checkbox(
                        AlienImGui::CheckboxParameters()
                            .name("Allow bidirectional connections")
                            .textWidth(RightColumnWidth)
                            .defaultValue(origParameters.legacyCellDirectionalConnections)
                            .tooltip("If activated, two connected cells can receive each other's input if the 'input execution number' matches."),
                        parameters.legacyCellDirectionalConnections);
                    AlienImGui::EndTreeNode();
                }
            }
        }
        ImGui::EndChild();

        validationAndCorrection(parameters);
        validationAndCorrectionLayout();

        if (parameters != lastParameters) {
            _simulationFacade->setSimulationParameters(parameters);
        }

        ImGui::EndTabItem();
    }
}

bool SimulationParametersWindow::processSpot(int index)
{
    std::string name = "Zone " + std::to_string(index + 1);
    bool isOpen = true;
    if (ImGui::BeginTabItem(name.c_str(), &isOpen, ImGuiTabItemFlags_None)) {
        auto parameters = _simulationFacade->getSimulationParameters();
        auto origParameters = _simulationFacade->getOriginalSimulationParameters();
        auto lastParameters = parameters;

        SimulationParametersSpot& spot = parameters.spots[index];
        SimulationParametersSpot const& origSpot = origParameters.spots[index];
        SimulationParametersSpot const& lastSpot = lastParameters.spots[index];

        if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
            auto worldSize = _simulationFacade->getWorldSize();

            /**
             * Colors and location
             */
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Visualization"))) {
                AlienImGui::ColorButtonWithPicker(
                    AlienImGui::ColorButtonWithPickerParameters().name("Background color").textWidth(RightColumnWidth).defaultValue(origSpot.color),
                    spot.color,
                    _backupColor,
                    _savedPalette);
                AlienImGui::EndTreeNode();
            }

            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Location"))) {
                if (AlienImGui::Switcher(
                        AlienImGui::SwitcherParameters()
                            .name("Shape")
                            .values({"Circular", "Rectangular"})
                            .textWidth(RightColumnWidth)
                            .defaultValue(origSpot.shapeType),
                        spot.shapeType)) {
                    createDefaultSpotData(spot);
                }

                auto getMousePickerEnabledFunc = [&]() { return SimulationInteractionController::get().isPositionSelectionMode(); };
                auto setMousePickerEnabledFunc = [&](bool value) { SimulationInteractionController::get().setPositionSelectionMode(value); };
                auto getMousePickerPositionFunc = [&]() { return SimulationInteractionController::get().getPositionSelectionData(); };

                AlienImGui::SliderFloat2(
                    AlienImGui::SliderFloat2Parameters()
                        .name("Position (x,y)")
                        .textWidth(RightColumnWidth)
                        .min({0, 0})
                        .max(toRealVector2D(worldSize))
                        .defaultValue(RealVector2D{origSpot.posX, origSpot.posY})
                        .format("%.2f")
                        .getMousePickerEnabledFunc(getMousePickerEnabledFunc)
                        .setMousePickerEnabledFunc(setMousePickerEnabledFunc)
                        .getMousePickerPositionFunc(getMousePickerPositionFunc),
                    spot.posX,
                    spot.posY);
                AlienImGui::SliderFloat2(
                    AlienImGui::SliderFloat2Parameters()
                        .name("Velocity (x,y)")
                        .textWidth(RightColumnWidth)
                        .min({-4.0f, -4.0f})
                        .max({4.0f, 4.0f})
                        .defaultValue(RealVector2D{origSpot.velX, origSpot.velY})
                        .format("%.2f"),
                    spot.velX,
                    spot.velY);
                auto maxRadius = toFloat(std::max(worldSize.x, worldSize.y));
                if (spot.shapeType == SpotShapeType_Circular) {
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Core radius")
                            .textWidth(RightColumnWidth)
                            .min(0)
                            .max(maxRadius)
                            .defaultValue(&origSpot.shapeData.circularSpot.coreRadius)
                            .format("%.1f"),
                        &spot.shapeData.circularSpot.coreRadius);
                }
                if (spot.shapeType == SpotShapeType_Rectangular) {
                    AlienImGui::SliderFloat2(
                        AlienImGui::SliderFloat2Parameters()
                            .name("Size (x,y)")
                            .textWidth(RightColumnWidth)
                            .min({0, 0})
                            .max({toFloat(worldSize.x), toFloat(worldSize.y)})
                            .defaultValue(RealVector2D{origSpot.shapeData.rectangularSpot.width, origSpot.shapeData.rectangularSpot.height})
                            .format("%.1f"),
                        spot.shapeData.rectangularSpot.width,
                        spot.shapeData.rectangularSpot.height);
                }

                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Fade-out radius")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(maxRadius)
                        .defaultValue(&origSpot.fadeoutRadius)
                        .format("%.1f"),
                    &spot.fadeoutRadius);
                AlienImGui::EndTreeNode();
            }

            /**
             * Flow
             */
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Force field"))) {
                auto isForceFieldActive = spot.flowType != FlowType_None;

                auto forceFieldTypeIntern = std::max(0, spot.flowType - 1);  //FlowType_None should not be selectable in ComboBox
                auto origForceFieldTypeIntern = std::max(0, origSpot.flowType - 1);
                if (ImGui::Checkbox("##forceField", &isForceFieldActive)) {
                    spot.flowType = isForceFieldActive ? FlowType_Radial : FlowType_None;
                }
                ImGui::SameLine();
                ImGui::BeginDisabled(!isForceFieldActive);
                auto posX = ImGui::GetCursorPos().x;
                if (AlienImGui::Combo(
                        AlienImGui::ComboParameters()
                            .name("Type")
                            .values({"Radial", "Central", "Linear"})
                            .textWidth(RightColumnWidth)
                            .defaultValue(origForceFieldTypeIntern),
                        forceFieldTypeIntern)) {
                    spot.flowType = forceFieldTypeIntern + 1;
                    if (spot.flowType == FlowType_Radial) {
                        spot.flowData.radialFlow = RadialFlow();
                    }
                    if (spot.flowType == FlowType_Central) {
                        spot.flowData.centralFlow = CentralFlow();
                    }
                    if (spot.flowType == FlowType_Linear) {
                        spot.flowData.linearFlow = LinearFlow();
                    }
                }
                if (spot.flowType == FlowType_Radial) {
                    ImGui::SetCursorPosX(posX);
                    AlienImGui::Combo(
                        AlienImGui::ComboParameters()
                            .name("Orientation")
                            .textWidth(RightColumnWidth)
                            .defaultValue(origSpot.flowData.radialFlow.orientation)
                            .values({"Clockwise", "Counter clockwise"}),
                        spot.flowData.radialFlow.orientation);
                    ImGui::SetCursorPosX(posX);
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Strength")
                            .textWidth(RightColumnWidth)
                            .min(0)
                            .max(0.5f)
                            .logarithmic(true)
                            .format("%.5f")
                            .defaultValue(&origSpot.flowData.radialFlow.strength),
                        &spot.flowData.radialFlow.strength);
                    ImGui::SetCursorPosX(posX);
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Drift angle")
                            .textWidth(RightColumnWidth)
                            .min(-180.0f)
                            .max(180.0f)
                            .format("%.1f")
                            .defaultValue(&origSpot.flowData.radialFlow.driftAngle),
                        &spot.flowData.radialFlow.driftAngle);
                }
                if (spot.flowType == FlowType_Central) {
                    ImGui::SetCursorPosX(posX);
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Strength")
                            .textWidth(RightColumnWidth)
                            .min(0)
                            .max(0.5f)
                            .logarithmic(true)
                            .format("%.5f")
                            .defaultValue(&origSpot.flowData.centralFlow.strength),
                        &spot.flowData.centralFlow.strength);
                }
                if (spot.flowType == FlowType_Linear) {
                    ImGui::SetCursorPosX(posX);
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Angle")
                            .textWidth(RightColumnWidth)
                            .min(-180.0f)
                            .max(180.0f)
                            .format("%.1f")
                            .defaultValue(&origSpot.flowData.linearFlow.angle),
                        &spot.flowData.linearFlow.angle);
                    ImGui::SetCursorPosX(posX);
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Strength")
                            .textWidth(RightColumnWidth)
                            .min(0)
                            .max(0.5f)
                            .logarithmic(true)
                            .format("%.5f")
                            .defaultValue(&origSpot.flowData.linearFlow.strength),
                        &spot.flowData.linearFlow.strength);
                }
                ImGui::EndDisabled();
                AlienImGui::EndTreeNode();
            }

            /**
             * Physics: Motion
             */
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Physics: Motion"))) {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Friction")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(1)
                        .logarithmic(true)
                        .defaultValue(&origSpot.values.friction)
                        .disabledValue(&parameters.baseValues.friction)
                        .format("%.4f"),
                    &spot.values.friction,
                    &spot.activatedValues.friction);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Rigidity")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(1)
                        .defaultValue(&origSpot.values.rigidity)
                        .disabledValue(&parameters.baseValues.rigidity)
                        .format("%.2f"),
                    &spot.values.rigidity,
                    &spot.activatedValues.rigidity);
                AlienImGui::EndTreeNode();
            }

            /**
             * Physics: Thresholds
             */
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Physics: Thresholds"))) {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Maximum force")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(3.0f)
                        .colorDependence(true)
                        .defaultValue(origSpot.values.cellMaxForce)
                        .disabledValue(parameters.baseValues.cellMaxForce),
                    spot.values.cellMaxForce,
                    &spot.activatedValues.cellMaxForce);
                AlienImGui::EndTreeNode();
            }

            /**
             * Physics: Binding
             */
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Physics: Binding"))) {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Binding creation velocity")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(2.0f)
                        .defaultValue(&origSpot.values.cellFusionVelocity)
                        .disabledValue(&parameters.baseValues.cellFusionVelocity),
                    &spot.values.cellFusionVelocity,
                    &spot.activatedValues.cellFusionVelocity);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Maximum energy")
                        .textWidth(RightColumnWidth)
                        .min(50.0f)
                        .max(10000000.0f)
                        .logarithmic(true)
                        .infinity(true)
                        .format("%.0f")
                        .defaultValue(&origSpot.values.cellMaxBindingEnergy)
                        .disabledValue(&parameters.baseValues.cellMaxBindingEnergy),
                    &spot.values.cellMaxBindingEnergy,
                    &spot.activatedValues.cellMaxBindingEnergy);
                AlienImGui::EndTreeNode();
            }

            /**
             * Physics: Radiation
             */
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Physics: Radiation"))) {
                AlienImGui::Checkbox(
                    AlienImGui::CheckboxParameters()
                        .name("Disable radiation sources")
                        .textWidth(RightColumnWidth)
                        .defaultValue(origSpot.values.radiationDisableSources)
                        .tooltip("If activated, all radiation sources within this spot are deactivated."),
                    spot.values.radiationDisableSources);
                spot.activatedValues.radiationDisableSources = spot.values.radiationDisableSources;

                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Absorption factor")
                        .textWidth(RightColumnWidth)
                        .logarithmic(true)
                        .colorDependence(true)
                        .min(0)
                        .max(1.0)
                        .format("%.4f")
                        .defaultValue(origSpot.values.radiationAbsorption)
                        .disabledValue(parameters.baseValues.radiationAbsorption),
                    spot.values.radiationAbsorption,
                    &spot.activatedValues.radiationAbsorption);

                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Radiation type 1: Strength")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(0)
                        .max(0.01f)
                        .logarithmic(true)
                        .defaultValue(origSpot.values.radiationCellAgeStrength)
                        .disabledValue(parameters.baseValues.radiationCellAgeStrength)
                        .format("%.6f"),
                    spot.values.radiationCellAgeStrength,
                    &spot.activatedValues.radiationCellAgeStrength);
                AlienImGui::EndTreeNode();
            }

            /**
             * Cell life cycle
             */
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Cell life cycle"))) {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Minimum energy")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(10.0f)
                        .max(200.0f)
                        .defaultValue(origSpot.values.cellMinEnergy)
                        .disabledValue(parameters.baseValues.cellMinEnergy),
                    spot.values.cellMinEnergy,
                    &spot.activatedValues.cellMinEnergy);
                AlienImGui::EndTreeNode();
            }

            /**
             * Mutation 
             */
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Genome copy mutations"))) {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Neuron weights and biases")
                        .textWidth(RightColumnWidth)
                        .min(0.0f)
                        .max(1.0f)
                        .format("%.7f")
                        .colorDependence(true)
                        .logarithmic(true)
                        .defaultValue(origSpot.values.cellCopyMutationNeuronData)
                        .disabledValue(parameters.baseValues.cellCopyMutationNeuronData),
                    spot.values.cellCopyMutationNeuronData,
                    &spot.activatedValues.cellCopyMutationNeuronData);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Cell properties")
                        .textWidth(RightColumnWidth)
                        .min(0.0f)
                        .max(1.0f)
                        .format("%.7f")
                        .logarithmic(true)
                        .colorDependence(true)
                        .defaultValue(origSpot.values.cellCopyMutationCellProperties)
                        .disabledValue(parameters.baseValues.cellCopyMutationCellProperties),
                    spot.values.cellCopyMutationCellProperties,
                    &spot.activatedValues.cellCopyMutationCellProperties);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Geometry")
                        .textWidth(RightColumnWidth)
                        .min(0.0f)
                        .max(1.0f)
                        .format("%.7f")
                        .logarithmic(true)
                        .colorDependence(true)
                        .defaultValue(origSpot.values.cellCopyMutationGeometry)
                        .disabledValue(parameters.baseValues.cellCopyMutationGeometry),
                    spot.values.cellCopyMutationGeometry,
                    &spot.activatedValues.cellCopyMutationGeometry);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Custom geometry")
                        .textWidth(RightColumnWidth)
                        .min(0.0f)
                        .max(1.0f)
                        .format("%.7f")
                        .logarithmic(true)
                        .colorDependence(true)
                        .defaultValue(origSpot.values.cellCopyMutationCustomGeometry)
                        .disabledValue(parameters.baseValues.cellCopyMutationCustomGeometry),
                    spot.values.cellCopyMutationCustomGeometry,
                    &spot.activatedValues.cellCopyMutationCustomGeometry);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Cell function type")
                        .textWidth(RightColumnWidth)
                        .min(0.0f)
                        .max(1.0f)
                        .format("%.7f")
                        .logarithmic(true)
                        .colorDependence(true)
                        .defaultValue(origSpot.values.cellCopyMutationCellFunction)
                        .disabledValue(parameters.baseValues.cellCopyMutationCellFunction),
                    spot.values.cellCopyMutationCellFunction,
                    &spot.activatedValues.cellCopyMutationCellFunction);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Cell insertion")
                        .textWidth(RightColumnWidth)
                        .min(0.0f)
                        .max(1.0f)
                        .format("%.7f")
                        .logarithmic(true)
                        .colorDependence(true)
                        .defaultValue(origSpot.values.cellCopyMutationInsertion)
                        .disabledValue(parameters.baseValues.cellCopyMutationInsertion),
                    spot.values.cellCopyMutationInsertion,
                    &spot.activatedValues.cellCopyMutationInsertion);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Cell deletion")
                        .textWidth(RightColumnWidth)
                        .min(0.0f)
                        .max(1.0f)
                        .format("%.7f")
                        .logarithmic(true)
                        .colorDependence(true)
                        .defaultValue(origSpot.values.cellCopyMutationDeletion)
                        .disabledValue(parameters.baseValues.cellCopyMutationDeletion),
                    spot.values.cellCopyMutationDeletion,
                    &spot.activatedValues.cellCopyMutationDeletion);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Translation")
                        .textWidth(RightColumnWidth)
                        .min(0.0f)
                        .max(1.0f)
                        .format("%.7f")
                        .logarithmic(true)
                        .colorDependence(true)
                        .defaultValue(origSpot.values.cellCopyMutationTranslation)
                        .disabledValue(parameters.baseValues.cellCopyMutationTranslation),
                    spot.values.cellCopyMutationTranslation,
                    &spot.activatedValues.cellCopyMutationTranslation);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Duplication")
                        .textWidth(RightColumnWidth)
                        .min(0.0f)
                        .max(1.0f)
                        .format("%.7f")
                        .logarithmic(true)
                        .colorDependence(true)
                        .defaultValue(origSpot.values.cellCopyMutationDuplication)
                        .disabledValue(parameters.baseValues.cellCopyMutationDuplication),
                    spot.values.cellCopyMutationDuplication,
                    &spot.activatedValues.cellCopyMutationDuplication);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Individual cell color")
                        .textWidth(RightColumnWidth)
                        .min(0.0f)
                        .max(1.0f)
                        .format("%.7f")
                        .logarithmic(true)
                        .colorDependence(true)
                        .defaultValue(origSpot.values.cellCopyMutationCellColor)
                        .disabledValue(parameters.baseValues.cellCopyMutationCellColor),
                    spot.values.cellCopyMutationCellColor,
                    &spot.activatedValues.cellCopyMutationCellColor);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Sub-genome color")
                        .textWidth(RightColumnWidth)
                        .min(0.0f)
                        .max(1.0f)
                        .format("%.7f")
                        .logarithmic(true)
                        .colorDependence(true)
                        .defaultValue(origSpot.values.cellCopyMutationSubgenomeColor)
                        .disabledValue(parameters.baseValues.cellCopyMutationSubgenomeColor),
                    spot.values.cellCopyMutationSubgenomeColor,
                    &spot.activatedValues.cellCopyMutationSubgenomeColor);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Genome color")
                        .textWidth(RightColumnWidth)
                        .min(0.0f)
                        .max(1.0f)
                        .format("%.7f")
                        .logarithmic(true)
                        .colorDependence(true)
                        .defaultValue(origSpot.values.cellCopyMutationGenomeColor)
                        .disabledValue(parameters.baseValues.cellCopyMutationGenomeColor),
                    spot.values.cellCopyMutationGenomeColor,
                    &spot.activatedValues.cellCopyMutationGenomeColor);
                AlienImGui::EndTreeNode();
            }

            /**
             * Attacker
             */
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Cell function: Attacker"))) {
                AlienImGui::InputFloatColorMatrix(
                    AlienImGui::InputFloatColorMatrixParameters()
                        .name("Food chain color matrix")
                        .max(1)
                        .textWidth(RightColumnWidth)
                        .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origSpot.values.cellFunctionAttackerFoodChainColorMatrix))
                        .disabledValue(toVector<MAX_COLORS, MAX_COLORS>(parameters.baseValues.cellFunctionAttackerFoodChainColorMatrix)),
                    spot.values.cellFunctionAttackerFoodChainColorMatrix,
                    &spot.activatedValues.cellFunctionAttackerFoodChainColorMatrix);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Energy cost")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(0)
                        .max(1.0f)
                        .format("%.5f")
                        .logarithmic(true)
                        .defaultValue(origSpot.values.cellFunctionAttackerEnergyCost)
                        .disabledValue(parameters.baseValues.cellFunctionAttackerEnergyCost),
                    spot.values.cellFunctionAttackerEnergyCost,
                    &spot.activatedValues.cellFunctionAttackerEnergyCost);
                AlienImGui::EndTreeNode();
            }

            /**
             * Addon: Advanced absorption control
             */
            if (parameters.features.advancedAbsorptionControl) {
                if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Addon: Advanced energy absorption control"))) {
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Low velocity penalty")
                            .textWidth(RightColumnWidth)
                            .colorDependence(true)
                            .min(0)
                            .max(1.0f)
                            .format("%.2f")
                            .defaultValue(origSpot.values.radiationAbsorptionLowVelocityPenalty)
                            .disabledValue(parameters.baseValues.radiationAbsorptionLowVelocityPenalty),
                        spot.values.radiationAbsorptionLowVelocityPenalty,
                        &spot.activatedValues.radiationAbsorptionLowVelocityPenalty);
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Low genome complexity penalty")
                            .textWidth(RightColumnWidth)
                            .colorDependence(true)
                            .min(0)
                            .max(1.0f)
                            .format("%.2f")
                            .defaultValue(origSpot.values.radiationAbsorptionLowGenomeComplexityPenalty)
                            .disabledValue(parameters.baseValues.radiationAbsorptionLowGenomeComplexityPenalty),
                        spot.values.radiationAbsorptionLowGenomeComplexityPenalty,
                        &spot.activatedValues.radiationAbsorptionLowGenomeComplexityPenalty);
                    AlienImGui::EndTreeNode();
                }
            }

            /**
             * Addon: Advanced attacker control
             */
            if (parameters.features.advancedAttackerControl) {
                if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Addon: Advanced attacker control"))) {
                    AlienImGui::InputFloatColorMatrix(
                        AlienImGui::InputFloatColorMatrixParameters()
                            .name("Complex genome protection")
                            .textWidth(RightColumnWidth)
                            .min(0)
                            .max(20.0f)
                            .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origSpot.values.cellFunctionAttackerGenomeComplexityBonus))
                            .disabledValue(toVector<MAX_COLORS, MAX_COLORS>(parameters.baseValues.cellFunctionAttackerGenomeComplexityBonus)),
                        spot.values.cellFunctionAttackerGenomeComplexityBonus,
                        &spot.activatedValues.cellFunctionAttackerGenomeComplexityBonus);
                    AlienImGui::InputFloatColorMatrix(
                        AlienImGui::InputFloatColorMatrixParameters()
                            .name("New complex mutant protection")
                            .textWidth(RightColumnWidth)
                            .min(0)
                            .max(1.0f)
                            .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origSpot.values.cellFunctionAttackerNewComplexMutantPenalty))
                            .disabledValue(toVector<MAX_COLORS, MAX_COLORS>(parameters.baseValues.cellFunctionAttackerNewComplexMutantPenalty)),
                        spot.values.cellFunctionAttackerNewComplexMutantPenalty,
                        &spot.activatedValues.cellFunctionAttackerNewComplexMutantPenalty);

                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Geometry penalty")
                            .textWidth(RightColumnWidth)
                            .colorDependence(true)
                            .min(0)
                            .max(5.0f)
                            .defaultValue(origSpot.values.cellFunctionAttackerGeometryDeviationExponent)
                            .disabledValue(parameters.baseValues.cellFunctionAttackerGeometryDeviationExponent),
                        spot.values.cellFunctionAttackerGeometryDeviationExponent,
                        &spot.activatedValues.cellFunctionAttackerGeometryDeviationExponent);
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Connections mismatch penalty")
                            .textWidth(RightColumnWidth)
                            .colorDependence(true)
                            .min(0)
                            .max(1.0f)
                            .defaultValue(origSpot.values.cellFunctionAttackerConnectionsMismatchPenalty)
                            .disabledValue(parameters.baseValues.cellFunctionAttackerConnectionsMismatchPenalty),
                        spot.values.cellFunctionAttackerConnectionsMismatchPenalty,
                        &spot.activatedValues.cellFunctionAttackerConnectionsMismatchPenalty);
                    AlienImGui::EndTreeNode();
                }
            }

            /**
             * Addon: Cell age limiter
             */
            if (parameters.features.cellAgeLimiter) {
                if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Addon: Cell age limiter").highlighted(false))) {
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name("Maximum inactive cell age")
                            .textWidth(RightColumnWidth)
                            .colorDependence(true)
                            .min(1.0f)
                            .max(10000000.0f)
                            .logarithmic(true)
                            .infinity(true)
                            .format("%.0f")
                            .disabledValue(parameters.baseValues.cellInactiveMaxAge)
                            .defaultValue(origSpot.values.cellInactiveMaxAge),
                        spot.values.cellInactiveMaxAge,
                        &spot.activatedValues.cellInactiveMaxAge);
                    AlienImGui::EndTreeNode();
                }
            }

            /**
             * Addon: Cell color transition rules
             */
            if (parameters.features.cellColorTransitionRules) {
                if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Addon: Cell color transition rules").highlighted(false))) {
                    ImGui::Checkbox("##cellColorTransition", &spot.activatedValues.cellColorTransition);
                    ImGui::SameLine();
                    ImGui::BeginDisabled(!spot.activatedValues.cellColorTransition);
                    auto posX = ImGui::GetCursorPos().x;
                    for (int color = 0; color < MAX_COLORS; ++color) {
                        ImGui::SetCursorPosX(posX);
                        ImGui::PushID(color);
                        auto parameters = AlienImGui::InputColorTransitionParameters()
                                              .textWidth(RightColumnWidth)
                                              .color(color)
                                              .defaultTargetColor(origSpot.values.cellColorTransitionTargetColor[color])
                                              .defaultTransitionAge(origSpot.values.cellColorTransitionDuration[color])
                                              .logarithmic(true)
                                              .infinity(true);
                        if (0 == color) {
                            parameters.name("Target color and duration");
                        }
                        AlienImGui::InputColorTransition(
                            parameters, color, spot.values.cellColorTransitionTargetColor[color], spot.values.cellColorTransitionDuration[color]);
                        ImGui::PopID();
                    }
                    ImGui::EndDisabled();
                    AlienImGui::EndTreeNode();
                    if (!spot.activatedValues.cellColorTransition) {
                        for (int color = 0; color < MAX_COLORS; ++color) {
                            spot.values.cellColorTransitionTargetColor[color] = parameters.baseValues.cellColorTransitionTargetColor[color];
                            spot.values.cellColorTransitionDuration[color] = parameters.baseValues.cellColorTransitionDuration[color];
                        }
                    }
                }
            }
        }
        ImGui::EndChild();
        validationAndCorrection(spot, parameters);

        if (spot != lastSpot) {
            _simulationFacade->setSimulationParameters(parameters);
        }

        ImGui::EndTabItem();
    }

    return isOpen;
}

void SimulationParametersWindow::processAddonList()
{
    if (_featureListOpen) {
        ImGui::Spacing();
        ImGui::Spacing();
        AlienImGui::MovableSeparator(_featureListHeight);
    } else {
        AlienImGui::Separator();
    }

    _featureListOpen = AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Addons").highlighted(true).defaultOpen(_featureListOpen));
    if (_featureListOpen) {
        if (ImGui::BeginChild("##addons", {scale(0), 0})) {
            auto parameters = _simulationFacade->getSimulationParameters();
            auto origFeatures = _simulationFacade->getOriginalSimulationParameters().features;
            auto lastFeatures= parameters.features;

            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Advanced absorption control")
                    .textWidth(0)
                    .defaultValue(origFeatures.advancedAbsorptionControl)
                    .tooltip("This addon offers extended possibilities for controlling the absorption of energy particles by cells."),
                parameters.features.advancedAbsorptionControl);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Advanced attacker control")
                    .textWidth(0)
                    .defaultValue(origFeatures.advancedAttackerControl)
                    .tooltip("It contains further settings that influence how much energy can be obtained from an attack by attacker cells."),
                parameters.features.advancedAttackerControl);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Cell age limiter")
                    .textWidth(0)
                    .defaultValue(origFeatures.cellAgeLimiter)
                    .tooltip("It enables additional possibilities to control the maximal cell age."),
                parameters.features.cellAgeLimiter);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Cell color transition rules")
                    .textWidth(0)
                    .defaultValue(origFeatures.cellColorTransitionRules)
                    .tooltip("This can be used to define color transitions for cells depending on their age."),
                parameters.features.cellColorTransitionRules);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Cell glow")
                    .textWidth(0)
                    .defaultValue(origFeatures.cellGlow)
                    .tooltip("It enables an additional rendering step that makes the cells glow."),
                parameters.features.cellGlow);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("External energy control")
                    .textWidth(0)
                    .defaultValue(origFeatures.externalEnergyControl)
                    .tooltip("This addon is used to add an external energy source. Its energy can be gradually transferred to the constructor cells in the "
                             "simulation. Vice versa, the energy from radiation and dying cells can also be transferred back to the external source."),
                parameters.features.externalEnergyControl);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Genome complexity measurement")
                    .textWidth(0)
                    .defaultValue(origFeatures.genomeComplexityMeasurement)
                    .tooltip("Parameters for the calculation of genome complexity are activated here. This genome complexity can be used for 'Advanced "
                             "absorption control' "
                             "and 'Advanced attacker control' to favor more complex genomes in natural selection. If it is deactivated, default values are "
                             "used that simply take the genome size into account."),
                parameters.features.genomeComplexityMeasurement);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Legacy behavior")
                    .textWidth(0)
                    .defaultValue(origFeatures.legacyModes)
                    .tooltip("It contains features for compatibility with older versions."),
                parameters.features.legacyModes);

            if (parameters.features != lastFeatures) {
                _simulationFacade->setSimulationParameters(parameters);
            }
        }
        ImGui::EndChild();
        AlienImGui::EndTreeNode();
    }
}

void SimulationParametersWindow::onAppendTab()
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();

    int index = parameters.numSpots;
    parameters.spots[index] = createSpot(parameters, index);
    origParameters.spots[index] = createSpot(parameters, index);
    ++parameters.numSpots;
    ++origParameters.numSpots;
    _simulationFacade->setSimulationParameters(parameters);
    _simulationFacade->setOriginalSimulationParameters(origParameters);
}

void SimulationParametersWindow::onDeleteTab(int index)
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();

    for (int i = index; i < parameters.numSpots - 1; ++i) {
        parameters.spots[i] = parameters.spots[i + 1];
        origParameters.spots[i] = origParameters.spots[i + 1];
    }
    --parameters.numSpots;
    --origParameters.numSpots;
    _simulationFacade->setSimulationParameters(parameters);
    _simulationFacade->setOriginalSimulationParameters(origParameters);
}

void SimulationParametersWindow::onOpenParameters()
{
    GenericFileDialogs::get().showOpenFileDialog(
        "Open simulation parameters", "Simulation parameters (*.parameters){.parameters},.*", _startingPath, [&](std::filesystem::path const& path) {
        auto firstFilename = ifd::FileDialog::Instance().GetResult();
        auto firstFilenameCopy = firstFilename;
        _startingPath = firstFilenameCopy.remove_filename().string();

        SimulationParameters parameters;
        if (!SerializerService::deserializeSimulationParametersFromFile(parameters, firstFilename.string())) {
            MessageDialog::get().information("Open simulation parameters", "The selected file could not be opened.");
        } else {
            _simulationFacade->setSimulationParameters(parameters);
        }
    });
}

void SimulationParametersWindow::onSaveParameters()
{
    GenericFileDialogs::get().showSaveFileDialog(
        "Save simulation parameters", "Simulation parameters (*.parameters){.parameters},.*", _startingPath, [&](std::filesystem::path const& path) {
        auto firstFilename = ifd::FileDialog::Instance().GetResult();
        auto firstFilenameCopy = firstFilename;
        _startingPath = firstFilenameCopy.remove_filename().string();

        auto parameters = _simulationFacade->getSimulationParameters();
        if (!SerializerService::serializeSimulationParametersToFile(firstFilename.string(), parameters)) {
            MessageDialog::get().information("Save simulation parameters", "The selected file could not be saved.");
        }
    });
}

void SimulationParametersWindow::validationAndCorrectionLayout()
{
    _featureListHeight = std::max(0.0f, _featureListHeight);
}

void SimulationParametersWindow::validationAndCorrection(SimulationParameters& parameters) const
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_COLORS; ++j) {
            parameters.baseValues.cellFunctionAttackerFoodChainColorMatrix[i][j] =
                std::max(0.0f, std::min(1.0f, parameters.baseValues.cellFunctionAttackerFoodChainColorMatrix[i][j]));
            parameters.cellFunctionAttackerSameMutantPenalty[i][j] = std::max(0.0f, std::min(1.0f, parameters.cellFunctionAttackerSameMutantPenalty[i][j]));
            parameters.baseValues.cellFunctionAttackerNewComplexMutantPenalty[i][j] =
                std::max(0.0f, std::min(1.0f, parameters.baseValues.cellFunctionAttackerNewComplexMutantPenalty[i][j]));
            parameters.baseValues.cellFunctionAttackerGenomeComplexityBonus[i][j] =
                std::max(0.0f, parameters.baseValues.cellFunctionAttackerGenomeComplexityBonus[i][j]);
        }
        parameters.baseValues.radiationAbsorption[i] = std::max(0.0f, std::min(1.0f, parameters.baseValues.radiationAbsorption[i]));
        parameters.radiationAbsorptionHighVelocityPenalty[i] = std::max(0.0f, parameters.radiationAbsorptionHighVelocityPenalty[i]);
        parameters.radiationAbsorptionLowConnectionPenalty[i] = std::max(0.0f, parameters.radiationAbsorptionLowConnectionPenalty[i]);
        parameters.externalEnergyConditionalInflowFactor[i] = std::max(0.0f, std::min(1.0f, parameters.externalEnergyConditionalInflowFactor[i]));
        parameters.cellFunctionAttackerSensorDetectionFactor[i] = std::max(0.0f, std::min(1.0f, parameters.cellFunctionAttackerSensorDetectionFactor[i]));
        parameters.cellFunctionDetonatorChainExplosionProbability[i] =
            std::max(0.0f, std::min(1.0f, parameters.cellFunctionDetonatorChainExplosionProbability[i]));
        parameters.externalEnergyInflowFactor[i] =
            std::max(0.0f, std::min(1.0f, parameters.externalEnergyInflowFactor[i]));
        parameters.baseValues.cellMinEnergy[i] = std::min(parameters.baseValues.cellMinEnergy[i], parameters.cellNormalEnergy[i] * 0.95f);
        parameters.particleSplitEnergy[i] = std::max(0.0f, parameters.particleSplitEnergy[i]);
        parameters.baseValues.radiationAbsorptionLowGenomeComplexityPenalty[i] =
            std::max(0.0f, std::min(1.0f, parameters.baseValues.radiationAbsorptionLowGenomeComplexityPenalty[i]));
        parameters.baseValues.radiationAbsorptionLowVelocityPenalty[i] =
            std::max(0.0f, std::min(1.0f, parameters.baseValues.radiationAbsorptionLowVelocityPenalty[i]));
        parameters.genomeComplexitySizeFactor[i] = std::max(0.0f, parameters.genomeComplexitySizeFactor[i]);
        parameters.genomeComplexityRamificationFactor[i] = std::max(0.0f, parameters.genomeComplexityRamificationFactor[i]);
        parameters.genomeComplexityNeuronFactor[i] = std::max(0.0f, parameters.genomeComplexityNeuronFactor[i]);
    }
    parameters.externalEnergy = std::max(0.0f, parameters.externalEnergy);
    parameters.baseValues.cellMaxBindingEnergy = std::max(10.0f, parameters.baseValues.cellMaxBindingEnergy);
    parameters.timestepSize = std::max(0.0f, parameters.timestepSize);
    parameters.cellMaxAgeBalancerInterval = std::max(1000, std::min(1000000, parameters.cellMaxAgeBalancerInterval));
    parameters.cellGlowRadius = std::max(1.0f, std::min(8.0f, parameters.cellGlowRadius));
    parameters.cellGlowStrength = std::max(0.0f, std::min(1.0f, parameters.cellGlowStrength));
}

void SimulationParametersWindow::validationAndCorrection(SimulationParametersSpot& spot, SimulationParameters const& parameters) const
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_COLORS; ++j) {
            spot.values.cellFunctionAttackerFoodChainColorMatrix[i][j] =
                std::max(0.0f, std::min(1.0f, spot.values.cellFunctionAttackerFoodChainColorMatrix[i][j]));
            spot.values.cellFunctionAttackerGenomeComplexityBonus[i][j] = std::max(0.0f, spot.values.cellFunctionAttackerGenomeComplexityBonus[i][j]);
            spot.values.cellFunctionAttackerNewComplexMutantPenalty[i][j] =
                std::max(0.0f, std::min(1.0f, spot.values.cellFunctionAttackerNewComplexMutantPenalty[i][j]));
        }
        spot.values.radiationAbsorption[i] = std::max(0.0f, std::min(1.0f, spot.values.radiationAbsorption[i]));
        spot.values.cellMinEnergy[i] = std::min(parameters.baseValues.cellMinEnergy[i], parameters.cellNormalEnergy[i] * 0.95f);
        spot.values.radiationAbsorptionLowGenomeComplexityPenalty[i] =
            std::max(0.0f, std::min(1.0f, spot.values.radiationAbsorptionLowGenomeComplexityPenalty[i]));
        spot.values.radiationAbsorptionLowVelocityPenalty[i] = std::max(0.0f, std::min(1.0f, spot.values.radiationAbsorptionLowVelocityPenalty[i]));
    }
    spot.values.cellMaxBindingEnergy = std::max(10.0f, spot.values.cellMaxBindingEnergy);
}
