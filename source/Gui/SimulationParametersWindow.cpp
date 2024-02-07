#include "SimulationParametersWindow.h"

#include <ImFileDialog.h>
#include <imgui.h>
#include <Fonts/IconsFontAwesome5.h>

#include "Base/GlobalSettings.h"
#include "EngineInterface/SerializerService.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/SimulationParametersService.h"

#include "AlienImGui.h"
#include "CellFunctionStrings.h"
#include "GenericFileDialogs.h"
#include "HelpStrings.h"
#include "MessageDialog.h"
#include "RadiationSourcesWindow.h"
#include "OverlayMessageController.h"
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

_SimulationParametersWindow::_SimulationParametersWindow(
    SimulationController const& simController,
    RadiationSourcesWindow const& radiationSourcesWindow)
    : _AlienWindow("Simulation parameters", "windows.simulation parameters", false)
    , _simController(simController)
    , _radiationSourcesWindow(radiationSourcesWindow)
{
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
    _startingPath = GlobalSettings::getInstance().getStringState("windows.simulation parameters.starting path", path.string());
    _featureListOpen = GlobalSettings::getInstance().getBoolState("windows.simulation parameters.feature list.open", _featureListOpen);
    _featureListHeight = GlobalSettings::getInstance().getFloatState("windows.simulation parameters.feature list.height", _featureListHeight);

    for (int i = 0; i < CellFunction_Count; ++i) {
        _cellFunctionStrings.emplace_back(Const::CellFunctionToStringMap.at(i));
    }
}

_SimulationParametersWindow::~_SimulationParametersWindow()
{
    GlobalSettings::getInstance().setStringState("windows.simulation parameters.starting path", _startingPath);
    GlobalSettings::getInstance().setBoolState("windows.simulation parameters.feature list.open", _featureListOpen);
    GlobalSettings::getInstance().setFloatState("windows.simulation parameters.feature list.height", _featureListHeight);
}

void _SimulationParametersWindow::processIntern()
{
    auto parameters = _simController->getSimulationParameters();
    auto origParameters = _simController->getOriginalSimulationParameters();
    auto lastParameters = parameters;

    processToolbar();
    if (ImGui::BeginChild("", {0, _featureListOpen ? -scale(_featureListHeight) : -scale(50.0f)})) {
        processTabWidget(parameters, lastParameters, origParameters);
    }
    ImGui::EndChild();
    processAddonList(parameters, lastParameters, origParameters);

    if (parameters != lastParameters) {
        _simController->setSimulationParameters(parameters);
    }
}

SimulationParametersSpot _SimulationParametersWindow::createSpot(SimulationParameters const& simParameters, int index)
{
    auto worldSize = _simController->getWorldSize();
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

void _SimulationParametersWindow::createDefaultSpotData(SimulationParametersSpot& spot)
{
    auto worldSize = _simController->getWorldSize();

    auto maxRadius = toFloat(std::min(worldSize.x, worldSize.y)) / 2;
    if (spot.shapeType == SpotShapeType_Circular) {
        spot.shapeData.circularSpot.coreRadius = maxRadius / 3;
    } else {
        spot.shapeData.rectangularSpot.height = maxRadius / 3;
        spot.shapeData.rectangularSpot.width = maxRadius / 3;
    }
}

void _SimulationParametersWindow::processToolbar()
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
        _copiedParameters = _simController->getSimulationParameters();
        printOverlayMessage("Simulation parameters copied");
    }
    AlienImGui::Tooltip("Copy simulation parameters");

    ImGui::SameLine();
    ImGui::BeginDisabled(!_copiedParameters);
    if (AlienImGui::ToolbarButton(ICON_FA_PASTE)) {
        _simController->setSimulationParameters(*_copiedParameters);
        _simController->setOriginalSimulationParameters(*_copiedParameters);
        printOverlayMessage("Simulation parameters pasted");
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Paste simulation parameters");

    AlienImGui::Separator();
}

void _SimulationParametersWindow::processTabWidget(
    SimulationParameters& parameters,
    SimulationParameters const& lastParameters,
    SimulationParameters& origParameters)
{
    auto focusBaseTab = !_numSpotsLastTime.has_value() || parameters.numSpots != *_numSpotsLastTime;

    if (ImGui::BeginChild("##", ImVec2(0, 0), false)) {

        if (ImGui::BeginTabBar("##Parameters", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown)) {

            //add spot
            if (parameters.numSpots < MAX_SPOTS) {
                if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) {
                    int index = parameters.numSpots;
                    parameters.spots[index] = createSpot(parameters, index);
                    origParameters.spots[index] = createSpot(parameters, index);
                    ++parameters.numSpots;
                    ++origParameters.numSpots;
                    _numSpotsLastTime = parameters.numSpots;
                    _simController->setSimulationParameters(parameters);
                    _simController->setOriginalSimulationParameters(origParameters);
                }
                AlienImGui::Tooltip("Add parameter zone");
            }

            if (ImGui::BeginTabItem("Base", nullptr, focusBaseTab ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None)) {
                processBase(parameters, origParameters);
                ImGui::EndTabItem();
            }

            for (int tab = 0; tab < parameters.numSpots; ++tab) {
                SimulationParametersSpot& spot = parameters.spots[tab];
                SimulationParametersSpot const& origSpot = origParameters.spots[tab];
                bool open = true;
                std::string name = "Zone " + std::to_string(tab + 1);
                if (ImGui::BeginTabItem(name.c_str(), &open, ImGuiTabItemFlags_None)) {
                    processSpot(spot, origSpot, parameters);
                    ImGui::EndTabItem();
                }

                //delete spot
                if (!open) {
                    for (int i = tab; i < parameters.numSpots - 1; ++i) {
                        parameters.spots[i] = parameters.spots[i + 1];
                        origParameters.spots[i] = origParameters.spots[i + 1];
                    }
                    --parameters.numSpots;
                    --origParameters.numSpots;
                    _numSpotsLastTime = parameters.numSpots;
                    _simController->setSimulationParameters(parameters);
                    _simController->setOriginalSimulationParameters(origParameters);
                }
            }

            ImGui::EndTabBar();
        }
    }
    ImGui::EndChild();
    _numSpotsLastTime = parameters.numSpots;
}

void _SimulationParametersWindow::processBase(
    SimulationParameters& parameters,
    SimulationParameters const& origParameters)
{
    if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {

        /**
         * Coloring
         */
        if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Visualization"))) {
            AlienImGui::ColorButtonWithPicker(
                AlienImGui::ColorButtonWithPickerParameters().name("Background color").textWidth(RightColumnWidth).defaultValue(origParameters.backgroundColor),
                parameters.backgroundColor,
                _backupColor,
                _savedPalette);
            AlienImGui::Switcher(
                AlienImGui::SwitcherParameters()
                    .name("Cell coloring")
                    .textWidth(RightColumnWidth)
                    .defaultValue(origParameters.cellColoring)
                    .values({"None", "Standard cell colors", "Mutants", "Cell state", "Genome complexity", "Highlight cell function"})
                    .tooltip("Here, one can set how the cells are to be colored during rendering. \n\n"
                            ICON_FA_CHEVRON_RIGHT " Standard cell colors: Each cell is assigned one of 7 default colors, which is displayed with this option. \n\n" ICON_FA_CHEVRON_RIGHT
                             " Mutants: Different mutants are represented by different colors (only larger structural mutations such as translations or duplications are taken into account).\n\n" ICON_FA_CHEVRON_RIGHT
                        " Cell state: green = under construction, blue = ready, red = dying\n\n" ICON_FA_CHEVRON_RIGHT
                        " Genome complexity: This property can be utilized by attacker cells when the parameter 'Complex genome protection' is "
                        "activated (see tooltip there). The coloring is as follows: blue = creature with low bonus (usually small or simple genome structure), red = large bonus"),
                parameters.cellColoring);
            AlienImGui::Switcher(
                AlienImGui::SwitcherParameters()
                    .name("Highlighted cell function")
                    .textWidth(RightColumnWidth)
                    .defaultValue(origParameters.highlightedCellFunction)
                    .values(_cellFunctionStrings)
                    .disabled(parameters.cellColoring != CellColoring_CellFunction)
                    .tooltip("The specific cell function type to be highlighted can be selected here."),
                parameters.highlightedCellFunction);
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
                    .name("Show detonations")
                    .textWidth(RightColumnWidth)
                    .defaultValue(origParameters.showDetonations)
                    .tooltip("If activated, the explosions of detonator cells will be visualized."),
                parameters.showDetonations);
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
                    .defaultValue(&origParameters.baseValues.cellMaxForce)
                    .tooltip(std::string("Maximum force that can be applied to a cell without causing it to disintegrate.")),
                &parameters.baseValues.cellMaxForce);
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
                    .defaultValue(&origParameters.cellMaxBindingDistance)
                    .tooltip(std::string("Maximum distance up to which a connection of two cells is possible.")),
                &parameters.cellMaxBindingDistance);
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
                    .tooltip(std::string("Maximum energy of a cell at which it does not disintegrate.")),
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
                _radiationSourcesWindow->setOn(true);
            }

            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Absorption factor")
                    .tooltip("")
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
                    .tooltip("")
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
                    .tooltip("Adjusts the maximum age at regular intervals. It increases the maximum age for the cell color where the fewest replicators exist. "
                             "Conversely, the maximum age is decreased for the cell color with the most replicators."),
                &parameters.cellMaxAgeBalancerInterval,
                &parameters.cellMaxAgeBalancer);
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
                    .tooltip("The normal energy value of a cell is defined here. This is used as a reference value in various contexts: \n"
                             ICON_FA_CARET_RIGHT" Attacker and Transmitter cells: When the energy of these cells is above the normal value, some of their energy is distributed to "
                             "surrounding cells.\n"
                             ICON_FA_CARET_RIGHT" Constructor cells: Creating new cells costs energy. The creation of new cells is executed only when the "
                             "residual energy of the constructor cell does not fall below the normal value.\n"
                             ICON_FA_CARET_RIGHT" If the transformation of energy particles to "
                             "cells is activated, an energy particle will transform into a cell if the energy of the particle exceeds the normal value."),
                parameters.cellNormalEnergy);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Decay rate of dying cells")
                    .colorDependence(true)
                    .textWidth(RightColumnWidth)
                    .min(1e-6f)
                    .max(0.05f)
                    .format("%.6f")
                    .logarithmic(true)
                    .defaultValue(origParameters.clusterDecayProb)
                    .tooltip(
                        "The probability per time step with which a cell will disintegrate (i.e. transform into an energy particle) provided that one of the following conditions is satisfied:\n" ICON_FA_CARET_RIGHT
                        " the cell has too low energy,\n" ICON_FA_CARET_RIGHT " the cell is in 'Dying' state\n" ICON_FA_CARET_RIGHT " the cell has exceeded the maximum age.")
                , parameters.clusterDecayProb);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Cell network decay")
                    .textWidth(RightColumnWidth)
                    .defaultValue(origParameters.clusterDecay)
                    .tooltip("If enabled, entire cell networks will disintegrate when one of their cells is dying because of insufficient energy or exceeding "
                             "the max. age. This option is useful to minimize the presence of damaged cell networks."),
                parameters.clusterDecay);
            AlienImGui::EndTreeNode();
        }
        ImGui::PopID();

        /**
         * Mutation 
         */
        if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Cell function: Genome mutation probabilities"))) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Neural net")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origParameters.baseValues.cellFunctionConstructorMutationNeuronDataProbability)
                    .tooltip("This type of mutation only changes the weights and biases of neural networks."),
                parameters.baseValues.cellFunctionConstructorMutationNeuronDataProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Cell properties")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origParameters.baseValues.cellFunctionConstructorMutationPropertiesProbability)
                    .tooltip("This type of mutation changes a random property (e.g. (input) execution order number, required energy, block output and "
                             "function-specific properties such as minimum density for sensors, neural net weights etc.). The spatial structure, color, cell "
                             "function type and self-replication capabilities are not changed."),
                parameters.baseValues.cellFunctionConstructorMutationPropertiesProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Geometry")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origParameters.baseValues.cellFunctionConstructorMutationGeometryProbability)
                    .tooltip("This type of mutation changes the geometry type, connection distance, stiffness and single construction flag."),
                parameters.baseValues.cellFunctionConstructorMutationGeometryProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Custom geometry")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origParameters.baseValues.cellFunctionConstructorMutationCustomGeometryProbability)
                    .tooltip("This type of mutation only changes angles and required connections of custom geometries ."),
                parameters.baseValues.cellFunctionConstructorMutationCustomGeometryProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Cell function type")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origParameters.baseValues.cellFunctionConstructorMutationCellFunctionProbability)
                    .tooltip("This type of mutation changes the type of cell function. The changed cell function will have random properties. If the "
                             "flag 'Preserve self-replication' is disabled it can also alter self-replication capabilities by changing a constructor to "
                             "something else or vice versa."),
                parameters.baseValues.cellFunctionConstructorMutationCellFunctionProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Insertion")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origParameters.baseValues.cellFunctionConstructorMutationInsertionProbability)
                    .tooltip("This type of mutation inserts a new cell description to the genome at a random position."),
                parameters.baseValues.cellFunctionConstructorMutationInsertionProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Deletion")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origParameters.baseValues.cellFunctionConstructorMutationDeletionProbability)
                    .tooltip("This type of mutation deletes a cell description from the genome at a random position."),
                parameters.baseValues.cellFunctionConstructorMutationDeletionProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Translation")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origParameters.baseValues.cellFunctionConstructorMutationTranslationProbability)
                    .tooltip("This type of mutation moves a block of cell descriptions from the genome at a random position to a new random position."),
                parameters.baseValues.cellFunctionConstructorMutationTranslationProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Duplication")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origParameters.baseValues.cellFunctionConstructorMutationDuplicationProbability)
                    .tooltip("This type of mutation copies a block of cell descriptions from the genome at a random position to a new random position."),
                parameters.baseValues.cellFunctionConstructorMutationDuplicationProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Individual cell color")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origParameters.baseValues.cellFunctionConstructorMutationCellColorProbability)
                    .tooltip("This type of mutation alters the color of a single cell descriptions in a genome by using the specified color transitions."),
                parameters.baseValues.cellFunctionConstructorMutationCellColorProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Sub-genome color")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origParameters.baseValues.cellFunctionConstructorMutationSubgenomeColorProbability)
                    .tooltip("This type of mutation alters the color of all cell descriptions in a sub-genome by using the specified color transitions."),
                parameters.baseValues.cellFunctionConstructorMutationSubgenomeColorProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Genome color")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origParameters.baseValues.cellFunctionConstructorMutationGenomeColorProbability)
                    .tooltip(
                        "This type of mutation alters the color of all cell descriptions in a genome by using the specified color transitions."),
                parameters.baseValues.cellFunctionConstructorMutationGenomeColorProbability);
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
                    .name("Pump energy")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0.00f)
                    .max(1.0f)
                    .defaultValue(origParameters.cellFunctionConstructorPumpEnergyFactor)
                    .tooltip("This parameter controls the energy pump system. It describes the fraction of the energy cost for a offspring which a constructor "
                             "can get for free. This additional energy is obtain from radiation of other cells."),
                parameters.cellFunctionConstructorPumpEnergyFactor);
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
                    .tooltip("")
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
                    .name("Forward/backward acceleration")
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
                    .max(512.0f)
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
                        .tooltip(""),
                    parameters.genomeComplexitySizeFactor);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Ramification factor")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(0.0f)
                        .max(20.0f)
                        .format("%.1f")
                        .defaultValue(origParameters.genomeComplexityRamificationFactor)
                        .tooltip(""),
                    parameters.genomeComplexityRamificationFactor);
                AlienImGui::EndTreeNode();
            }
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
                        .name("Velocity penalty")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(0)
                        .max(30.0f)
                        .logarithmic(true)
                        .format("%.1f")
                        .defaultValue(origParameters.radiationAbsorptionVelocityPenalty)
                        .tooltip("When this parameter is increased, cells with higher velocity will absorb less energy from an incoming energy particle."),
                    parameters.radiationAbsorptionVelocityPenalty);
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
         * Addon: External energy control
         */
        if (parameters.features.externalEnergyControl) {
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Addon: External energy control").highlighted(false))) {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("External energy amount")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(0.0f)
                        .max(100000000.0f)
                        .format("%.0f")
                        .logarithmic(true)
                        .infinity(true)
                        .defaultValue(origParameters.cellFunctionConstructorExternalEnergy)
                        .tooltip(
                            "This parameter can be used to set the amount of energy (per color) of an external energy source. This type of energy is "
                            "transferred to all constructor cells at a certain rate.\n\nTip: You can explicitly enter a numerical value by selecting the "
                            "slider and then pressing TAB.\n\nWarning: Too much external energy can result in a massive production of cells and slow down or "
                            "even crash the simulation."),
                    parameters.cellFunctionConstructorExternalEnergy);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("External energy supply rate")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(0.0f)
                        .max(1.0f)
                        .defaultValue(origParameters.cellFunctionConstructorExternalEnergySupplyRate)
                        .tooltip(
                            "The energy from the external source is transferred to all constructor cells at a rate defined here: 0 = no energy transfer, 1 = "
                            "constructor cells receive all the required energy"),
                    parameters.cellFunctionConstructorExternalEnergySupplyRate);
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
    }
    ImGui::EndChild();
    validationAndCorrection(parameters);
    validationAndCorrectionLayout();
}

void _SimulationParametersWindow::processSpot(
    SimulationParametersSpot& spot,
    SimulationParametersSpot const& origSpot,
    SimulationParameters const& parameters)
{
    if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
        auto worldSize = _simController->getWorldSize();

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
            if (AlienImGui::Combo(
                AlienImGui::ComboParameters()
                    .name("Shape")
                    .values({"Circular", "Rectangular"})
                    .textWidth(RightColumnWidth)
                    .defaultValue(origSpot.shapeType),
                spot.shapeType)) {
                createDefaultSpotData(spot);
            }
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Position X")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(toFloat(worldSize.x))
                    .defaultValue(&origSpot.posX)
                    .format("%.1f"),
                &spot.posX);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Position Y")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(toFloat(worldSize.y))
                    .defaultValue(&origSpot.posY)
                    .format("%.2f"),
                &spot.posY);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Velocity X")
                    .textWidth(RightColumnWidth)
                    .min(-4.0f)
                    .max(4.0f)
                    .defaultValue(&origSpot.velX)
                    .format("%.2f"),
                &spot.velX);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Velocity Y")
                    .textWidth(RightColumnWidth)
                    .min(-4.0f)
                    .max(4.0f)
                    .defaultValue(&origSpot.velY)
                    .format("%.2f"),
                &spot.velY);
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
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Core width")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(worldSize.x)
                        .defaultValue(&origSpot.shapeData.rectangularSpot.width)
                        .format("%.1f"),
                    &spot.shapeData.rectangularSpot.width);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Core height")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(worldSize.y)
                        .defaultValue(&origSpot.shapeData.rectangularSpot.height)
                        .format("%.1f"),
                    &spot.shapeData.rectangularSpot.height);
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

            auto forceFieldTypeIntern = std::max(0, spot.flowType - 1); //FlowType_None should not be selectable in ComboBox
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
                    .defaultValue(&origSpot.values.cellMaxForce)
                    .disabledValue(&parameters.baseValues.cellMaxForce),
                &spot.values.cellMaxForce,
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
        if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Cell function: Genome mutation probabilities"))) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Neuron weights and biases")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .colorDependence(true)
                    .logarithmic(true)
                    .defaultValue(origSpot.values.cellFunctionConstructorMutationNeuronDataProbability)
                    .disabledValue(parameters.baseValues.cellFunctionConstructorMutationNeuronDataProbability),
                spot.values.cellFunctionConstructorMutationNeuronDataProbability,
                &spot.activatedValues.cellFunctionConstructorMutationNeuronDataProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Cell properties")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSpot.values.cellFunctionConstructorMutationPropertiesProbability)
                    .disabledValue(parameters.baseValues.cellFunctionConstructorMutationPropertiesProbability),
                spot.values.cellFunctionConstructorMutationPropertiesProbability,
                &spot.activatedValues.cellFunctionConstructorMutationPropertiesProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Geometry")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSpot.values.cellFunctionConstructorMutationGeometryProbability)
                    .disabledValue(parameters.baseValues.cellFunctionConstructorMutationGeometryProbability),
                spot.values.cellFunctionConstructorMutationGeometryProbability,
                &spot.activatedValues.cellFunctionConstructorMutationGeometryProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Custom geometry")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSpot.values.cellFunctionConstructorMutationCustomGeometryProbability)
                    .disabledValue(parameters.baseValues.cellFunctionConstructorMutationCustomGeometryProbability),
                spot.values.cellFunctionConstructorMutationCustomGeometryProbability,
                &spot.activatedValues.cellFunctionConstructorMutationCustomGeometryProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Cell function type")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSpot.values.cellFunctionConstructorMutationCellFunctionProbability)
                    .disabledValue(parameters.baseValues.cellFunctionConstructorMutationCellFunctionProbability),
                spot.values.cellFunctionConstructorMutationCellFunctionProbability,
                &spot.activatedValues.cellFunctionConstructorMutationCellFunctionProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Cell insertion")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSpot.values.cellFunctionConstructorMutationInsertionProbability)
                    .disabledValue(parameters.baseValues.cellFunctionConstructorMutationInsertionProbability),
                spot.values.cellFunctionConstructorMutationInsertionProbability,
                &spot.activatedValues.cellFunctionConstructorMutationInsertionProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Cell deletion")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSpot.values.cellFunctionConstructorMutationDeletionProbability)
                    .disabledValue(parameters.baseValues.cellFunctionConstructorMutationDeletionProbability),
                spot.values.cellFunctionConstructorMutationDeletionProbability,
                &spot.activatedValues.cellFunctionConstructorMutationDeletionProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Translation")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSpot.values.cellFunctionConstructorMutationTranslationProbability)
                    .disabledValue(parameters.baseValues.cellFunctionConstructorMutationTranslationProbability),
                spot.values.cellFunctionConstructorMutationTranslationProbability,
                &spot.activatedValues.cellFunctionConstructorMutationTranslationProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Duplication")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSpot.values.cellFunctionConstructorMutationDuplicationProbability)
                    .disabledValue(parameters.baseValues.cellFunctionConstructorMutationDuplicationProbability),
                spot.values.cellFunctionConstructorMutationDuplicationProbability,
                &spot.activatedValues.cellFunctionConstructorMutationDuplicationProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Color")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSpot.values.cellFunctionConstructorMutationSubgenomeColorProbability)
                    .disabledValue(parameters.baseValues.cellFunctionConstructorMutationSubgenomeColorProbability),
                spot.values.cellFunctionConstructorMutationSubgenomeColorProbability,
                &spot.activatedValues.cellFunctionConstructorMutationSubgenomeColorProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Uniform color")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSpot.values.cellFunctionConstructorMutationGenomeColorProbability)
                    .disabledValue(parameters.baseValues.cellFunctionConstructorMutationGenomeColorProbability),
                spot.values.cellFunctionConstructorMutationGenomeColorProbability,
                &spot.activatedValues.cellFunctionConstructorMutationGenomeColorProbability);
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
}

void _SimulationParametersWindow::processAddonList(
    SimulationParameters& parameters,
    SimulationParameters const& lastParameters,
    SimulationParameters const& origParameters)
{
    if (_featureListOpen) {
        ImGui::Spacing();
        ImGui::Spacing();
        AlienImGui::MovableSeparator(_featureListHeight);
    } else {
        AlienImGui::Separator();
    }

    _featureListOpen = AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Addons").highlighted(true));
    if (_featureListOpen) {
        if (ImGui::BeginChild("##addons", {scale(0), 0})) {
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Genome complexity measurement")
                    .textWidth(0)
                    .defaultValue(origParameters.features.genomeComplexityMeasurement)
                    .tooltip("Parameters for the calculation of genome complexity are activated here. This genome complexity can be used for 'Advanced absorption control' "
                             "and 'Advanced attacker control' to favor more complex genomes in natural selection. If it is deactivated, default values are "
                             "used that simply take the genome size into account."),
                parameters.features.genomeComplexityMeasurement);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Advanced absorption control")
                    .textWidth(0)
                    .defaultValue(origParameters.features.advancedAbsorptionControl)
                    .tooltip("This addon offers extended possibilities for controlling the absorption of energy particles by cells."),
                parameters.features.advancedAbsorptionControl);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Advanced attacker control")
                    .textWidth(0)
                    .defaultValue(origParameters.features.advancedAttackerControl)
                    .tooltip("It contains further settings that influence how much energy can be obtained from an attack by attacker cells."),
                parameters.features.advancedAttackerControl);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("External energy control")
                    .textWidth(0)
                    .defaultValue(origParameters.features.externalEnergyControl)
                    .tooltip("This addon is used to add an external energy source. The energy is gradually transferred to the cells in the simulation."),
                parameters.features.externalEnergyControl);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Cell color transition rules")
                    .textWidth(0)
                    .defaultValue(origParameters.features.cellColorTransitionRules)
                    .tooltip("This can be used to define color transitions for cells depending on their age."),
                parameters.features.cellColorTransitionRules);
        }
        ImGui::EndChild();
        AlienImGui::EndTreeNode();
    }
}

void _SimulationParametersWindow::onOpenParameters()
{
    GenericFileDialogs::getInstance().showOpenFileDialog(
        "Open simulation parameters", "Simulation parameters (*.parameters){.parameters},.*", _startingPath, [&](std::filesystem::path const& path) {
        auto firstFilename = ifd::FileDialog::Instance().GetResult();
        auto firstFilenameCopy = firstFilename;
        _startingPath = firstFilenameCopy.remove_filename().string();

        SimulationParameters parameters;
        if (!SerializerService::deserializeSimulationParametersFromFile(parameters, firstFilename.string())) {
            MessageDialog::getInstance().information("Open simulation parameters", "The selected file could not be opened.");
        } else {
            _simController->setSimulationParameters(parameters);
        }
    });
}

void _SimulationParametersWindow::onSaveParameters()
{
    GenericFileDialogs::getInstance().showSaveFileDialog(
        "Save simulation parameters", "Simulation parameters (*.parameters){.parameters},.*", _startingPath, [&](std::filesystem::path const& path) {
        auto firstFilename = ifd::FileDialog::Instance().GetResult();
        auto firstFilenameCopy = firstFilename;
        _startingPath = firstFilenameCopy.remove_filename().string();

        auto parameters = _simController->getSimulationParameters();
        if (!SerializerService::serializeSimulationParametersToFile(firstFilename.string(), parameters)) {
            MessageDialog::getInstance().information("Save simulation parameters", "The selected file could not be saved.");
        }
    });
}

void _SimulationParametersWindow::validationAndCorrectionLayout()
{
    _featureListHeight = std::max(0.0f, _featureListHeight);
}

void _SimulationParametersWindow::validationAndCorrection(SimulationParameters& parameters) const
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_COLORS; ++j) {
            parameters.baseValues.cellFunctionAttackerFoodChainColorMatrix[i][j] =
                std::max(0.0f, std::min(1.0f, parameters.baseValues.cellFunctionAttackerFoodChainColorMatrix[i][j]));
            parameters.cellFunctionAttackerSameMutantPenalty[i][j] = std::max(0.0f, std::min(1.0f, parameters.cellFunctionAttackerSameMutantPenalty[i][j]));
            parameters.baseValues.cellFunctionAttackerGenomeComplexityBonus[i][j] =
                std::max(0.0f, parameters.baseValues.cellFunctionAttackerGenomeComplexityBonus[i][j]);
        }
        parameters.baseValues.radiationAbsorption[i] = std::max(0.0f, std::min(1.0f, parameters.baseValues.radiationAbsorption[i]));
        parameters.radiationAbsorptionVelocityPenalty[i] = std::max(0.0f, parameters.radiationAbsorptionVelocityPenalty[i]);
        parameters.radiationAbsorptionLowConnectionPenalty[i] = std::max(0.0f, parameters.radiationAbsorptionLowConnectionPenalty[i]);
        parameters.cellFunctionConstructorPumpEnergyFactor[i] = std::max(0.0f, std::min(1.0f, parameters.cellFunctionConstructorPumpEnergyFactor[i]));
        parameters.cellFunctionAttackerSensorDetectionFactor[i] = std::max(0.0f, std::min(1.0f, parameters.cellFunctionAttackerSensorDetectionFactor[i]));
        parameters.cellFunctionDetonatorChainExplosionProbability[i] =
            std::max(0.0f, std::min(1.0f, parameters.cellFunctionDetonatorChainExplosionProbability[i]));
        parameters.cellFunctionConstructorExternalEnergy[i] = std::max(0.0f, parameters.cellFunctionConstructorExternalEnergy[i]);
        parameters.cellFunctionConstructorExternalEnergySupplyRate[i] =
            std::max(0.0f, std::min(1.0f, parameters.cellFunctionConstructorExternalEnergySupplyRate[i]));
        parameters.baseValues.cellMinEnergy[i] = std::min(parameters.baseValues.cellMinEnergy[i], parameters.cellNormalEnergy[i] * 0.95f);
        parameters.particleSplitEnergy[i] = std::max(0.0f, parameters.particleSplitEnergy[i]);
        parameters.baseValues.radiationAbsorptionLowGenomeComplexityPenalty[i] =
            std::max(0.0f, std::min(1.0f, parameters.baseValues.radiationAbsorptionLowGenomeComplexityPenalty[i]));
    }
    parameters.baseValues.cellMaxBindingEnergy = std::max(10.0f, parameters.baseValues.cellMaxBindingEnergy);
    parameters.timestepSize = std::max(0.0f, parameters.timestepSize);
    parameters.cellMaxAgeBalancerInterval = std::max(1000, std::min(1000000, parameters.cellMaxAgeBalancerInterval));
}

void _SimulationParametersWindow::validationAndCorrection(SimulationParametersSpot& spot, SimulationParameters const& parameters) const
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_COLORS; ++j) {
            spot.values.cellFunctionAttackerFoodChainColorMatrix[i][j] =
                std::max(0.0f, std::min(1.0f, spot.values.cellFunctionAttackerFoodChainColorMatrix[i][j]));
            spot.values.cellFunctionAttackerGenomeComplexityBonus[i][j] = std::max(0.0f, spot.values.cellFunctionAttackerGenomeComplexityBonus[i][j]);
        }
        spot.values.radiationAbsorption[i] = std::max(0.0f, std::min(1.0f, spot.values.radiationAbsorption[i]));
        spot.values.cellMinEnergy[i] = std::min(parameters.baseValues.cellMinEnergy[i], parameters.cellNormalEnergy[i] * 0.95f);
        spot.values.radiationAbsorptionLowGenomeComplexityPenalty[i] =
            std::max(0.0f, std::min(1.0f, spot.values.radiationAbsorptionLowGenomeComplexityPenalty[i]));
    }
    spot.values.cellMaxBindingEnergy = std::max(10.0f, spot.values.cellMaxBindingEnergy);
}
