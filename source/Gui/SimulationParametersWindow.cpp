#include "SimulationParametersWindow.h"

#include <ImFileDialog.h>
#include <imgui.h>
#include <Fonts/IconsFontAwesome5.h>

#include "EngineInterface/Serializer.h"
#include "EngineInterface/SimulationController.h"

#include "AlienImGui.h"
#include "GenericFileDialogs.h"
#include "StyleRepository.h"
#include "GlobalSettings.h"
#include "MessageDialog.h"
#include "RadiationSourcesWindow.h"
#include "BalancerController.h"

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
    RadiationSourcesWindow const& radiationSourcesWindow,
    BalancerController const& balancerController)
    : _AlienWindow("Simulation parameters", "windows.simulation parameters", false)
    , _simController(simController)
    , _radiationSourcesWindow(radiationSourcesWindow)
    , _balancerController(balancerController)
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
}

_SimulationParametersWindow::~_SimulationParametersWindow()
{
    GlobalSettings::getInstance().setStringState("windows.simulation parameters.starting path", _startingPath);
}

void _SimulationParametersWindow::processIntern()
{
    processToolbar();

    auto parameters = _simController->getSimulationParameters();
    auto origParameters = _simController->getOriginalSimulationParameters();
    auto lastParameters = parameters;

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
                    _simController->setSimulationParameters_async(parameters);
                    _simController->setOriginalSimulationParameters(origParameters);
                }
                AlienImGui::Tooltip("Add spot");
            }

            if (ImGui::BeginTabItem("Base", nullptr, ImGuiTabItemFlags_None)) {
                processBase(parameters, origParameters);
                ImGui::EndTabItem();
            }

            for (int tab = 0; tab < parameters.numSpots; ++tab) {
                SimulationParametersSpot& spot = parameters.spots[tab];
                SimulationParametersSpot const& origSpot = origParameters.spots[tab];
                bool open = true;
                std::string name = "Spot " + std::to_string(tab+1);
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
                    _simController->setSimulationParameters_async(parameters);
                    _simController->setOriginalSimulationParameters(origParameters);
                }
            }

            ImGui::EndTabBar();
        }
    }
    ImGui::EndChild();

    if (parameters != lastParameters) {
        _simController->setSimulationParameters_async(parameters);
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
    }
    AlienImGui::Tooltip("Copy simulation parameters");

    ImGui::SameLine();
    ImGui::BeginDisabled(!_copiedParameters);
    if (AlienImGui::ToolbarButton(ICON_FA_PASTE)) {
        _simController->setSimulationParameters(*_copiedParameters);
        _simController->setOriginalSimulationParameters(*_copiedParameters);
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Paste simulation parameters");

    AlienImGui::Separator();
}

void _SimulationParametersWindow::processBase(
    SimulationParameters& simParameters,
    SimulationParameters const& origSimParameters)
{
    if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_DefaultOpen;

        /**
         * Colors
         */
        if (ImGui::TreeNodeEx("Visualization", flags)) {
            AlienImGui::ColorButtonWithPicker(
                AlienImGui::ColorButtonWithPickerParameters().name("Background color").textWidth(RightColumnWidth).defaultValue(origSimParameters.backgroundColor),
                simParameters.backgroundColor,
                _backupColor,
                _savedPalette);
            AlienImGui::Combo(
                    AlienImGui::ComboParameters()
                        .name("Cell colorization")
                        .textWidth(RightColumnWidth)
                    .defaultValue(origSimParameters.cellColorization)
                        .values({"None", "Cell colors", "Mutations"}),
                    simParameters.cellColorization);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Zoom level for cell activity")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(32.0f)
                    .infinity(true)
                    .defaultValue(&origSimParameters.zoomLevelNeuronalActivity)
                    .tooltip(std::string("The zoom level from which the neuronal activities become visible.")),
                &simParameters.zoomLevelNeuronalActivity);
            ImGui::TreePop();
        }

        /**
         * Numerics
         */
        if (ImGui::TreeNodeEx("Numerics", flags)) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Time step size")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(1.0f)
                    .defaultValue(&origSimParameters.timestepSize)
                    .tooltip(std::string("The time duration calculated in a single simulation step. Smaller values increase the accuracy of the simulation "
                                         "while larger values can lead to numerical instabilities.")),
                &simParameters.timestepSize);
            ImGui::TreePop();
        }

        /**
         * Physics: Motion
         */
        if (ImGui::TreeNodeEx("Physics: Motion", flags)) {
            if (AlienImGui::Combo(
                    AlienImGui::ComboParameters()
                        .name("Motion type")
                        .textWidth(RightColumnWidth)
                        .defaultValue(origSimParameters.motionType)
                        .values({"Fluid dynamics", "Collision-based"})
                        .tooltip(std::string(
                            "The algorithm for the particle motions is defined here. If 'Fluid dynamics' is selected, an SPH fluid solver is used for the "
                            "calculation of the forces. The particles then behave like (compressible) liquids or gases. The other option 'Collision-based' "
                            "calculates the forces based on particle collisions and should be preferred for mechanical simulation with solids.")),
                    simParameters.motionType)) {
                if (simParameters.motionType == MotionType_Fluid) {
                    simParameters.motionData.fluidMotion = FluidMotion();
                } else {
                    simParameters.motionData.collisionMotion = CollisionMotion();
                }
            }
            if (simParameters.motionType == MotionType_Fluid) {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Smoothing length")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(3.0f)
                        .defaultValue(&origSimParameters.motionData.fluidMotion.smoothingLength)
                        .tooltip(std::string("The smoothing length determines the region of influence of the neighboring particles for the calculation of "
                                             "density, pressure and viscosity. Values that are too small lead to numerical instabilities, while values that "
                                             "are too large cause the particles to drift apart.")),
                    &simParameters.motionData.fluidMotion.smoothingLength);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Pressure")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(0.3f)
                        .defaultValue(&origSimParameters.motionData.fluidMotion.pressureStrength)
                        .tooltip(std::string("This parameter allows to control the strength of the pressure.")),
                    &simParameters.motionData.fluidMotion.pressureStrength);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Viscosity")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(0.3f)
                        .defaultValue(&origSimParameters.motionData.fluidMotion.viscosityStrength)
                        .tooltip(std::string("This parameter be used to control the strength of the viscosity. Larger values lead to a smoother movement.")),
                    &simParameters.motionData.fluidMotion.viscosityStrength);
            } else {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Repulsion strength")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(0.3f)
                        .defaultValue(&origSimParameters.motionData.collisionMotion.cellRepulsionStrength)
                        .tooltip(std::string("The strength of the repulsive forces, between two cells that are not connected.")),
                    &simParameters.motionData.collisionMotion.cellRepulsionStrength);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Maximum collision distance")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(3.0f)
                        .defaultValue(&origSimParameters.motionData.collisionMotion.cellMaxCollisionDistance)
                        .tooltip(std::string("Maximum distance up to which a collision of two cells is possible.")),
                    &simParameters.motionData.collisionMotion.cellMaxCollisionDistance);
            }
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Friction")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.4f")
                    .defaultValue(&origSimParameters.baseValues.friction)
                    .tooltip(std::string("This specifies the fraction of the velocity that is slowed down per time step.")),
                &simParameters.baseValues.friction);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Rigidity")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(1.0f)
                    .format("%.2f")
                    .defaultValue(&origSimParameters.baseValues.rigidity)
                    .tooltip(std::string(
                        "Controls the rigidity of connected cells.\nA higher value will cause connected cells to move more uniformly as a rigid body.")),
                &simParameters.baseValues.rigidity);
            ImGui::TreePop();
        }

        /**
         * Physics: Thresholds
         */
        if (ImGui::TreeNodeEx("Physics: Thresholds", flags)) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Maximum velocity")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(6.0f)
                    .defaultValue(&origSimParameters.cellMaxVelocity)
                    .tooltip(std::string("Maximum velocity that a cell can reach.")),
                &simParameters.cellMaxVelocity);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Maximum force")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(3.0f)
                    .defaultValue(&origSimParameters.baseValues.cellMaxForce)
                    .tooltip(std::string("Maximum force that can be applied to a cell without causing it to disintegrate.")),
                &simParameters.baseValues.cellMaxForce);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Minimum distance")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(1.0f)
                    .defaultValue(&origSimParameters.cellMinDistance)
                    .tooltip(std::string("Minimum distance between two cells.")),
                &simParameters.cellMinDistance);
            ImGui::TreePop();
        }

        /**
         * Physics: Binding
         */
        if (ImGui::TreeNodeEx("Physics: Binding", flags)) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Maximum distance")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(5.0f)
                    .defaultValue(&origSimParameters.cellMaxBindingDistance)
                    .tooltip(std::string("Maximum distance up to which a connection of two cells is possible.")),
                &simParameters.cellMaxBindingDistance);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Fusion velocity")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(2.0f)
                    .defaultValue(&origSimParameters.baseValues.cellFusionVelocity)
                    .tooltip(std::string("Minimum relative velocity of two colliding cells so that a connection can be established.")),
                &simParameters.baseValues.cellFusionVelocity);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Maximum energy")
                    .textWidth(RightColumnWidth)
                    .min(50.0f)
                    .max(10000000.0f)
                    .logarithmic(true)
                    .infinity(true)
                    .format("%.0f")
                    .defaultValue(&origSimParameters.baseValues.cellMaxBindingEnergy)
                    .tooltip(std::string("Maximum energy of a cell at which it does not disintegrate.")),
                &simParameters.baseValues.cellMaxBindingEnergy);
            ImGui::TreePop();
        }

        /**
         * Physics: Radiation
         */
        if (ImGui::TreeNodeEx("Physics: Radiation", flags)) {
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
                    .defaultValue(origSimParameters.baseValues.radiationAbsorption)
                    .tooltip("The fraction of energy that a cell can absorb from an incoming energy particle can be specified here."),
                simParameters.baseValues.radiationAbsorption);

            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Radiation type I: Strength")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(0.01f)
                    .logarithmic(true)
                    .format("%.6f")
                    .defaultValue(origSimParameters.baseValues.radiationCellAgeStrength)
                    .tooltip("Indicates how energetic the emitted particles of aged cells are."),
                simParameters.baseValues.radiationCellAgeStrength);
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
                    .defaultValue(origSimParameters.radiationMinCellAge)
                    .tooltip("The minimum age of a cell can be defined here, from which it emits energy particles."),
                simParameters.radiationMinCellAge);

            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Radiation type II: Strength")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(0.01f)
                    .logarithmic(true)
                    .format("%.6f")
                    .defaultValue(origSimParameters.highRadiationFactor)
                    .tooltip("Indicates how energetic the emitted particles of high energy cells are."),
                simParameters.highRadiationFactor);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Radiation type II: Energy threshold")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .infinity(true)
                    .min(0)
                    .max(100000)
                    .logarithmic(true)
                    .format("%.1f")
                    .defaultValue(origSimParameters.highRadiationMinCellEnergy)
                    .tooltip("The minimum energy of a cell can be defined here, from which it emits energy particles."),
                simParameters.highRadiationMinCellEnergy);

            ImGui::TreePop();
        }

        /**
         * Cell life cycle
         */
        ImGui::PushID("Transformation");
        if (ImGui::TreeNodeEx("Cell life cycle", flags)) {
            AlienImGui::SliderInt(
                AlienImGui::SliderIntParameters()
                    .name("Maximum age")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .logarithmic(true)
                    .infinity(true)
                    .min(1)
                    .max(10000000)
                    .defaultValue(origSimParameters.cellMaxAge)
                    .tooltip("Defines the maximum age of a cell. If a cell exceeds this age it will be transformed to an energy particle."),
                simParameters.cellMaxAge);
            AlienImGui::SliderInt(
                AlienImGui::SliderIntParameters()
                    .name("Maximum age balancing")
                    .textWidth(RightColumnWidth)
                    .logarithmic(true)
                    .min(1000)
                    .max(1000000)
                    .disabledValue(&simParameters.cellMaxAgeBalancerInterval)
                    .defaultEnabledValue(&origSimParameters.cellMaxAgeBalancer)
                    .defaultValue(&origSimParameters.cellMaxAgeBalancerInterval)
                    .tooltip("Adjusts the maximum age at regular intervals. It increases the maximum age for the cell color where the fewest replicators exist. "
                             "Conversely, the maximum age is decreased for the cell color with the most replicators."),
                &simParameters.cellMaxAgeBalancerInterval,
                &simParameters.cellMaxAgeBalancer);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Minimum energy")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(10.0f)
                    .max(200.0f)
                    .defaultValue(origSimParameters.baseValues.cellMinEnergy)
                    .tooltip("Minimum energy a cell needs to exist."),
                simParameters.baseValues.cellMinEnergy);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Normal energy")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(10.0f)
                    .max(200.0f)
                    .defaultValue(origSimParameters.cellNormalEnergy)
                    .tooltip("The normal energy value of a cell is defined here. This is used as a reference value in various contexts: \n"
                             ICON_FA_CARET_RIGHT" Attacker and Transmitter cells: When the energy of these cells is above the normal value, some of their energy is distributed to "
                             "surrounding cells.\n"
                             ICON_FA_CARET_RIGHT" Constructor cells: Creating new cells costs energy. The creation of new cells is executed only when the "
                             "residual energy of the constructor cell does not fall below the normal value.\n"
                             ICON_FA_CARET_RIGHT" If the transformation of energy particles to "
                             "cells is activated, an energy particle will transform into a cell if the energy of the particle exceeds the normal value."),
                simParameters.cellNormalEnergy);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Energy to cell transformation")
                    .textWidth(RightColumnWidth)
                    .defaultValue(origSimParameters.particleTransformationAllowed)
                    .tooltip("If activated, an energy particle will transform into a cell if the energy of the particle exceeds the normal energy value."),
                simParameters.particleTransformationAllowed);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Cell cluster decay")
                    .textWidth(RightColumnWidth)
                    .defaultValue(origSimParameters.clusterDecay)
                    .tooltip("If enabled, entire cell clusters will disintegrate when one of their cells is dying because of insufficient energy. This option "
                             "is useful to minimize the presence of cell corpses."),
                simParameters.clusterDecay);
            ImGui::TreePop();
        }
        ImGui::PopID();

        /**
         * Mutation 
         */
        if (ImGui::TreeNodeEx("Cell function: Genome mutation probabilities", flags)) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Neural net")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSimParameters.baseValues.cellFunctionConstructorMutationNeuronDataProbability)
                    .tooltip("This type of mutation only changes the weights and biases of neural networks."),
                simParameters.baseValues.cellFunctionConstructorMutationNeuronDataProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Cell properties")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSimParameters.baseValues.cellFunctionConstructorMutationPropertiesProbability)
                    .tooltip("This type of mutation changes a random property (e.g. (input) execution order number, required energy, block output and "
                             "function-specific properties such as minimum density for sensors, neural net weights etc.). The spatial structure, color, cell "
                             "function type and self-replication capabilities are not changed."),
                simParameters.baseValues.cellFunctionConstructorMutationPropertiesProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Geometry")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSimParameters.baseValues.cellFunctionConstructorMutationGeometryProbability)
                    .tooltip("This type of mutation changes the geometry type, connection distance, stiffness and single construction flag."),
                simParameters.baseValues.cellFunctionConstructorMutationGeometryProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Custom geometry")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSimParameters.baseValues.cellFunctionConstructorMutationCustomGeometryProbability)
                    .tooltip("This type of mutation only changes angles and required connections of custom geometries ."),
                simParameters.baseValues.cellFunctionConstructorMutationCustomGeometryProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Cell function type")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSimParameters.baseValues.cellFunctionConstructorMutationCellFunctionProbability)
                    .tooltip("This type of mutation changes the type of cell function. The changed cell function will have random properties. If the "
                             "flag 'Preserve self-replication' is disabled it can also alter self-replication capabilities by changing a constructor to "
                             "something else or vice versa."),
                simParameters.baseValues.cellFunctionConstructorMutationCellFunctionProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Insertion")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSimParameters.baseValues.cellFunctionConstructorMutationInsertionProbability)
                    .tooltip("This type of mutation inserts a new cell description to the genome at a random position."),
                simParameters.baseValues.cellFunctionConstructorMutationInsertionProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Deletion")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSimParameters.baseValues.cellFunctionConstructorMutationDeletionProbability)
                    .tooltip("This type of mutation deletes a cell description from the genome at a random position."),
                simParameters.baseValues.cellFunctionConstructorMutationDeletionProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Translation")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSimParameters.baseValues.cellFunctionConstructorMutationTranslationProbability)
                    .tooltip("This type of mutation moves a block of cell descriptions from the genome at a random position to a new random position."),
                simParameters.baseValues.cellFunctionConstructorMutationTranslationProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Duplication")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSimParameters.baseValues.cellFunctionConstructorMutationDuplicationProbability)
                    .tooltip("This type of mutation copies a block of cell descriptions from the genome at a random position to a new random position."),
                simParameters.baseValues.cellFunctionConstructorMutationDuplicationProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Color")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSimParameters.baseValues.cellFunctionConstructorMutationColorProbability)
                    .tooltip("This type of mutation alters the color of all cell descriptions in a sub-genome by using the specified color transitions."),
                simParameters.baseValues.cellFunctionConstructorMutationColorProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Uniform color")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSimParameters.baseValues.cellFunctionConstructorMutationUniformColorProbability)
                    .tooltip(
                        "This type of mutation alters the color of all cell descriptions in a genome by using the specified color transitions."),
                simParameters.baseValues.cellFunctionConstructorMutationUniformColorProbability);
            AlienImGui::CheckboxColorMatrix(
                AlienImGui::CheckboxColorMatrixParameters()
                    .name("Color transitions")
                    .textWidth(RightColumnWidth)
                    .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origSimParameters.cellFunctionConstructorMutationColorTransitions))
                    .tooltip(
                        "The color transitions are used for color mutations. The row index indicates the source color and the column index the target color."),
                simParameters.cellFunctionConstructorMutationColorTransitions);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Prevent genome depth increase")
                    .textWidth(RightColumnWidth)
                    .defaultValue(origSimParameters.cellFunctionConstructorMutationPreventDepthIncrease)
                    .tooltip(std::string("A genome has a tree-like structure because it can contain sub-genomes. If this flag is activated, the mutations will "
                                         "not increase the depth of the genome structure.")),
                simParameters.cellFunctionConstructorMutationPreventDepthIncrease);
            auto preserveSelfReplication = !simParameters.cellFunctionConstructorMutationSelfReplication;
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Preserve self-replication")
                    .textWidth(RightColumnWidth)
                    .defaultValue(!origSimParameters.cellFunctionConstructorMutationSelfReplication)
                    .tooltip("If deactivated, a mutation can also alter self-replication capabilities in the genome by changing a constructor cell to "
                             "something else or vice versa."),
                preserveSelfReplication);
            simParameters.cellFunctionConstructorMutationSelfReplication = !preserveSelfReplication;
            ImGui::TreePop();
        }

        /**
         * Attacker
         */
        ImGui::PushID("Attacker");
        if (ImGui::TreeNodeEx("Cell function: Attacker", flags)) {
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
                    .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origSimParameters.baseValues.cellFunctionAttackerFoodChainColorMatrix)),
                simParameters.baseValues.cellFunctionAttackerFoodChainColorMatrix);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Energy cost")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(1.0f)
                    .format("%.5f")
                    .logarithmic(true)
                    .defaultValue(origSimParameters.baseValues.cellFunctionAttackerEnergyCost)
                    .tooltip("Amount of energy lost by an attempted attack of a cell in form of emitted energy particles."),
                simParameters.baseValues.cellFunctionAttackerEnergyCost);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Attack strength")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(0.1f)
                    .defaultValue(origSimParameters.cellFunctionAttackerStrength),
                simParameters.cellFunctionAttackerStrength);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Attack radius")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(2.5f)
                    .defaultValue(origSimParameters.cellFunctionAttackerRadius)
                    .tooltip("The maximum distance over which an attacker cell can attack another cell."),
                simParameters.cellFunctionAttackerRadius);
            AlienImGui::InputFloatColorMatrix(
                AlienImGui::InputFloatColorMatrixParameters()
                    .name("Genome size bonus")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(5.0f)
                    .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origSimParameters.cellFunctionAttackerGenomeSizeBonus))
                    .tooltip("The larger this parameter is, the less energy can be gained by attacking creatures with larger genomes."),
                simParameters.cellFunctionAttackerGenomeSizeBonus);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Velocity penalty")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(5.0f)
                    .defaultValue(origSimParameters.cellFunctionAttackerVelocityPenalty)
                    .tooltip("This parameter reduces the captured energy during an attack when the attacker cell is moving. The faster the cell moves or the "
                             "higher this parameter is, the less energy is captured."),
                simParameters.cellFunctionAttackerVelocityPenalty);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Geometry penalty")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(5.0f)
                    .defaultValue(origSimParameters.baseValues.cellFunctionAttackerGeometryDeviationExponent)
                    .tooltip("The larger this value is, the less energy a cell can gain from an attack if the local "
                             "geometry of the attacked cell does not match the attacking cell."),
                simParameters.baseValues.cellFunctionAttackerGeometryDeviationExponent);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Connections mismatch penalty")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(1.0f)
                    .defaultValue(origSimParameters.baseValues.cellFunctionAttackerConnectionsMismatchPenalty)
                    .tooltip("The larger this parameter is, the more difficult it is to attack cells that contain more connections."),
                simParameters.baseValues.cellFunctionAttackerConnectionsMismatchPenalty);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Energy distribution radius")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(5.0f)
                    .defaultValue(origSimParameters.cellFunctionAttackerEnergyDistributionRadius)
                    .tooltip("The maximum distance over which an attacker cell transfers its energy captured during an attack to nearby transmitter or "
                             "constructor cells."),
                simParameters.cellFunctionAttackerEnergyDistributionRadius);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Energy distribution Value")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(20.0f)
                    .defaultValue(origSimParameters.cellFunctionAttackerEnergyDistributionValue)
                    .tooltip("The amount of energy which a attacker cell can transfer to nearby transmitter or constructor cells or to connected cells."),
                simParameters.cellFunctionAttackerEnergyDistributionValue);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Color inhomogeneity factor")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(2.0f)
                    .defaultValue(origSimParameters.cellFunctionAttackerColorInhomogeneityFactor)
                    .tooltip("If the attacked cell is connected to cells with different colors, this factor affects the energy of the captured energy."),
                simParameters.cellFunctionAttackerColorInhomogeneityFactor);
            ImGui::TreePop();
        }
        ImGui::PopID();

        /**
         * Constructor
         */
        if (ImGui::TreeNodeEx("Cell function: Constructor", flags)) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Pump energy")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0.00f)
                    .max(1.0f)
                    .defaultValue(origSimParameters.cellFunctionConstructorPumpEnergyFactor)
                    .tooltip("This parameter controls the energy pump system. It describes the fraction of the energy cost for a offspring which a constructor "
                             "can get for free. This additional energy is obtain from radiation of other cells."),
                simParameters.cellFunctionConstructorPumpEnergyFactor);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Offspring distance")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0.1f)
                    .max(3.0f)
                    .defaultValue(origSimParameters.cellFunctionConstructorOffspringDistance)
                    .tooltip("The distance of the constructed cell from the constructor cell."),
                simParameters.cellFunctionConstructorOffspringDistance);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Connection distance")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0.1f)
                    .max(3.0f)
                    .defaultValue(origSimParameters.cellFunctionConstructorConnectingCellMaxDistance)
                    .tooltip("The constructor can automatically connect constructed cells to other cells in the vicinity within this distance."),
                simParameters.cellFunctionConstructorConnectingCellMaxDistance);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Completeness check")
                    .textWidth(RightColumnWidth)
                    .defaultValue(origSimParameters.cellFunctionConstructorCheckCompletenessForSelfReplication)
                    .tooltip("If activated, a self-replication process can only start when all other non-self-replicating constructors in the cell clusters are "
                             "finished."),
                simParameters.cellFunctionConstructorCheckCompletenessForSelfReplication);
            ImGui::TreePop();
        }

        /**
         * Defender
         */
        if (ImGui::TreeNodeEx("Cell function: Defender", flags)) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Anti-attacker strength")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(1.0f)
                    .max(5.0f)
                    .defaultValue(origSimParameters.cellFunctionDefenderAgainstAttackerStrength)
                    .tooltip("If an attacked cell is connected to defender cells or itself a defender cell the attack strength is reduced by this factor."),
                simParameters.cellFunctionDefenderAgainstAttackerStrength);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Anti-injector strength")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(1.0f)
                    .max(5.0f)
                    .defaultValue(origSimParameters.cellFunctionDefenderAgainstInjectorStrength)
                    .tooltip("If a constructor cell is attacked by an injector and connected to defender cells, the injection duration is increased by this "
                             "factor."),
                simParameters.cellFunctionDefenderAgainstInjectorStrength);
            ImGui::TreePop();
        }

        /**
         * Injector
         */
        if (ImGui::TreeNodeEx("Cell function: Injector", flags)) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Injection radius")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0.1f)
                    .max(3.0f)
                    .defaultValue(origSimParameters.cellFunctionInjectorRadius)
                    .tooltip("The maximum distance over which an injector cell can infect another cell."),
                simParameters.cellFunctionInjectorRadius);
            AlienImGui::InputIntColorMatrix(
                AlienImGui::InputIntColorMatrixParameters()
                    .name("Injection time")
                    .logarithmic(true)
                    .max(100000)
                    .textWidth(RightColumnWidth)
                    .tooltip("")
                    .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origSimParameters.cellFunctionInjectorDurationColorMatrix))
                    .tooltip("The number of activations an injector cell requires to infect another cell. One activation usually takes 6 time steps. The row "
                             "number determines the color of the injector cell, while the column number corresponds to the color of the infected cell."),
                simParameters.cellFunctionInjectorDurationColorMatrix);
            ImGui::TreePop();
        }

        /**
         * Muscle
         */
        if (ImGui::TreeNodeEx("Cell function: Muscle", flags)) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Contraction and expansion delta")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(0.1f)
                    .defaultValue(origSimParameters.cellFunctionMuscleContractionExpansionDelta)
                    .tooltip("The maximum length that a muscle cell can shorten or lengthen a cell connection. This parameter applies only to muscle cells "
                             "which are in contraction/expansion mode."),
                simParameters.cellFunctionMuscleContractionExpansionDelta);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Forward/backward acceleration")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(0.4f)
                    .logarithmic(true)
                    .defaultValue(origSimParameters.cellFunctionMuscleMovementAcceleration)
                    .tooltip("The maximum value by which a muscle cell can modify its velocity during activation. This parameter applies only to muscle cells "
                             "which are in movement mode."),
                simParameters.cellFunctionMuscleMovementAcceleration);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Bending angle")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(10.0f)
                    .defaultValue(origSimParameters.cellFunctionMuscleBendingAngle)
                    .tooltip("The maximum value by which a muscle cell can increase/decrease the angle between two cell connections. This parameter applies "
                             "only to muscle cells which are in bending mode."),
                simParameters.cellFunctionMuscleBendingAngle);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Bending acceleration")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(0.5f)
                    .defaultValue(origSimParameters.cellFunctionMuscleBendingAcceleration)
                    .tooltip("The maximum value by which a muscle cell can modify its velocity during a bending action. This parameter applies "
                             "only to muscle cells which are in bending mode."),
                simParameters.cellFunctionMuscleBendingAcceleration);
            ImGui::TreePop();
        }

        /**
         * Sensor
         */
        if (ImGui::TreeNodeEx("Cell function: Sensor", flags)) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Range")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(10.0f)
                    .max(512.0f)
                    .defaultValue(origSimParameters.cellFunctionSensorRange)
                    .tooltip("The maximum radius in which a sensor cell can detect mass concentrations."),
                simParameters.cellFunctionSensorRange);
            ImGui::TreePop();
        }

        /**
         * Transmitter
         */
        if (ImGui::TreeNodeEx("Cell function: Transmitter", flags)) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Energy distribution radius")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(5.0f)
                    .defaultValue(origSimParameters.cellFunctionTransmitterEnergyDistributionRadius)
                    .tooltip("The maximum distance over which a transmitter cell transfers its additional energy to nearby transmitter or constructor cells."),
                simParameters.cellFunctionTransmitterEnergyDistributionRadius);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Energy distribution Value")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(20.0f)
                    .defaultValue(origSimParameters.cellFunctionTransmitterEnergyDistributionValue)
                    .tooltip("The amount of energy which a transmitter cell can transfer to nearby transmitter or constructor cells or to connected cells."),
                simParameters.cellFunctionTransmitterEnergyDistributionValue);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Same creature energy distribution")
                    .textWidth(RightColumnWidth)
                    .defaultValue(origSimParameters.cellFunctionTransmitterEnergyDistributionSameCreature)
                    .tooltip("If activated, the transmitter cells can only transfer energy to nearby cells belonging to the same creature."),
                simParameters.cellFunctionTransmitterEnergyDistributionSameCreature);
            ImGui::TreePop();
        }

        /**
         * Cell color transition rules
         */
        if (ImGui::TreeNodeEx("Cell color transition rules", flags)) {
            for (int color = 0; color < MAX_COLORS; ++color) {
                ImGui::PushID(color);
                auto parameters = AlienImGui::InputColorTransitionParameters()
                                      .textWidth(RightColumnWidth)
                                      .color(color)
                                      .defaultTargetColor(origSimParameters.baseValues.cellColorTransitionTargetColor[color])
                                      .defaultTransitionAge(origSimParameters.baseValues.cellColorTransitionDuration[color])
                                      .logarithmic(true)
                                      .infinity(true);
                if (0 == color) {
                    parameters.name("Target color and duration")
                        .tooltip("Rules can be defined that describe how the colors of cells will change over time. For this purpose, a subsequent color can "
                                 "be defined for each cell color. In addition, durations must be specified that define how many time steps the corresponding "
                                 "color are kept.");
                }
                AlienImGui::InputColorTransition(
                    parameters,
                    color,
                    simParameters.baseValues.cellColorTransitionTargetColor[color],
                    simParameters.baseValues.cellColorTransitionDuration[color]);
                ImGui::PopID();
            }
            ImGui::TreePop();
        }

        /**
         * Danger zone
         */
        if (ImGui::TreeNodeEx("Danger zone", flags)) {
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Unlimited energy for constructors")
                    .textWidth(RightColumnWidth)
                    .defaultValue(origSimParameters.cellFunctionConstructionUnlimitedEnergy)
                    .tooltip("If activated, all constructor cells receive energy for free to construct offspring. The number of cells can increase rapidly. "
                             "This parameter should only be activated for a short time."),
                simParameters.cellFunctionConstructionUnlimitedEnergy);
            ImGui::TreePop();
        }
    }
    ImGui::EndChild();
    validationAndCorrection(simParameters);
}

void _SimulationParametersWindow::processSpot(
    SimulationParametersSpot& spot,
    SimulationParametersSpot const& origSpot,
    SimulationParameters const& parameters)
{
    if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_DefaultOpen;
        auto worldSize = _simController->getWorldSize();

        /**
         * Colors and location
         */
        if (ImGui::TreeNodeEx("Visualization", flags)) {
            AlienImGui::ColorButtonWithPicker(
                AlienImGui::ColorButtonWithPickerParameters().name("Background color").textWidth(RightColumnWidth).defaultValue(origSpot.color),
                spot.color,
                _backupColor,
                _savedPalette);
            ImGui::TreePop();
        }

        if (ImGui::TreeNodeEx("Location", flags)) {
            if (AlienImGui::Combo(
                AlienImGui::ComboParameters()
                    .name("Shape")
                    .values({"Circular", "Rectangular"})
                    .textWidth(RightColumnWidth)
                    .defaultValue(origSpot.shapeType),
                spot.shapeType)) {
                createDefaultSpotData(spot);
            }
            auto maxRadius = toFloat(std::min(worldSize.x, worldSize.y));
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
                    .format("%.1f"),
                &spot.posY);
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
            ImGui::TreePop();
        }

        /**
         * Flow
         */
        if (ImGui::TreeNodeEx("Force field", flags)) {
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
            ImGui::TreePop();
        }

        /**
         * Physics: Motion
         */
        if (ImGui::TreeNodeEx("Physics: Motion", flags)) {
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
            ImGui::TreePop();
        }

        /**
         * Physics: Thresholds
         */
        if (ImGui::TreeNodeEx("Physics: Thresholds", flags)) {
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
            ImGui::TreePop();
        }

        /**
         * Physics: Binding
         */
        if (ImGui::TreeNodeEx("Physics: Binding", flags)) {
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
            ImGui::TreePop();
        }

        /**
         * Physics: Radiation
         */
        if (ImGui::TreeNodeEx("Physics: Radiation", flags)) {
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
            ImGui::TreePop();
        }

        /**
         * Cell life cycle
         */
        if (ImGui::TreeNodeEx("Cell life cycle", flags)) {
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
            ImGui::TreePop();
        }

       /**
         * Mutation 
         */
        if (ImGui::TreeNodeEx("Cell function: Genome mutation probabilities", flags)) {
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
                    .defaultValue(origSpot.values.cellFunctionConstructorMutationColorProbability)
                    .disabledValue(parameters.baseValues.cellFunctionConstructorMutationColorProbability),
                spot.values.cellFunctionConstructorMutationColorProbability,
                &spot.activatedValues.cellFunctionConstructorMutationColorProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Uniform color")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSpot.values.cellFunctionConstructorMutationUniformColorProbability)
                    .disabledValue(parameters.baseValues.cellFunctionConstructorMutationUniformColorProbability),
                spot.values.cellFunctionConstructorMutationUniformColorProbability,
                &spot.activatedValues.cellFunctionConstructorMutationUniformColorProbability);
            ImGui::TreePop();
        }

        /**
         * Attacker
         */
        if (ImGui::TreeNodeEx("Cell function: Attacker", flags)) {
            ImGui::Checkbox("##foodChainColorMatrix", &spot.activatedValues.cellFunctionAttackerFoodChainColorMatrix);
            ImGui::SameLine();
            ImGui::BeginDisabled(!spot.activatedValues.cellFunctionAttackerFoodChainColorMatrix);
            AlienImGui::InputFloatColorMatrix(
                AlienImGui::InputFloatColorMatrixParameters()
                    .name("Food chain color matrix")
                    .max(1)
                    .textWidth(RightColumnWidth)
                    .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origSpot.values.cellFunctionAttackerFoodChainColorMatrix)),
                spot.values.cellFunctionAttackerFoodChainColorMatrix);
            ImGui::EndDisabled();
            if (!spot.activatedValues.cellFunctionAttackerFoodChainColorMatrix) {
                for (int i = 0; i < MAX_COLORS; ++i) {
                    for (int j = 0; j < MAX_COLORS; ++j) {
                        spot.values.cellFunctionAttackerFoodChainColorMatrix[i][j] = parameters.baseValues.cellFunctionAttackerFoodChainColorMatrix[i][j];
                        spot.values.cellFunctionAttackerFoodChainColorMatrix[i][j] = parameters.baseValues.cellFunctionAttackerFoodChainColorMatrix[i][j];
                    }
                }
            }

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
            ImGui::TreePop();
        }

        /**
         * Cell color transition rules
         */
        if (ImGui::TreeNodeEx("Cell color transition rules", flags)) {
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
            ImGui::TreePop();
            if (!spot.activatedValues.cellColorTransition) {
                for (int color = 0; color < MAX_COLORS; ++color) {
                    spot.values.cellColorTransitionTargetColor[color] = parameters.baseValues.cellColorTransitionTargetColor[color];
                    spot.values.cellColorTransitionDuration[color] = parameters.baseValues.cellColorTransitionDuration[color];
                }
            }
        }
    }
    ImGui::EndChild();
    validationAndCorrection(spot);
}

void _SimulationParametersWindow::onOpenParameters()
{
    GenericFileDialogs::getInstance().showOpenFileDialog(
        "Open simulation parameters", "Simulation parameters (*.parameters){.parameters},.*", _startingPath, [&](std::filesystem::path const& path) {
        auto firstFilename = ifd::FileDialog::Instance().GetResult();
        auto firstFilenameCopy = firstFilename;
        _startingPath = firstFilenameCopy.remove_filename().string();

        SimulationParameters parameters;
        if (!Serializer::deserializeSimulationParametersFromFile(parameters, firstFilename.string())) {
            MessageDialog::getInstance().show("Open simulation parameters", "The selected file could not be opened.");
        } else {
            _simController->setSimulationParameters_async(parameters);
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
        if (!Serializer::serializeSimulationParametersToFile(firstFilename.string(), parameters)) {
            MessageDialog::getInstance().show("Save simulation parameters", "The selected file could not be saved.");
        }
    });
}

void _SimulationParametersWindow::validationAndCorrection(SimulationParameters& parameters) const
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_COLORS; ++j) {
            parameters.baseValues.cellFunctionAttackerFoodChainColorMatrix[i][j] =
                std::max(0.0f, std::min(1.0f, parameters.baseValues.cellFunctionAttackerFoodChainColorMatrix[i][j]));
        }
        parameters.baseValues.radiationAbsorption[i] = std::max(0.0f, std::min(1.0f, parameters.baseValues.radiationAbsorption[i]));
        parameters.cellFunctionConstructorPumpEnergyFactor[i] = std::max(0.0f, std::min(1.0f, parameters.cellFunctionConstructorPumpEnergyFactor[i]));
    }
    parameters.baseValues.cellMaxBindingEnergy = std::max(10.0f, parameters.baseValues.cellMaxBindingEnergy);
    parameters.timestepSize = std::max(0.0f, parameters.timestepSize);
    parameters.cellMaxAgeBalancerInterval = std::max(1000, std::min(1000000, parameters.cellMaxAgeBalancerInterval));
}

void _SimulationParametersWindow::validationAndCorrection(SimulationParametersSpot& spot) const
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_COLORS; ++j) {
            spot.values.cellFunctionAttackerFoodChainColorMatrix[i][j] =
                std::max(0.0f, std::min(1.0f, spot.values.cellFunctionAttackerFoodChainColorMatrix[i][j]));
        }
        spot.values.radiationAbsorption[i] = std::max(0.0f, std::min(1.0f, spot.values.radiationAbsorption[i]));
    }
    spot.values.cellMaxBindingEnergy = std::max(10.0f, spot.values.cellMaxBindingEnergy);
}
