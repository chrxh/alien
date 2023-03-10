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

namespace
{
    auto constexpr RightColumnWidth = 260.0f;

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

_SimulationParametersWindow::_SimulationParametersWindow(SimulationController const& simController, RadiationSourcesWindow const& radiationSourcesWindow)
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

        if (ImGui::BeginTabBar("##Flow", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown)) {

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
    spot.shapeType = ShapeType_Circular;
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
    if (spot.shapeType == ShapeType_Circular) {
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
        if (ImGui::TreeNodeEx("Colors", flags)) {
            AlienImGui::ColorButtonWithPicker(
                AlienImGui::ColorButtonWithPickerParameters().name("Background color").textWidth(RightColumnWidth).defaultValue(origSimParameters.backgroundColor),
                simParameters.backgroundColor,
                _backupColor,
                _savedPalette);
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
                    .tooltip(std::string("Time duration calculated in a single step. Smaller values increase the accuracy "
                                         "of the simulation.")),
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
                        .values({"Fluid dynamics", "Collision-based"}),
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
                        .tooltip(std::string("")),
                    &simParameters.motionData.fluidMotion.smoothingLength);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Pressure")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(0.3f)
                        .defaultValue(&origSimParameters.motionData.fluidMotion.pressureStrength)
                        .tooltip(std::string("")),
                    &simParameters.motionData.fluidMotion.pressureStrength);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Viscosity")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(0.3f)
                        .defaultValue(&origSimParameters.motionData.fluidMotion.viscosityStrength)
                        .tooltip(std::string("")),
                    &simParameters.motionData.fluidMotion.viscosityStrength);
            } else {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Repulsion strength")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(0.3f)
                        .defaultValue(&origSimParameters.motionData.collisionMotion.cellRepulsionStrength)
                        .tooltip(std::string("The strength of the repulsive forces, between two cells that do not connect.")),
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
                    .tooltip(std::string("Specifies how much the movements are slowed down per time step.")),
                &simParameters.baseValues.friction);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Rigidity")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(1.0f)
                    .format("%.2f")
                    .defaultValue(&origSimParameters.baseValues.rigidity)
                    .tooltip(std::string("Controls the rigidity of connected cells.\nA higher value will cause connected cells to move more uniformly.")),
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
                    .tooltip(std::string("Minimum distance between two cells without them annihilating each other.")),
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
                    .tooltip(std::string("Minimum velocity of two colliding cells so that a connection can be established.")),
                &simParameters.baseValues.cellFusionVelocity);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Maximum energy")
                    .textWidth(RightColumnWidth)
                    .min(50.0f)
                    .max(1000000.0f)
                    .logarithmic(true)
                    .format("%.0f")
                    .defaultValue(&origSimParameters.baseValues.cellMaxBindingEnergy)
                    .tooltip(std::string("Maximum energy of a cell at which they can maintain a connection.")),
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
                                       .tooltip("")
                                       .showDisabledRevertButton(true))) {
                _radiationSourcesWindow->setOn(true);
            }

            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Absorption factor")
                    .tooltip("")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(1.0)
                    .format("%.2f")
                    .defaultValue(origSimParameters.radiationAbsorption),
                simParameters.radiationAbsorption);

            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Cell age radiation strength")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(0.01f)
                    .logarithmic(true)
                    .format("%.6f")
                    .defaultValue(origSimParameters.baseValues.radiationCellAgeStrength)
                    .tooltip(std::string("Indicates how energetic the emitted particles of cells are.")),
                simParameters.baseValues.radiationCellAgeStrength);
            AlienImGui::SliderInt(
                AlienImGui::SliderIntParameters()
                    .name("Minimum age")
                    .tooltip("")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(1000000)
                    .logarithmic(true)
                    .defaultValue(origSimParameters.radiationMinCellAge),
                simParameters.radiationMinCellAge);

            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("High energy radiation")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(0.01f)
                    .logarithmic(true)
                    .format("%.6f")
                    .defaultValue(origSimParameters.highRadiationFactor)
                    .tooltip(""),
                simParameters.highRadiationFactor);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("High energy radiation threshold")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(100000)
                    .logarithmic(true)
                    .defaultValue(origSimParameters.highRadiationMinCellEnergy)
                    .tooltip(""),
                simParameters.highRadiationMinCellEnergy);

            ImGui::TreePop();
        }

        /**
         * Physics: Particle transformation
         */
        ImGui::PushID("Transformation");
        if (ImGui::TreeNodeEx("Physics: Particle transformation", flags)) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Minimum energy")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(10.0f)
                    .max(200.0f)
                    .defaultValue(origSimParameters.baseValues.cellMinEnergy)
                    .tooltip(std::string("Minimum energy a cell needs to exist.")),
                simParameters.baseValues.cellMinEnergy);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Normal energy")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(10.0f)
                    .max(200.0f)
                    .defaultValue(origSimParameters.cellNormalEnergy),
                simParameters.cellNormalEnergy);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters().name("Cell cluster decay").textWidth(RightColumnWidth).defaultValue(origSimParameters.clusterDecay),
                simParameters.clusterDecay);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Cell cluster decay probability")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0.0)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .defaultValue(origSimParameters.clusterDecayProb),
                simParameters.clusterDecayProb);
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
                    .format("%.6f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSimParameters.baseValues.cellFunctionConstructorMutationNeuronDataProbability),
                simParameters.baseValues.cellFunctionConstructorMutationNeuronDataProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Cell properties")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.6f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSimParameters.baseValues.cellFunctionConstructorMutationDataProbability),
                simParameters.baseValues.cellFunctionConstructorMutationDataProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Cell function type")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.6f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSimParameters.baseValues.cellFunctionConstructorMutationCellFunctionProbability),
                simParameters.baseValues.cellFunctionConstructorMutationCellFunctionProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Insertion")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.6f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSimParameters.baseValues.cellFunctionConstructorMutationInsertionProbability),
                simParameters.baseValues.cellFunctionConstructorMutationInsertionProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Deletion")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.6f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSimParameters.baseValues.cellFunctionConstructorMutationDeletionProbability),
                simParameters.baseValues.cellFunctionConstructorMutationDeletionProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Translation")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.6f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSimParameters.baseValues.cellFunctionConstructorMutationTranslationProbability),
                simParameters.baseValues.cellFunctionConstructorMutationTranslationProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Duplication")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.6f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSimParameters.baseValues.cellFunctionConstructorMutationDuplicationProbability),
                simParameters.baseValues.cellFunctionConstructorMutationDuplicationProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Color")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.6f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSimParameters.baseValues.cellFunctionConstructorMutationColorProbability),
                simParameters.baseValues.cellFunctionConstructorMutationColorProbability);
            AlienImGui::CheckboxColorMatrix(
                AlienImGui::CheckboxColorMatrixParameters()
                    .name("Color transitions")
                    .textWidth(RightColumnWidth)
                    .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origSimParameters.cellFunctionConstructorMutationColorTransitions)),
                simParameters.cellFunctionConstructorMutationColorTransitions);
            auto preserveSelfReplication = !simParameters.cellFunctionConstructorMutationSelfReplication;
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Preserve self-replication")
                    .textWidth(RightColumnWidth)
                    .defaultValue(!origSimParameters.cellFunctionConstructorMutationSelfReplication),
                preserveSelfReplication);
            simParameters.cellFunctionConstructorMutationSelfReplication = !preserveSelfReplication;
            ImGui::TreePop();
        }

        /**
         * Constructor
         */
        if (ImGui::TreeNodeEx("Cell function: Constructor", flags)) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Offspring distance")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0.1f)
                    .max(3.0f)
                    .defaultValue(origSimParameters.cellFunctionConstructorOffspringDistance),
                simParameters.cellFunctionConstructorOffspringDistance);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Connection distance")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0.1f)
                    .max(3.0f)
                    .defaultValue(origSimParameters.cellFunctionConstructorConnectingCellMaxDistance),
                simParameters.cellFunctionConstructorConnectingCellMaxDistance);
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
                    .defaultValue(origSimParameters.cellFunctionInjectorRadius),
                simParameters.cellFunctionInjectorRadius);
            AlienImGui::InputIntColorMatrix(
                AlienImGui::InputIntColorMatrixParameters()
                    .name("Injection time")
                    .logarithmic(true)
                    .max(100000)
                    .textWidth(RightColumnWidth)
                    .tooltip("")
                    .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origSimParameters.cellFunctionInjectorDurationColorMatrix)),
                simParameters.cellFunctionInjectorDurationColorMatrix);
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
                        "This matrix can be used to determine how well one cell can attack another cell. The color of the attacking cell is shown in the "
                        "header "
                        "column while the color of the attacked cell is shown in the header row. A value of 0 means that the attacked cell cannot be digested, "
                        "i.e. no energy can be obtained. A value of 1 means that the maximum energy can be obtained in the digestion process.\n\nExample: If a "
                        "0 "
                        "is "
                        "entered in row 2 (red) and column 3 (green), it means that red cells cannot eat green cells.")
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
                    .tooltip(std::string("Amount of energy lost by an attempted attack of a cell in the form of emitted energy particles.")),
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
                    .name("Velocity penalty")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(5.0f)
                    .defaultValue(origSimParameters.cellFunctionAttackerVelocityPenalty)
                    .tooltip(std::string("")),
                simParameters.cellFunctionAttackerVelocityPenalty);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Geometry penalty")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(5.0f)
                    .defaultValue(origSimParameters.baseValues.cellFunctionAttackerGeometryDeviationExponent)
                    .tooltip(std::string("The larger this value is, the less energy a cell can gain from an attack if the local "
                                         "geometry of the attacked cell does not match the attacking cell.")),
                simParameters.baseValues.cellFunctionAttackerGeometryDeviationExponent);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Connections mismatch penalty")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(1.0f)
                    .defaultValue(origSimParameters.baseValues.cellFunctionAttackerConnectionsMismatchPenalty)
                    .tooltip(std::string("The larger this parameter is, the more difficult it is to digest cells that contain more connections.")),
                simParameters.baseValues.cellFunctionAttackerConnectionsMismatchPenalty);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Attack radius")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(2.5f)
                    .defaultValue(origSimParameters.cellFunctionAttackerRadius),
                simParameters.cellFunctionAttackerRadius);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Energy distribution radius")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(5.0f)
                    .defaultValue(origSimParameters.cellFunctionAttackerEnergyDistributionRadius),
                simParameters.cellFunctionAttackerEnergyDistributionRadius);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Energy distribution Value")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(20.0f)
                    .defaultValue(origSimParameters.cellFunctionAttackerEnergyDistributionValue),
                simParameters.cellFunctionAttackerEnergyDistributionValue);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Same color energy distribution")
                    .textWidth(RightColumnWidth)
                    .defaultValue(origSimParameters.cellFunctionAttackerEnergyDistributionSameColor),
                simParameters.cellFunctionAttackerEnergyDistributionSameColor);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Color inhomogeneity factor")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(1.0f)
                    .defaultValue(origSimParameters.cellFunctionAttackerColorInhomogeneityFactor),
                simParameters.cellFunctionAttackerColorInhomogeneityFactor);
            ImGui::TreePop();
        }
        ImGui::PopID();

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
                    .defaultValue(origSimParameters.cellFunctionDefenderAgainstAttackerStrength),
                simParameters.cellFunctionDefenderAgainstAttackerStrength);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Anti-injector strength")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(1.0f)
                    .max(5.0f)
                    .defaultValue(origSimParameters.cellFunctionDefenderAgainstInjectorStrength),
                simParameters.cellFunctionDefenderAgainstInjectorStrength);
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
                    .defaultValue(origSimParameters.cellFunctionTransmitterEnergyDistributionRadius),
                simParameters.cellFunctionTransmitterEnergyDistributionRadius);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Energy distribution Value")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(20.0f)
                    .defaultValue(origSimParameters.cellFunctionTransmitterEnergyDistributionValue),
                simParameters.cellFunctionTransmitterEnergyDistributionValue);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Same color energy distribution")
                    .textWidth(RightColumnWidth)
                    .defaultValue(origSimParameters.cellFunctionTransmitterEnergyDistributionSameColor),
                simParameters.cellFunctionTransmitterEnergyDistributionSameColor);
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
                    .defaultValue(origSimParameters.cellFunctionMuscleContractionExpansionDelta),
                simParameters.cellFunctionMuscleContractionExpansionDelta);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Forward/backward acceleration")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(0.15f)
                    .logarithmic(true)
                    .defaultValue(origSimParameters.cellFunctionMuscleMovementAcceleration),
                simParameters.cellFunctionMuscleMovementAcceleration);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Bending angle")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(10.0f)
                    .defaultValue(origSimParameters.cellFunctionMuscleBendingAngle),
                simParameters.cellFunctionMuscleBendingAngle);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Bending acceleration")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(0.5f)
                    .defaultValue(origSimParameters.cellFunctionMuscleBendingAcceleration),
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
                    .tooltip(std::string("The maximum radius in which a sensor can detect mass concentrations.")),
                simParameters.cellFunctionSensorRange);
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
                                      .logarithmic(true);
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
                    .defaultValue(origSimParameters.cellFunctionConstructionUnlimitedEnergy),
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
        if (ImGui::TreeNodeEx("Colors and location", flags)) {
            AlienImGui::ColorButtonWithPicker(
                AlienImGui::ColorButtonWithPickerParameters().name("Background color").textWidth(RightColumnWidth).defaultValue(origSpot.color),
                spot.color,
                _backupColor,
                _savedPalette);

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
            if (spot.shapeType == ShapeType_Circular) {
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
            if (spot.shapeType == ShapeType_Rectangular) {
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
                    .name("Binding max energy")
                    .textWidth(RightColumnWidth)
                    .min(50.0f)
                    .max(1000000.0f)
                    .logarithmic(true)
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
                    .name("Radiation strength")
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
         * Physics: Particle transformation
         */
        if (ImGui::TreeNodeEx("Physics: Particle transformation", flags)) {
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
                    .format("%.6f")
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
                    .format("%.6f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSpot.values.cellFunctionConstructorMutationDataProbability)
                    .disabledValue(parameters.baseValues.cellFunctionConstructorMutationDataProbability),
                spot.values.cellFunctionConstructorMutationDataProbability,
                &spot.activatedValues.cellFunctionConstructorMutationDataProbability);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Cell function type")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.1f)
                    .format("%.6f")
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
                    .format("%.6f")
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
                    .format("%.6f")
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
                    .format("%.6f")
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
                    .format("%.6f")
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
                    .format("%.6f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSpot.values.cellFunctionConstructorMutationColorProbability)
                    .disabledValue(parameters.baseValues.cellFunctionConstructorMutationColorProbability),
                spot.values.cellFunctionConstructorMutationColorProbability,
                &spot.activatedValues.cellFunctionConstructorMutationColorProbability);
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
                                      .logarithmic(true);
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
        parameters.radiationAbsorption[i] = std::max(0.0f, std::min(1.0f, parameters.radiationAbsorption[i]));
    }
    parameters.baseValues.cellMaxBindingEnergy = std::max(10.0f, parameters.baseValues.cellMaxBindingEnergy);
}

void _SimulationParametersWindow::validationAndCorrection(SimulationParametersSpot& spot) const
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_COLORS; ++j) {
            spot.values.cellFunctionAttackerFoodChainColorMatrix[i][j] =
                std::max(0.0f, std::min(1.0f, spot.values.cellFunctionAttackerFoodChainColorMatrix[i][j]));
        }
    }
    spot.values.cellMaxBindingEnergy = std::max(10.0f, spot.values.cellMaxBindingEnergy);
}
