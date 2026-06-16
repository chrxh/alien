#include "MutationRatesDialog.h"

#include <imgui.h>

#include <Base/GlobalSettings.h>

#include "AlienGui.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr HeaderMinRightColumnWidth = 160.0f;
    auto constexpr HeaderMaxLeftColumnWidth = 200.0f;
    auto constexpr HeaderMinColumnWidth = 300.0f;

    void processConnectionMutationRate(std::string const& name, std::string const& id, ConnectionMutationDesc& mutation, float rightColumnWidth)
    {
        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name(name).rank(AlienGui::TreeNodeRank::Default))) {
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Node probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._nodeProbability);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().name("Value change sigma").id(id).min(0.0f).max(1.0f).logarithmic(true).format("%.3f").textWidth(rightColumnWidth),
                &mutation._valueChangeSigma);
        }
        AlienGui::EndTreeNode();
    }

    void processNeuronMutationRate(std::string const& name, std::string const& id, NeuronMutationDesc& mutation, float rightColumnWidth)
    {
        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name(name).rank(AlienGui::TreeNodeRank::Default))) {
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Node probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._nodeProbability);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().name("Weight change sigma").id(id).min(0.0f).max(2.0f).logarithmic(true).format("%.2f").textWidth(rightColumnWidth),
                &mutation._weightChangeSigma);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().name("Bias change sigma").id(id).min(0.0f).max(2.0f).logarithmic(true).format("%.3f").textWidth(rightColumnWidth),
                &mutation._biasChangeSigma);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("ActFn change probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._actfnChangeProbability);
        }
        AlienGui::EndTreeNode();
    }

    void processCellTypePropertiesMutationRate(std::string const& name, std::string const& id, CellTypePropertiesMutationDesc& mutation, float rightColumnWidth)
    {
        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name(name).rank(AlienGui::TreeNodeRank::Default))) {
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Node probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._nodeProbability);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().name("Value change sigma").id(id).min(0.0f).max(1.0f).logarithmic(true).format("%.3f").textWidth(rightColumnWidth),
                &mutation._valueChangeSigma);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Enum change probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._enumChangeProbability);
        }
        AlienGui::EndTreeNode();
    }

    void processCellTypeModeMutationRate(std::string const& name, std::string const& id, CellTypeModeMutationDesc& mutation, float rightColumnWidth)
    {
        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name(name).rank(AlienGui::TreeNodeRank::Default))) {
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Node probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._nodeProbability);
        }
        AlienGui::EndTreeNode();
    }

    void processCellTypeMutationRate(std::string const& name, std::string const& id, CellTypeMutationDesc& mutation, float rightColumnWidth)
    {
        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name(name).rank(AlienGui::TreeNodeRank::Default))) {
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Node probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._nodeProbability);
        }
        AlienGui::EndTreeNode();
    }

    void processVoidMutationRate(std::string const& name, std::string const& id, VoidMutationDesc& mutation, float rightColumnWidth)
    {
        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name(name).rank(AlienGui::TreeNodeRank::Default))) {
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Node probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._nodeProbability);
        }
        AlienGui::EndTreeNode();
    }

    template <typename MutationDesc>
    void processGeneProbabilityMutationRate(std::string const& name, std::string const& id, MutationDesc& mutation, float rightColumnWidth)
    {
        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name(name).rank(AlienGui::TreeNodeRank::Default))) {
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Gene probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._geneProbability);
        }
        AlienGui::EndTreeNode();
    }

    void processConstructorMutationRate(std::string const& name, std::string const& id, ConstructorMutationDesc& mutation, float rightColumnWidth)
    {
        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name(name).rank(AlienGui::TreeNodeRank::Default))) {
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Node probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._nodeProbability);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().name("Value change sigma").id(id).min(0.0f).max(1.0f).logarithmic(true).format("%.3f").textWidth(rightColumnWidth),
                &mutation._valueChangeSigma);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Enum change probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._enumChangeProbability);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Constructor toggle probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._constructorToggleProbability);
        }
        AlienGui::EndTreeNode();
    }

    template <typename Func>
    void processConcreteMutationRates(Func&& processMutationRates)
    {
        AlienGui::DynamicTableLayout table(HeaderMinColumnWidth);
        if (table.begin()) {
            processMutationRates(table);
            table.end();
        }
    }
}

MutationRatesDialog::MutationRatesDialog()
    : AlienDialog("Mutation rates", {800.0f, 400.0f})
{}

void MutationRatesDialog::initIntern() {}

void MutationRatesDialog::shutdownIntern() {}

void MutationRatesDialog::loadSettings(MutationRatesDesc& mutationRates, std::string const& settingsPrefix) const
{
    auto& settings = GlobalSettings::get();

    for (auto i = 0; i < 2; ++i) {
        auto const indexSuffix = i == 0 ? "1" : "2";

        mutationRates._connectionMutations[i]._nodeProbability =
            settings.getValue(settingsPrefix + "connection mutation " + indexSuffix + ".node probability", mutationRates._connectionMutations[i]._nodeProbability);
        mutationRates._connectionMutations[i]._valueChangeSigma =
            settings.getValue(settingsPrefix + "connection mutation " + indexSuffix + ".sigma", mutationRates._connectionMutations[i]._valueChangeSigma);

        mutationRates._neuronMutations[i]._nodeProbability =
            settings.getValue(settingsPrefix + "neuron mutation " + indexSuffix + ".node probability", mutationRates._neuronMutations[i]._nodeProbability);
        mutationRates._neuronMutations[i]._weightChangeSigma =
            settings.getValue(settingsPrefix + "neuron mutation " + indexSuffix + ".weight sigma", mutationRates._neuronMutations[i]._weightChangeSigma);
        mutationRates._neuronMutations[i]._biasChangeSigma =
            settings.getValue(settingsPrefix + "neuron mutation " + indexSuffix + ".bias sigma", mutationRates._neuronMutations[i]._biasChangeSigma);
        mutationRates._neuronMutations[i]._actfnChangeProbability = settings.getValue(
            settingsPrefix + "neuron mutation " + indexSuffix + ".activation function probability",
            mutationRates._neuronMutations[i]._actfnChangeProbability);
    }

    for (auto i = 0; i < 2; ++i) {
        auto const indexSuffix = i == 0 ? "" : " 2";

        mutationRates._cellTypePropertiesMutations[i]._nodeProbability = settings.getValue(
            settingsPrefix + "cell type property mutation" + indexSuffix + ".node probability", mutationRates._cellTypePropertiesMutations[i]._nodeProbability);
        mutationRates._cellTypePropertiesMutations[i]._valueChangeSigma =
            settings.getValue(settingsPrefix + "cell type property mutation" + indexSuffix + ".sigma", mutationRates._cellTypePropertiesMutations[i]._valueChangeSigma);
        mutationRates._cellTypePropertiesMutations[i]._enumChangeProbability = settings.getValue(
            settingsPrefix + "cell type property mutation" + indexSuffix + ".discrete change probability", mutationRates._cellTypePropertiesMutations[i]._enumChangeProbability);
    }

    mutationRates._cellTypeModeMutation._nodeProbability =
        settings.getValue(settingsPrefix + "cell type mode mutation.node probability", mutationRates._cellTypeModeMutation._nodeProbability);
    mutationRates._cellTypeMutation._nodeProbability =
        settings.getValue(settingsPrefix + "cell type mutation.node probability", mutationRates._cellTypeMutation._nodeProbability);
    mutationRates._voidMutation._nodeProbability =
        settings.getValue(settingsPrefix + "void mutation.node probability", mutationRates._voidMutation._nodeProbability);
    mutationRates._appendNodeMutation._geneProbability =
        settings.getValue(settingsPrefix + "append node mutation.gene probability", mutationRates._appendNodeMutation._geneProbability);
    mutationRates._addNodeMutation._geneProbability =
        settings.getValue(settingsPrefix + "add node mutation.gene probability", mutationRates._addNodeMutation._geneProbability);
    mutationRates._trimNodeMutation._geneProbability =
        settings.getValue(settingsPrefix + "trim node mutation.gene probability", mutationRates._trimNodeMutation._geneProbability);
    mutationRates._deleteNodeMutation._geneProbability =
        settings.getValue(settingsPrefix + "delete node mutation.gene probability", mutationRates._deleteNodeMutation._geneProbability);

    for (auto i = 0; i < 2; ++i) {
        auto const indexSuffix = i == 0 ? "1" : "2";

        mutationRates._constructorMutations[i]._nodeProbability = settings.getValue(
            settingsPrefix + "constructor mutation " + indexSuffix + ".node probability", mutationRates._constructorMutations[i]._nodeProbability);
        mutationRates._constructorMutations[i]._valueChangeSigma =
            settings.getValue(settingsPrefix + "constructor mutation " + indexSuffix + ".sigma", mutationRates._constructorMutations[i]._valueChangeSigma);
        mutationRates._constructorMutations[i]._enumChangeProbability = settings.getValue(
            settingsPrefix + "constructor mutation " + indexSuffix + ".discrete change probability", mutationRates._constructorMutations[i]._enumChangeProbability);
        mutationRates._constructorMutations[i]._constructorToggleProbability = settings.getValue(
            settingsPrefix + "constructor mutation " + indexSuffix + ".exist constructor probability",
            mutationRates._constructorMutations[i]._constructorToggleProbability);
    }
}

void MutationRatesDialog::saveSettings(MutationRatesDesc const& mutationRates, std::string const& settingsPrefix) const
{
    auto& settings = GlobalSettings::get();

    for (auto i = 0; i < 2; ++i) {
        auto const indexSuffix = i == 0 ? "1" : "2";

        settings.setValue(settingsPrefix + "connection mutation " + indexSuffix + ".node probability", mutationRates._connectionMutations[i]._nodeProbability);
        settings.setValue(settingsPrefix + "connection mutation " + indexSuffix + ".sigma", mutationRates._connectionMutations[i]._valueChangeSigma);

        settings.setValue(settingsPrefix + "neuron mutation " + indexSuffix + ".node probability", mutationRates._neuronMutations[i]._nodeProbability);
        settings.setValue(settingsPrefix + "neuron mutation " + indexSuffix + ".weight sigma", mutationRates._neuronMutations[i]._weightChangeSigma);
        settings.setValue(settingsPrefix + "neuron mutation " + indexSuffix + ".bias sigma", mutationRates._neuronMutations[i]._biasChangeSigma);
        settings.setValue(
            settingsPrefix + "neuron mutation " + indexSuffix + ".activation function probability",
            mutationRates._neuronMutations[i]._actfnChangeProbability);
    }

    for (auto i = 0; i < 2; ++i) {
        auto const indexSuffix = i == 0 ? "1" : "2";

        settings.setValue(
            settingsPrefix + "cell type property mutation " + indexSuffix + ".node probability", mutationRates._cellTypePropertiesMutations[i]._nodeProbability);
        settings.setValue(settingsPrefix + "cell type property mutation " + indexSuffix + ".sigma", mutationRates._cellTypePropertiesMutations[i]._valueChangeSigma);
        settings.setValue(
            settingsPrefix + "cell type property mutation " + indexSuffix + ".discrete change probability", mutationRates._cellTypePropertiesMutations[i]._enumChangeProbability);
    }

    settings.setValue(settingsPrefix + "cell type mode mutation.node probability", mutationRates._cellTypeModeMutation._nodeProbability);
    settings.setValue(settingsPrefix + "cell type mutation.node probability", mutationRates._cellTypeMutation._nodeProbability);
    settings.setValue(settingsPrefix + "void mutation.node probability", mutationRates._voidMutation._nodeProbability);
    settings.setValue(settingsPrefix + "append node mutation.gene probability", mutationRates._appendNodeMutation._geneProbability);
    settings.setValue(settingsPrefix + "add node mutation.gene probability", mutationRates._addNodeMutation._geneProbability);
    settings.setValue(settingsPrefix + "trim node mutation.gene probability", mutationRates._trimNodeMutation._geneProbability);
    settings.setValue(settingsPrefix + "delete node mutation.gene probability", mutationRates._deleteNodeMutation._geneProbability);

    for (auto i = 0; i < 2; ++i) {
        auto const indexSuffix = i == 0 ? "1" : "2";

        settings.setValue(settingsPrefix + "constructor mutation " + indexSuffix + ".node probability", mutationRates._constructorMutations[i]._nodeProbability);
        settings.setValue(settingsPrefix + "constructor mutation " + indexSuffix + ".sigma", mutationRates._constructorMutations[i]._valueChangeSigma);
        settings.setValue(settingsPrefix + "constructor mutation " + indexSuffix + ".discrete change probability", mutationRates._constructorMutations[i]._enumChangeProbability);
        settings.setValue(
            settingsPrefix + "constructor mutation " + indexSuffix + ".exist constructor probability",
            mutationRates._constructorMutations[i]._constructorToggleProbability);
    }
}

void MutationRatesDialog::processIntern()
{
    // Use a child window with scrolling for the content, reserving space for buttons
    auto buttonAreaHeight = scale(50.0f);
    if (ImGui::BeginChild("MutationRateContent", ImVec2(0, -buttonAreaHeight), false)) {
        auto rightColumnWidth = std::max(HeaderMinRightColumnWidth, scaleInverse(ImGui::GetContentRegionAvail().x - scale(HeaderMaxLeftColumnWidth)));

        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Connection weight mutations").rank(AlienGui::TreeNodeRank::High))) {
            processConcreteMutationRates([&](AlienGui::DynamicTableLayout& table) {
                processConnectionMutationRate("Mutation rate 1", "CMR1", _mutation._connectionMutations[0], rightColumnWidth);
                table.next();
                processConnectionMutationRate("Mutation rate 2", "CMR2", _mutation._connectionMutations[1], rightColumnWidth);
                table.next();
            });
        }
        AlienGui::EndTreeNode();

        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Neuron mutations").rank(AlienGui::TreeNodeRank::High))) {
            processConcreteMutationRates([&](AlienGui::DynamicTableLayout& table) {
                processNeuronMutationRate("Mutation rate 1", "NMR1", _mutation._neuronMutations[0], rightColumnWidth);
                table.next();
                processNeuronMutationRate("Mutation rate 2", "NMR2", _mutation._neuronMutations[1], rightColumnWidth);
                table.next();
            });
        }
        AlienGui::EndTreeNode();

        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Cell type property mutations").rank(AlienGui::TreeNodeRank::High))) {
            processConcreteMutationRates([&](AlienGui::DynamicTableLayout& table) {
                processCellTypePropertiesMutationRate("Mutation rate 1", "CTPM1", _mutation._cellTypePropertiesMutations[0], rightColumnWidth);
                table.next();
                processCellTypePropertiesMutationRate("Mutation rate 2", "CTPM2", _mutation._cellTypePropertiesMutations[1], rightColumnWidth);
                table.next();
            });
        }
        AlienGui::EndTreeNode();

        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Cell type mode mutations").rank(AlienGui::TreeNodeRank::High))) {
            processConcreteMutationRates([&](AlienGui::DynamicTableLayout& table) {
                processCellTypeModeMutationRate("Mutation rate", "CTMM", _mutation._cellTypeModeMutation, rightColumnWidth);
                table.next();
            });
        }
        AlienGui::EndTreeNode();

        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Cell type mutations").rank(AlienGui::TreeNodeRank::High))) {
            processConcreteMutationRates([&](AlienGui::DynamicTableLayout& table) {
                processCellTypeMutationRate("Mutation rate", "CTM", _mutation._cellTypeMutation, rightColumnWidth);
                table.next();
            });
        }
        AlienGui::EndTreeNode();

        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Void mutations").rank(AlienGui::TreeNodeRank::High))) {
            processConcreteMutationRates([&](AlienGui::DynamicTableLayout& table) {
                processVoidMutationRate("Mutation rate", "VM", _mutation._voidMutation, rightColumnWidth);
                table.next();
            });
        }
        AlienGui::EndTreeNode();

        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Append node mutations").rank(AlienGui::TreeNodeRank::High))) {
            processConcreteMutationRates([&](AlienGui::DynamicTableLayout& table) {
                processGeneProbabilityMutationRate("Mutation rate", "APNM", _mutation._appendNodeMutation, rightColumnWidth);
                table.next();
            });
        }
        AlienGui::EndTreeNode();

        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Add node mutations").rank(AlienGui::TreeNodeRank::High))) {
            processConcreteMutationRates([&](AlienGui::DynamicTableLayout& table) {
                processGeneProbabilityMutationRate("Mutation rate", "ADNM", _mutation._addNodeMutation, rightColumnWidth);
                table.next();
            });
        }
        AlienGui::EndTreeNode();

        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Trim node mutations").rank(AlienGui::TreeNodeRank::High))) {
            processConcreteMutationRates([&](AlienGui::DynamicTableLayout& table) {
                processGeneProbabilityMutationRate("Mutation rate", "TRNM", _mutation._trimNodeMutation, rightColumnWidth);
                table.next();
            });
        }
        AlienGui::EndTreeNode();

        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Delete node mutations").rank(AlienGui::TreeNodeRank::High))) {
            processConcreteMutationRates([&](AlienGui::DynamicTableLayout& table) {
                processGeneProbabilityMutationRate("Mutation rate", "DLNM", _mutation._deleteNodeMutation, rightColumnWidth);
                table.next();
            });
        }
        AlienGui::EndTreeNode();

        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Constructor mutations").rank(AlienGui::TreeNodeRank::High))) {
            processConcreteMutationRates([&](AlienGui::DynamicTableLayout& table) {
                processConstructorMutationRate("Mutation rate 1", "COM1", _mutation._constructorMutations[0], rightColumnWidth);
                table.next();
                processConstructorMutationRate("Mutation rate 2", "COM2", _mutation._constructorMutations[1], rightColumnWidth);
                table.next();
            });
        }
        AlienGui::EndTreeNode();
    }
    ImGui::EndChild();

    AlienGui::Separator();

    if (AlienGui::Button("Adopt")) {
        ImGui::CloseCurrentPopup();
        onAdopt();
        close();
    }
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienGui::Button("Cancel")) {
        ImGui::CloseCurrentPopup();
        close();
    }
}

void MutationRatesDialog::open(MutationRatesDesc const& mutationRates, std::function<void(MutationRatesDesc const&)> const& onAdoptCallback)
{
    _mutation = mutationRates;
    _onAdoptCallback = onAdoptCallback;
    AlienDialog::open();
}

void MutationRatesDialog::openNested(MutationRatesDesc const& mutationRates, std::function<void(MutationRatesDesc const&)> const& onAdoptCallback)
{
    _mutation = mutationRates;
    _onAdoptCallback = onAdoptCallback;
    AlienDialog::openNested();
}

void MutationRatesDialog::openIntern() {}

void MutationRatesDialog::onAdopt()
{
    if (_onAdoptCallback) {
        _onAdoptCallback(_mutation);
    }
}
