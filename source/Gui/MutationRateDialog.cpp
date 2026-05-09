#include "MutationRateDialog.h"

#include <imgui.h>

#include "AlienGui.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr HeaderMinRightColumnWidth = 160.0f;
    auto constexpr HeaderMaxLeftColumnWidth = 200.0f;
    auto constexpr HeaderMinColumnWidth = 300.0f;
}

MutationRateDialog::MutationRateDialog()
    : AlienDialog("Set mutation rates")
{}

void MutationRateDialog::initIntern() {}

void MutationRateDialog::shutdownIntern() {}

void MutationRateDialog::processIntern()
{
    AlienGui::DynamicTableLayout table(HeaderMinColumnWidth);
    if (table.begin()) {
        auto rightColumnWidth = std::max(HeaderMinRightColumnWidth, scaleInverse(ImGui::GetContentRegionAvail().x - scale(HeaderMaxLeftColumnWidth)));

        AlienGui::Group(AlienGui::GroupParameters().text("Connection weight mutation rate 1"));
        AlienGui::SliderFloat(
            AlienGui::SliderFloatParameters().name("Probability").id("CMR1").min(0.0f).max(1.0f).logarithmic(true).format("%.5f").textWidth(rightColumnWidth),
            &_connectionMutationRate1._probability);
        AlienGui::SliderFloat(
            AlienGui::SliderFloatParameters().name("Sigma").id("CMR1").min(0.0f).max(1.0f).logarithmic(true).format("%.3f").textWidth(rightColumnWidth),
            &_connectionMutationRate1._sigma);

        AlienGui::Group(AlienGui::GroupParameters().text("Connection weight mutation rate 2"));
        AlienGui::SliderFloat(
            AlienGui::SliderFloatParameters().name("Probability").id("CMR2").min(0.0f).max(1.0f).logarithmic(true).format("%.5f").textWidth(rightColumnWidth),
            &_connectionMutationRate2._probability);
        AlienGui::SliderFloat(
            AlienGui::SliderFloatParameters().name("Sigma").id("CMR2").min(0.0f).max(1.0f).logarithmic(true).format("%.3f").textWidth(rightColumnWidth),
            &_connectionMutationRate2._sigma);
        table.next();

        AlienGui::Group(AlienGui::GroupParameters().text("Neuron weight mutation rate 1"));
        AlienGui::SliderFloat(
            AlienGui::SliderFloatParameters().name("Probability").id("NMR1").min(0.0f).max(1.0f).logarithmic(true).format("%.5f").textWidth(rightColumnWidth),
            &_neuronMutation1._probability);
        AlienGui::SliderFloat(
            AlienGui::SliderFloatParameters().name("Weight sigma").id("NMR1").min(0.0f).max(2.0f).logarithmic(true).format("%.2f").textWidth(rightColumnWidth),
            &_neuronMutation1._weightSigma);
        AlienGui::SliderFloat(
            AlienGui::SliderFloatParameters().name("Bias sigma").id("NMR1").min(0.0f).max(2.0f).logarithmic(true).format("%.3f").textWidth(rightColumnWidth),
            &_neuronMutation1._biasSigma);
        AlienGui::SliderFloat(
            AlienGui::SliderFloatParameters()
                .name("ActFn probability")
                .id("NMR1")
                .min(0.0f)
                .max(1.0f)
                .logarithmic(true)
                .format("%.5f")
                .textWidth(rightColumnWidth),
            &_neuronMutation1._activationFunctionProbability);
        table.next();

        AlienGui::Group(AlienGui::GroupParameters().text("Neuron weight mutation rate 2"));
        AlienGui::SliderFloat(
            AlienGui::SliderFloatParameters().name("Probability").id("NMR2").min(0.0f).max(1.0f).logarithmic(true).format("%.5f").textWidth(rightColumnWidth),
            &_neuronMutation2._probability);
        AlienGui::SliderFloat(
            AlienGui::SliderFloatParameters().name("Weight sigma").id("NMR2").min(0.0f).max(2.0f).logarithmic(true).format("%.2f").textWidth(rightColumnWidth),
            &_neuronMutation2._weightSigma);
        AlienGui::SliderFloat(
            AlienGui::SliderFloatParameters().name("Bias sigma").id("NMR2").min(0.0f).max(2.0f).logarithmic(true).format("%.3f").textWidth(rightColumnWidth),
            &_neuronMutation2._biasSigma);
        AlienGui::SliderFloat(
            AlienGui::SliderFloatParameters()
                .name("ActFn probability")
                .id("NMR2")
                .min(0.0f)
                .max(1.0f)
                .logarithmic(true)
                .format("%.5f")
                .textWidth(rightColumnWidth),
            &_neuronMutation2._activationFunctionProbability);
        table.next();

        AlienGui::Group(AlienGui::GroupParameters().text("Lineage mutation rate"));
        AlienGui::SliderFloat(
            AlienGui::SliderFloatParameters().name("Probability").min(0.0f).max(1.0f).logarithmic(true).format("%.5f").textWidth(rightColumnWidth),
            &_lineageMutationProbability);
        table.next();

        table.end();
    }

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
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

void MutationRateDialog::openIntern()
{
    // Initialize with default values or current values when the dialog is opened
    // For now, the dialog is not connected, so we just initialize with defaults
    _connectionMutationRate1 = ConnectionMutationDesc();
    _connectionMutationRate2 = ConnectionMutationDesc();
    _neuronMutation1 = NeuronMutationDesc();
    _neuronMutation2 = NeuronMutationDesc();
    _lineageMutationProbability = 0.0f;
}

void MutationRateDialog::onAdopt()
{
    // This method will be connected later to actually apply the mutation rates
    // For now, it's just a placeholder
}
