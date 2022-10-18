#include "ImageToPatternDialog.h"

#include <boost/range/adaptor/indexed.hpp>
#include <stb_image.h>
#include <imgui.h>
#include <ImFileDialog.h>

#include "Base/Definitions.h"
#include "Base/NumberGenerator.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/Colors.h"
#include "AlienImGui.h"
#include "Viewport.h"
#include "GenericOpenFileDialog.h"
#include "GlobalSettings.h"


_ImageToPatternDialog::_ImageToPatternDialog(Viewport const& viewport, SimulationController const& simController)
    : _viewport(viewport)
    , _simController(simController)
{
    auto path = std::filesystem::current_path();
    if (path.has_parent_path()) {
        path = path.parent_path();
    }
    _startingPath = GlobalSettings::getInstance().getStringState("dialogs.open image.starting path", path.string());
}

_ImageToPatternDialog::~_ImageToPatternDialog()
{
    GlobalSettings::getInstance().setStringState("dialogs.open image.starting path", _startingPath);
}

namespace
{
    void getMatchedCellColor(ImColor const& color, int& matchedCellColor, float& matchedCellIntensity)
    {
        using Color = std::array<float,3>;
        static std::vector<Color> cellColors;
        auto toHsv = [](uint32_t color) {
            float h, s, v;
            AlienImGui::convertRGBtoHSV(color, h, s, v);
            return Color{h, s, v}; 
        };
        if (cellColors.empty()) {
            cellColors.emplace_back(toHsv(Const::IndividualCellColor1));
            cellColors.emplace_back(toHsv(Const::IndividualCellColor2));
            cellColors.emplace_back(toHsv(Const::IndividualCellColor3));
            cellColors.emplace_back(toHsv(Const::IndividualCellColor4));
            cellColors.emplace_back(toHsv(Const::IndividualCellColor5));
            cellColors.emplace_back(toHsv(Const::IndividualCellColor6));
            cellColors.emplace_back(toHsv(Const::IndividualCellColor7));
        }

        std::optional<int> bestMatchIndex;
        std::optional<float> bestMatchDistance;
        auto colorHsv = toHsv((ImU32)color);
        for (auto const& [index, cellColor] : cellColors | boost::adaptors::indexed(0)) {
            auto distance = colorHsv[0] - cellColor[0];
            if (distance > 0.5f) {
                distance -= 1.0f;
            }
            if (distance < -0.5f) {
                distance += 1.0f;
            }
            distance = std::abs(distance) * colorHsv[1] + std::abs(colorHsv[1] - cellColor[1]);
            if (!bestMatchDistance || *bestMatchDistance > distance) {
                bestMatchIndex = toInt(index);
                bestMatchDistance = distance;
            }
        }
        matchedCellColor = *bestMatchIndex;
        matchedCellIntensity = colorHsv[2];
    }
}

void _ImageToPatternDialog::show()
{
    GenericOpenFileDialog::getInstance().show(
        "Open image", "Image (*.png){.png},.*", _startingPath, [&](std::filesystem::path const& path) {

        auto firstFilename = ifd::FileDialog::Instance().GetResult();
        auto firstFilenameCopy = firstFilename;
        _startingPath = firstFilenameCopy.remove_filename().string();

        int width, height, nrChannels;
        unsigned char* dataImage = stbi_load(firstFilename.string().c_str(), &width, &height, &nrChannels, 0);

        auto parameters = _simController->getSimulationParameters();
        auto maxConnections = parameters.cellMaxBonds;

        DataDescription dataDesc;
        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < height; ++y) {
                auto address = (x + y * width) * nrChannels;
                int r = dataImage[address + 2];
                int g = dataImage[address + 1];
                int b = dataImage[address];
                auto xOffset = y % 2 == 0 ? 0.0f : 0.5f;
                if (r > 20 || g > 20 || b > 20) {
                    int matchedCellColor;
                    float matchedCellIntensity;
                    getMatchedCellColor(ImColor(r, g, b, 255), matchedCellColor, matchedCellIntensity);
                    dataDesc.addCell(CellDescription()
                                         .setId(NumberGenerator::getInstance().getId())
                                         .setEnergy(matchedCellIntensity * 200)
                                         .setPos({toFloat(x) + xOffset, toFloat(y)})
                                         .setMaxConnections(maxConnections)
                                         .setColor(matchedCellColor)
                                         .setBarrier(false));
                }
            }
        }

        DescriptionHelper::reconnectCells(dataDesc, 1 * 1.5f);
        dataDesc.setCenter(_viewport->getCenterInWorldPos());

        _simController->addAndSelectSimulationData(dataDesc);
        //TODO: update pattern editor
    });
}
