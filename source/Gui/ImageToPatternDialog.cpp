#include "ImageToPatternDialog.h"

#include <boost/range/adaptor/indexed.hpp>

#include <stb_image.h>

#include <imgui.h>

#include <Base/Definitions.h>
#include <Base/GlobalSettings.h>

#include <EngineInterface/Colors.h>
#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>
#include <EngineInterface/NumberGenerator.h>
#include <EngineInterface/SimulationFacade.h>

#include "AlienGui.h"
#include "GenericFileDialog.h"
#include "Viewport.h"

#include <ImFileDialog.h>


void ImageToPatternDialog::init()
{

    auto path = std::filesystem::current_path();
    if (path.has_parent_path()) {
        path = path.parent_path();
    }
    _startingPath = GlobalSettings::get().getValue("dialogs.open image.starting path", path.string());
}

void ImageToPatternDialog::shutdown()
{
    GlobalSettings::get().setValue("dialogs.open image.starting path", _startingPath);
}

namespace
{
    void getMatchedCellColor(ImColor const& color, ColorVector<FloatColorRGB> const& customizationColors, int& matchedCellColor, float& matchedCellIntensity)
    {
        using Color = std::array<float, 3>;
        std::vector<Color> cellColors;
        auto toHsv = [](uint32_t color) {
            float h, s, v;
            AlienGui::ConvertRGBtoHSV(color, h, s, v);
            return Color{h, s, v};
        };
        for (int i = 0; i < MAX_COLORS; ++i) {
            cellColors.emplace_back(toHsv(customizationColors.values[i].toRgbColor()));
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

void ImageToPatternDialog::show()
{
    GenericFileDialog::get().showOpenFileDialog("Open image", "Image (*.png){.png},.*", _startingPath, [&](std::filesystem::path const& path) {
        auto const& customizationColors = _SimulationFacade::get()->getSimulationParameters().customizationColors.value;
        auto firstFilename = ifd::FileDialog::Instance().GetResult();
        auto firstFilenameCopy = firstFilename;
        _startingPath = firstFilenameCopy.remove_filename().string();

        int width, height, nrChannels;
        unsigned char* dataImage = stbi_load(firstFilename.string().c_str(), &width, &height, &nrChannels, 0);

        Desc dataDesc;
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
                    getMatchedCellColor(ImColor(r, g, b, 255), customizationColors, matchedCellColor, matchedCellIntensity);
                    dataDesc._objects.emplace_back(ObjectDesc()
                                                       .id(NumberGenerator::get().createEntityId())
                                                       .pos({toFloat(x) + xOffset, toFloat(y)})
                                                       .color(matchedCellColor)
                                                       .isStatic(false)
                                                       .type(SolidDesc()));
                }
            }
        }

        DescEditService::get().reconnectObjects(dataDesc, 1 * 1.5f);
        DescEditService::get().setCenter(dataDesc, Viewport::get().getCenterInWorldPos());

        _SimulationFacade::get()->addAndSelectSimulationData(std::move(dataDesc));
        //TODO: update pattern editor
    });
}
