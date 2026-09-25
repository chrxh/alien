#include "MultiplierService.h"

#include "DescEditService.h"
#include "NumberGenerator.h"
#include "SimulationFacade.h"
#include "SpaceCalculator.h"

namespace
{
    void replaceSelection(ContentDesc&& content)
    {
        _SimulationFacade::get()->removeSelectedObjects(true);
        _SimulationFacade::get()->addAndSelectSimulationData(std::move(content));
    }

    ContentDesc calcGridMultiplication(ContentDesc const& input, MultiplierService::GridParameters const& parameters)
    {
        auto const& editService = DescEditService::get();
        ContentDesc result;
        auto clone = input;
        auto cloneTemplate = input;
        for (int i = 0; i < parameters._horizontalNumber; ++i) {
            for (int j = 0; j < parameters._verticalNumber; ++j) {
                auto templateData = [&] {
                    if (i == 0 && j == 0) {
                        return clone;
                    }
                    return cloneTemplate;
                }();
                editService.shift(templateData, {i * parameters._horizontalDistance, j * parameters._verticalDistance});
                editService.rotate(templateData, i * parameters._horizontalAngleInc + j * parameters._verticalAngleInc);
                editService.accelerate(
                    templateData,
                    {i * parameters._horizontalVelXinc + j * parameters._verticalVelXinc, i * parameters._horizontalVelYinc + j * parameters._verticalVelYinc},
                    i * parameters._horizontalAngularVelInc + j * parameters._verticalAngularVelInc);

                result.add(std::move(templateData));
            }
        }

        return result;
    }
}

MultiplierService::Result MultiplierService::multiplyInGrid(GridParameters const& parameters) const
{
    auto origSelection = _SimulationFacade::get()->getSelectedSimulationData(true);
    replaceSelection(calcGridMultiplication(origSelection, parameters));
    return {.origSelection = std::move(origSelection)};
}

namespace
{
    struct RandomMultiplication
    {
        ContentDesc content;
        bool overlappingCheckSuccessful = true;
    };

    RandomMultiplication calcRandomMultiplication(ContentDesc const& input, MultiplierService::RandomParameters const& parameters, IntVector2D const& worldSize)
    {
        auto const& editService = DescEditService::get();
        auto overlappingCheckSuccessful = true;
        SpaceCalculator spaceCalculator(worldSize);
        DescEditService::Occupancy cellPosBySlot;

        // Create map for overlapping check
        if (parameters._overlappingCheck) {
            for (auto const& object : input._objects) {
                auto intPos = toIntVector2D(spaceCalculator.getCorrectedPosition(object._pos));
                cellPosBySlot[intPos].emplace_back(object._pos);
            }
        }

        // Do multiplication
        ContentDesc result = input;
        auto& numberGen = NumberGenerator::get();
        for (int i = 0; i < parameters._number; ++i) {
            bool overlapping = false;
            ContentDesc copy;
            int attempts = 0;
            do {
                copy = input;
                editService.shift(copy, {toFloat(numberGen.getRandomDouble(0, toInt(worldSize.x))), toFloat(numberGen.getRandomDouble(0, toInt(worldSize.y)))});
                editService.rotate(copy, toInt(numberGen.getRandomDouble(parameters._minAngle, parameters._maxAngle)));
                editService.accelerate(
                    copy,
                    {toFloat(numberGen.getRandomDouble(parameters._minVelX, parameters._maxVelX)),
                     toFloat(numberGen.getRandomDouble(parameters._minVelY, parameters._maxVelY))},
                    toFloat(numberGen.getRandomDouble(parameters._minAngularVel, parameters._maxAngularVel)));

                // Overlapping check
                overlapping = false;
                if (parameters._overlappingCheck) {
                    for (auto const& object : copy._objects) {
                        auto pos = spaceCalculator.getCorrectedPosition(object._pos);
                        if (editService.isCellPresent(cellPosBySlot, spaceCalculator, pos, 2.0f)) {
                            overlapping = true;
                        }
                    }
                }
                ++attempts;
            } while (overlapping && attempts < 200 && overlappingCheckSuccessful);
            if (attempts == 200) {
                overlappingCheckSuccessful = false;
            }

            if (parameters._overlappingCheck) {
                for (auto const& object : copy._objects) {
                    auto intPos = toIntVector2D(spaceCalculator.getCorrectedPosition(object._pos));
                    cellPosBySlot[intPos].emplace_back(object._pos);
                }
            }

            result.add(std::move(copy));
        }

        return {.content = std::move(result), .overlappingCheckSuccessful = overlappingCheckSuccessful};
    }
}

MultiplierService::Result MultiplierService::multiplyRandomly(RandomParameters const& parameters) const
{
    auto origSelection = _SimulationFacade::get()->getSelectedSimulationData(true);
    auto multiplication = calcRandomMultiplication(origSelection, parameters, _SimulationFacade::get()->getWorldSize());
    replaceSelection(std::move(multiplication.content));
    return {.origSelection = std::move(origSelection), .overlappingCheckSuccessful = multiplication.overlappingCheckSuccessful};
}

void MultiplierService::undo(ContentDesc const& origSelection) const
{
    replaceSelection(ContentDesc(origSelection));
}
