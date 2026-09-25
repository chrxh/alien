#include "MultiplierService.h"

#include "DescEditService.h"
#include "NumberGenerator.h"
#include "SpaceCalculator.h"

ContentDesc MultiplierService::multiplyInGrid(ContentDesc const& content, GridParameters const& parameters) const
{
    auto const& editService = DescEditService::get();
    ContentDesc result;
    for (int i = 0; i < parameters._horizontalNumber; ++i) {
        for (int j = 0; j < parameters._verticalNumber; ++j) {
            auto copy = content;
            editService.shift(copy, {i * parameters._horizontalDistance, j * parameters._verticalDistance});
            editService.rotate(copy, i * parameters._horizontalAngleInc + j * parameters._verticalAngleInc);
            editService.accelerate(
                copy,
                {i * parameters._horizontalVelXinc + j * parameters._verticalVelXinc, i * parameters._horizontalVelYinc + j * parameters._verticalVelYinc},
                i * parameters._horizontalAngularVelInc + j * parameters._verticalAngularVelInc);

            result.add(std::move(copy));
        }
    }

    return result;
}

MultiplierService::RandomMultiplicationResult MultiplierService::multiplyRandomly(ContentDesc const& content, RandomParameters const& parameters) const
{
    auto const& editService = DescEditService::get();
    auto overlappingCheckSuccessful = true;
    SpaceCalculator spaceCalculator(parameters._maxDelta);
    DescEditService::Occupancy cellPosBySlot;

    // Create map for overlapping check
    if (parameters._overlappingCheck) {
        for (auto const& object : content._objects) {
            auto intPos = toIntVector2D(spaceCalculator.getCorrectedPosition(object._pos));
            cellPosBySlot[intPos].emplace_back(object._pos);
        }
    }

    // Do multiplication
    ContentDesc result = content;
    auto& numberGen = NumberGenerator::get();
    for (int i = 0; i < parameters._number; ++i) {
        bool overlapping = false;
        ContentDesc copy;
        int attempts = 0;
        do {
            copy = content;
            editService.shift(
                copy, {toFloat(numberGen.getRandomDouble(0, parameters._maxDelta.x)), toFloat(numberGen.getRandomDouble(0, parameters._maxDelta.y))});
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
