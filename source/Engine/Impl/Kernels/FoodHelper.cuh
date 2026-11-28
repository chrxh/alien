#pragma once

#include "Entities.cuh"

class FoodHelper
{
public:
    __inline__ __device__ static void mixFood(Cell& cell, float origRawEnergy, float inflow, float inflowShare, uint8_t inflowColor);

private:
    // The food takes the color of other prey only once that prey clearly outweighs the prey of its color
    static int constexpr MaxFoodColorShare = 15;
    static float constexpr FoodColorHysteresis = 1.5f;

    __inline__ __device__ static uint8_t toQuantizedShare(float share, int maxValue, float dither);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

// Mixes an inflow of raw energy, which has its own share of prey in one color, into the food of a cell
__inline__ __device__ void FoodHelper::mixFood(Cell& cell, float origRawEnergy, float inflow, float inflowShare, uint8_t inflowColor)
{
    auto rawEnergy = max(origRawEnergy, 0.0f);
    auto preyEnergy = rawEnergy * toFloat(cell.foodShare) / 255.0f;
    auto inflowPreyEnergy = inflowColor != NoFoodColor ? inflow * inflowShare : 0.0f;

    // Shares are rounded stochastically, so that small inflows into a lot of food still shift them on average
    auto dither = toFloat(((__float_as_uint(origRawEnergy) * 2654435761u ^ __float_as_uint(inflow)) * 2246822519u) >> 8) / 16777216.0f;

    if (inflowPreyEnergy > NEAR_ZERO) {
        auto colorEnergy = cell.foodColor != NoFoodColor ? preyEnergy * toFloat(cell.foodColorShare) / toFloat(MaxFoodColorShare) : 0.0f;
        auto otherEnergy = preyEnergy - colorEnergy;
        if (inflowColor == cell.foodColor) {
            colorEnergy += inflowPreyEnergy;
        } else {
            otherEnergy += inflowPreyEnergy;
        }
        if (otherEnergy > colorEnergy * FoodColorHysteresis) {
            cell.foodColor = inflowColor;
            colorEnergy = otherEnergy;
        }
        cell.foodColorShare = toQuantizedShare(colorEnergy / (preyEnergy + inflowPreyEnergy), MaxFoodColorShare, dither);
    }
    cell.foodShare = toQuantizedShare((preyEnergy + inflowPreyEnergy) / max(rawEnergy + inflow, NEAR_ZERO), 255, dither);
}

__inline__ __device__ uint8_t FoodHelper::toQuantizedShare(float share, int maxValue, float dither)
{
    return static_cast<uint8_t>(min(max(share, 0.0f), 1.0f) * toFloat(maxValue) + dither);
}
