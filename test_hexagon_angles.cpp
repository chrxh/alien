#include "source/EngineInterface/ShapeGenerator.h"
#include <iostream>
#include <iomanip>

int main() {
    ShapeGenerator gen;

    for (int i = 0; i < 300; ++i) {
        auto result = gen.generateNextConstructionData(ConstructorShape_Hexagon);

        std::cout << "n=" << i << ": ";

        bool hasAnyAngle = false;
        for (int j = 0; j < 3; ++j) {
            if (result.requiredNodeId[j] != -1) {
                if (hasAnyAngle) std::cout << ", ";
                std::cout << std::fixed << std::setprecision(6) << result.requiredNodeAngle2[j];
                hasAnyAngle = true;
            }
        }

        if (!hasAnyAngle) {
            std::cout << "-";
        }

        std::cout << std::endl;
    }

    return 0;
}
