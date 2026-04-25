#include "source/EngineInterface/ShapeGenerator.h"
#include <iostream>
#include <iomanip>

int main() {
    ShapeGenerator gen;

    for (int i = 0; i < 50; ++i) {
        auto result = gen.generateNextConstructionData(ConstructorShape_Hexagon);

        std::cout << "n=" << i << ": angle1=[";

        for (int j = 0; j < 3; ++j) {
            if (result.requiredNodeId[j] != -1) {
                if (j > 0 && result.requiredNodeId[j-1] != -1) std::cout << ", ";
                std::cout << std::fixed << std::setprecision(1) << result.requiredNodeAngle1[j];
            }
        }

        std::cout << "] angle2=[";

        for (int j = 0; j < 3; ++j) {
            if (result.requiredNodeId[j] != -1) {
                if (j > 0 && result.requiredNodeId[j-1] != -1) std::cout << ", ";
                std::cout << std::fixed << std::setprecision(1) << result.requiredNodeAngle2[j];
            }
        }

        std::cout << "]" << std::endl;
    }

    return 0;
}
