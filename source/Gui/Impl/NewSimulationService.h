#pragma once

#include <optional>
#include <string>

#include <Base/Definitions.h>
#include <Base/Macros.h>
#include <Base/Singleton.h>

#include <Data/Descs.h>

class NewSimulationService
{
    MAKE_SINGLETON(NewSimulationService);

public:
    struct Parameters
    {
        MEMBER(Parameters, std::string, projectName, "");
        MEMBER(Parameters, IntVector2D, worldSize, IntVector2D());
        MEMBER(Parameters, float, externalEnergy, 0.0f);
        MEMBER(Parameters, bool, adoptSimulationParameters, true);
    };
    void createSimulation(Parameters const& parameters);

    std::optional<std::string> loadSimulation(
        SimulationDesc const& simulation,
        std::optional<std::string> const& originResourceId = std::nullopt,
        std::optional<std::string> const& nameInLibrary = std::nullopt);

    std::optional<std::string> const& getOriginResourceId() const;
    void setOriginResourceId(std::optional<std::string> const& value);

    // Name under which the running simulation has been loaded from or saved to the simulations of this computer
    std::optional<std::string> const& getNameInLibrary() const;
    void setNameInLibrary(std::optional<std::string> const& value);

    void storeAutosaveOrigin() const;
    void restoreAutosaveOrigin();

private:
    std::optional<std::string> _originResourceId;
    std::optional<std::string> _nameInLibrary;
};
