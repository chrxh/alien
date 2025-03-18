#pragma once

#include <map>
#include <string>
#include <vector>

#include "Base/Definitions.h"

enum class ParameterType
{
    Float,
    ColorVector
};

enum class LocationType
{
    Base,
    Zone,
    RadiationSource
};

struct BaseParameterDescription
{
    MEMBER_DECLARATION(BaseParameterDescription, ParameterType, type, ParameterType::ColorVector);
    MEMBER_DECLARATION(BaseParameterDescription, int, offset, 0);
    MEMBER_DECLARATION(BaseParameterDescription, std::string, name, std::string());
    MEMBER_DECLARATION(BaseParameterDescription, std::string, group, std::string());
};

class SimulationParametersDescription
{
public:
    void setGroup(std::string const& name);

    struct AddParameters
    {
        MEMBER_DECLARATION(AddParameters, LocationType, locationType, LocationType::Base);
        MEMBER_DECLARATION(AddParameters, ParameterType, type, ParameterType::ColorVector);
        MEMBER_DECLARATION(AddParameters, int, offset, 0);
        MEMBER_DECLARATION(AddParameters, std::string, name, std::string());
    };
    void add(AddParameters const& parameters);

    std::vector<BaseParameterDescription> getEntriesForBase() const;
    //std::vector<Entry> getEntriesForZones() const;
    //std::vector<Entry> getEntriesForParticleSources() const;

private:
    std::optional<std::string> _currentGroup;

    struct Entry
    {
        LocationType locationType = LocationType::Base;
        ParameterType type = ParameterType::ColorVector;
        int offset = 0;
        std::string name;
        std::string group;
    };
    std::vector<Entry> _entries;
};
