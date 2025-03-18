#include "SimulationParametersDescription.h"

void SimulationParametersDescription::setGroup(std::string const& name)
{
    _currentGroup = name;
}

void SimulationParametersDescription::add(AddParameters const& parameters)
{
    _entries.emplace_back(parameters._locationType, parameters._type, parameters._offset, parameters._name, *_currentGroup);
}

std::vector<BaseParameterDescription> SimulationParametersDescription::getEntriesForBase() const
{
    std::vector<BaseParameterDescription> result;

    for (auto const& entry : _entries) {
        if (entry.locationType == LocationType::Base) {
            result.emplace_back(BaseParameterDescription().type(entry.type).offset(entry.offset).name(entry.name).group(entry.group));
        }
    }
    return result;
}
