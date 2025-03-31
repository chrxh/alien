#pragma once

#include "Base/Singleton.h"

#include "LocationHelper.h"
#include "SimulationParameters.h"
#include "SimulationParametersSpecification.h"
#include "SimulationParametersTypes.h"

class SimulationParametersSpecificationService
{
    MAKE_SINGLETON(SimulationParametersSpecificationService);

public:
    ParametersSpec const& getSpec();

    bool* getBoolRef(BoolMemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const;
    int* getIntRef(IntMemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const;
    float* getFloatRef(FloatMemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const;
    char* getChar64Ref(Char64MemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const;
    int* getAlternativeRef(AlternativeMemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const;
    FloatColorRGB* getFloatColorRGBRef(ColorPickerMemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const;
    ColorTransitionRules* getColorTransitionRulesRef(ColorTransitionRulesMemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const;

    //template <typename T>
    //T* getValueRef(ValueSpec const& spec, SimulationParameters& parameters, int locationIndex) const;
    bool* getPinnedValueRef(ValueSpec const& spec, SimulationParameters& parameters, int locationIndex) const;
    bool* getEnabledValueRef(ValueSpec const& spec, SimulationParameters& parameters, int locationIndex) const;
    bool* getExpertToggleValueRef(ParameterGroupSpec const& spec, SimulationParameters& parameters) const;

private:
    void createSpec();
    std::optional<ParametersSpec> _parametersSpec;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

//template <typename T>
//T* SimulationParametersSpecificationService::getValueRef(ValueSpec const& spec, SimulationParameters& parameters, int locationIndex) const
//{
//    if (std::holds_alternative<BaseValueSpec>(spec)) {
//        auto baseValueSpec = std::get<BaseValueSpec>(spec);
//        if (baseValueSpec._valueAddress.has_value()) {
//            return reinterpret_cast<T*>(reinterpret_cast<char*>(&parameters) + baseValueSpec._valueAddress.value());
//        }
//    } else if (std::holds_alternative<BaseZoneValueSpec>(spec)) {
//        auto baseZoneValueSpec = std::get<BaseZoneValueSpec>(spec);
//
//        if (locationIndex == 0 && baseZoneValueSpec._valueAddress.has_value()) {
//            return reinterpret_cast<T*>(reinterpret_cast<char*>(&parameters.baseValues) + baseZoneValueSpec._valueAddress.value());
//        }
//        for (int i = 0; i < parameters.numZones; ++i) {
//            if (parameters.zone[i].locationIndex == locationIndex && baseZoneValueSpec._valueAddress.has_value()) {
//                return reinterpret_cast<T*>(reinterpret_cast<char*>(&parameters.zone[i].values) + baseZoneValueSpec._valueAddress.value());
//            }
//        }
//    }
//    return nullptr;
//}
