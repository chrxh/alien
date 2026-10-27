#include "SmoothCurve.h"

#include <algorithm>
#include <cmath>
#include <iterator>

#include <Base/Definitions.h>

namespace
{
    // The kernel is negligible beyond this many standard deviations
    auto constexpr KernelRange = 8.0;

    auto constexpr InvSqrt2Pi = 0.398942280401432678;
    auto constexpr InvSqrt2 = 0.707106781186547524;

    // Smoothing the function max(t, 0) changes its value at t = distance by this amount
    double calcKinkCorrection(double distance, double smoothing)
    {
        auto absDistance = std::abs(distance);
        auto z = absDistance / smoothing;
        return smoothing * InvSqrt2Pi * std::exp(-0.5 * z * z) - absDistance * 0.5 * std::erfc(z * InvSqrt2);
    }
}

SmoothCurve::SmoothCurve(std::vector<Keyframe> keyframes, double smoothing)
    : _keyframes(std::move(keyframes))
    , _smoothing(std::max(0.0, smoothing))
{
    auto numChannels = _keyframes.front().values.size();
    for (size_t index = 1; index + 1 < _keyframes.size(); ++index) {
        auto const& previous = _keyframes.at(index - 1);
        auto const& current = _keyframes.at(index);
        auto const& next = _keyframes.at(index + 1);

        Kink kink{.time = current.time};
        for (size_t channel = 0; channel < numChannels; ++channel) {
            auto slopeBefore = (current.values.at(channel) - previous.values.at(channel)) / (current.time - previous.time);
            auto slopeAfter = (next.values.at(channel) - current.values.at(channel)) / (next.time - current.time);
            kink.slopeChanges.emplace_back(slopeAfter - slopeBefore);
        }
        _kinks.emplace_back(kink);
    }
}

double SmoothCurve::getStartTime() const
{
    return _keyframes.front().time;
}

double SmoothCurve::getEndTime() const
{
    return _keyframes.back().time;
}

std::vector<double> SmoothCurve::getValues(double time) const
{
    auto numChannels = toInt(_keyframes.front().values.size());

    std::vector<double> result;
    for (int channel = 0; channel < numChannels; ++channel) {
        result.emplace_back(getLinearValue(channel, time));
    }
    if (_smoothing <= 0 || _kinks.empty()) {
        return result;
    }

    // The smoothed curve deviates from the linear interpolation only around its kinks, i.e. the interior keyframes and their mirror images,
    // which repeat with the period of the continuation
    auto startTime = getStartTime();
    auto period = 2 * (getEndTime() - startTime);
    auto range = KernelRange * _smoothing;
    auto addCorrections = [&](double kinkTime, std::vector<double> const& slopeChanges, double sign) {
        auto firstPeriod = std::ceil((time - range - kinkTime) / period);
        auto lastPeriod = std::floor((time + range - kinkTime) / period);
        for (auto periodIndex = firstPeriod; periodIndex <= lastPeriod; ++periodIndex) {
            auto correction = sign * calcKinkCorrection(time - kinkTime - periodIndex * period, _smoothing);
            for (int channel = 0; channel < numChannels; ++channel) {
                result.at(channel) += slopeChanges.at(channel) * correction;
            }
        }
    };
    for (auto const& kink : _kinks) {
        addCorrections(kink.time, kink.slopeChanges, 1.0);
        addCorrections(2 * startTime - kink.time, kink.slopeChanges, -1.0);
    }
    return result;
}

double SmoothCurve::getLinearValue(int channel, double time) const
{
    auto const& first = _keyframes.front();
    auto const& last = _keyframes.back();
    auto duration = last.time - first.time;
    if (duration <= 0) {
        return first.values.at(channel);
    }

    // Mirroring at both ends results in a continuation that repeats with twice the duration, each repetition shifted by twice the change from
    // the first to the last keyframe
    auto period = 2 * duration;
    auto numPeriods = std::floor((time - first.time) / period);
    auto timeInPeriod = time - first.time - numPeriods * period;
    auto value = timeInPeriod <= duration ? getLinearValueWithinKeyframes(channel, first.time + timeInPeriod)
                                          : 2 * last.values.at(channel) - getLinearValueWithinKeyframes(channel, first.time + period - timeInPeriod);
    return value + 2 * numPeriods * (last.values.at(channel) - first.values.at(channel));
}

double SmoothCurve::getLinearValueWithinKeyframes(int channel, double time) const
{
    auto next = std::ranges::upper_bound(_keyframes, time, {}, &Keyframe::time);
    if (next == _keyframes.begin()) {
        return next->values.at(channel);
    }
    if (next == _keyframes.end()) {
        return _keyframes.back().values.at(channel);
    }
    auto previous = std::prev(next);
    auto fraction = (time - previous->time) / (next->time - previous->time);
    return previous->values.at(channel) + (next->values.at(channel) - previous->values.at(channel)) * fraction;
}
