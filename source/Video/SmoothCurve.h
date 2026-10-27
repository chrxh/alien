#pragma once

#include <vector>

// Interpolates keyframes linearly and smooths the result with a Gaussian kernel. Before the first and after the last keyframe, the linear
// interpolation continues point-symmetrically to them, so that the smoothed curve still starts and ends exactly at these keyframes.
class SmoothCurve
{
public:
    struct Keyframe
    {
        double time = 0;
        std::vector<double> values;
    };

    // The keyframes need strictly increasing times and equally many values. The smoothing is the standard deviation of the kernel in units of
    // time, zero leaves the linear interpolation as it is.
    SmoothCurve(std::vector<Keyframe> keyframes, double smoothing);

    double getStartTime() const;
    double getEndTime() const;

    std::vector<double> getValues(double time) const;

private:
    // Includes the continuation beyond the first and the last keyframe
    double getLinearValue(int channel, double time) const;
    double getLinearValueWithinKeyframes(int channel, double time) const;

    std::vector<Keyframe> _keyframes;
    double _smoothing = 0;

    // Interior keyframes where the slope of the linear interpolation changes. Their mirror images under the point-symmetric continuation are
    // derived from them.
    struct Kink
    {
        double time = 0;
        std::vector<double> slopeChanges;
    };
    std::vector<Kink> _kinks;
};
