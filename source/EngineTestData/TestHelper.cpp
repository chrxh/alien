#include "TestHelper.h"

bool TestHelper::approxCompare(double expected, double actual, float precision /* = 0.001f*/)
{
    return approxCompare(toFloat(expected), toFloat(actual), precision);
}

bool TestHelper::approxCompare(float expected, float actual, float precision /* = 0.001f*/)
{
    auto absNorm = std::abs(expected) + std::abs(actual);
    if (absNorm < precision) {
        return true;
    }
    return std::abs(expected - actual) / absNorm < precision;
}

bool TestHelper::approxCompare(RealVector2D const& expected, RealVector2D const& actual, float precision /* = 0.001f*/)
{
    return approxCompare(expected.x, actual.x, precision) && approxCompare(expected.y, actual.y, precision);
}

bool TestHelper::approxCompare(std::vector<float> const& expected, std::vector<float> const& actual, float precision /* = 0.001f*/)
{
    if (expected.size() != actual.size()) {
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!approxCompare(expected[i], actual[i], precision)) {
            return false;
        }
    }
    return true;
}

bool TestHelper::approxCompareAngles(float expected, float actual, float precision /* = 0.001f*/)
{
    return approxCompare(Math::getNormalizedAngle(expected - actual, -180.0f), 0.0f, precision);
}

bool TestHelper::compare(Description left, Description right)
{
    std::sort(left._cells.begin(), left._cells.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    std::sort(right._cells.begin(), right._cells.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    std::sort(left._particles.begin(), left._particles.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    std::sort(right._particles.begin(), right._particles.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    std::sort(left._creatures.begin(), left._creatures.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    std::sort(right._creatures.begin(), right._creatures.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    std::sort(left._genomes.begin(), left._genomes.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    std::sort(right._genomes.begin(), right._genomes.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    return left == right;
}

bool TestHelper::compare(CellDescription left, CellDescription right)
{
    left._id = 0;
    right._id = 0;
    return left == right;
}

bool TestHelper::compare(ParticleDescription left, ParticleDescription right)
{
    left._id = 0;
    right._id = 0;
    return left == right;
}
